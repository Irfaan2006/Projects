import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import logging

logger = logging.getLogger("DeepSORTTracker")

class KalmanFilter:
    """
    Standard 8D Kalman Filter for bounding box state tracking in 2D image coordinates:
    State vector: [x, y, a, h, vx, vy, va, vh]
    where:
      (x, y) is bounding box center position
      a is aspect ratio (width / height)
      h is height
      (vx, vy, va, vh) are respective velocities
    """
    def __init__(self):
        # State transition matrix (8x8)
        self._motion_mat = np.eye(8, 8)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

        # Measurement matrix (4x8): maps state to [x, y, a, h]
        self._update_mat = np.eye(4, 8)

        # Standard deviation parameters
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement):
        """
        Create track state from unobserved measurement [x, y, a, h].
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman Filter prediction step.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.dot(np.dot(self._motion_mat, covariance), self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Run Kalman Filter update step with observed measurement.
        """
        projected_mean = np.dot(self._update_mat, mean)
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        projected_cov = np.dot(np.dot(self._update_mat, covariance), self._update_mat.T) + innovation_cov

        kalman_gain = np.dot(np.dot(covariance, self._update_mat.T), np.linalg.inv(projected_cov))
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.dot(np.dot(kalman_gain, projected_cov), kalman_gain.T)
        return new_mean, new_covariance


class VisualFeatureExtractor:
    """
    Appearance Feature Extractor for DeepSORT Association.
    Extracts normalized 128-dimensional embedding vectors from cropped bounding box patches.
    """
    def __init__(self, feature_dim=128):
        self.feature_dim = feature_dim

    def extract(self, frame, bboxes):
        features = []
        h_img, w_img, _ = frame.shape

        for bbox in bboxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (64, 128))
                # Compute color histogram + spatial gradient feature vector as lightweight robust embedding
                hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
                hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
                grad_x = cv2.Sobel(crop_resized, cv2.CV_32F, 1, 0, ksize=3).mean(axis=(0, 1))
                grad_y = cv2.Sobel(crop_resized, cv2.CV_32F, 0, 1, ksize=3).mean(axis=(0, 1))

                feat = np.concatenate([
                    hist_h.flatten(), hist_s.flatten(), hist_v.flatten(),
                    grad_x.flatten(), grad_y.flatten()
                ])
                # Pad/truncate to feature_dim and L2 normalize
                if len(feat) < self.feature_dim:
                    feat = np.pad(feat, (0, self.feature_dim - len(feat)))
                else:
                    feat = feat[:self.feature_dim]

                norm = np.linalg.norm(feat) + 1e-6
                feat = feat / norm
            else:
                feat = np.random.randn(self.feature_dim)
                feat /= (np.linalg.norm(feat) + 1e-6)

            features.append(feat)
        return np.array(features)


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    Represents a persistent single wildlife target trajectory.
    """
    def __init__(self, mean, covariance, track_id, n_init, max_age, class_name, confidence, feature):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.n_init = n_init
        self.max_age = max_age
        self.class_name = class_name
        self.confidence = confidence
        self.features = [feature]
        self.trajectory = []  # List of center (x, y) tuples

    def to_tlwh(self):
        """
        Convert bounding box state [x, y, a, h] to Top-Left-Width-Height format.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[0] -= ret[2] / 2.0
        ret[1] -= ret[3] / 2.0
        return ret

    def to_tlbr(self):
        """
        Convert bounding box state to [x1, y1, x2, y2].
        """
        ret = self.to_tlwh()
        ret[2] += ret[0]
        ret[3] += ret[1]
        return ret

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection, feature):
        measurement = bbox_to_xyah(detection['bbox'])
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)
        self.features.append(feature)
        if len(self.features) > 50:
            self.features.pop(0)

        self.hits += 1
        self.time_since_update = 0
        self.confidence = detection['confidence']
        self.class_name = detection['class_name']

        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

        # Record trajectory centroid
        tlbr = self.to_tlbr()
        cx = int((tlbr[0] + tlbr[2]) / 2)
        cy = int((tlbr[1] + tlbr[3]) / 2)
        self.trajectory.append((cx, cy))
        if len(self.trajectory) > 40:
            self.trajectory.pop(0)

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted

    def is_confirmed(self):
        return self.state == TrackState.Confirmed


def bbox_to_xyah(bbox):
    """
    Convert [x1, y1, x2, y2] box to [x_center, y_center, aspect_ratio, height].
    """
    x1, y1, x2, y2 = bbox
    w = float(x2 - x1)
    h = float(y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    aspect_ratio = w / (h + 1e-6)
    return np.array([cx, cy, aspect_ratio, h])


class DeepSORTTracker:
    """
    Main DeepSORT Tracker API combining Kalman Filtering, Appearance Distance,
    and Hungarian Assignment algorithm.
    """
    def __init__(self, max_cosine_distance=0.3, max_age=60, n_init=3):
        self.max_cosine_distance = max_cosine_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = KalmanFilter()
        self.feature_extractor = VisualFeatureExtractor()
        self.tracks = []
        self._next_id = 1

    def update(self, detections, frame):
        """
        Update tracker with new frame detections.
        Returns active confirmed tracks: List of dicts with track_id, bbox, class_name, trajectory, etc.
        """
        # 1. Predict track states using Kalman Filter
        for track in self.tracks:
            track.predict(self.kf)

        if len(detections) == 0:
            for track in self.tracks:
                track.mark_missed()
            self.tracks = [t for t in self.tracks if not t.state == TrackState.Deleted]
            return self._get_active_tracks()

        # 2. Extract visual appearance embeddings
        bboxes = [d['bbox'] for d in detections]
        features = self.feature_extractor.extract(frame, bboxes)

        # 3. Associate detections to existing tracks via Cosine Distance + IoU
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        unconfirmed_tracks = [t for t in self.tracks if not t.is_confirmed()]

        matches, unmatched_tracks, unmatched_detections = self._match(confirmed_tracks, detections, features)

        # 4. Update matched confirmed tracks
        for track_idx, det_idx in matches:
            confirmed_tracks[track_idx].update(self.kf, detections[det_idx], features[det_idx])

        # 5. Try matching unmatched detections to unconfirmed tracks via IoU
        iou_matches, unconfirmed_unmatched_tracks, unmatched_detections = self._match_iou(
            unconfirmed_tracks + [confirmed_tracks[i] for i in unmatched_tracks],
            detections,
            unmatched_detections
        )

        for track_idx, det_idx in iou_matches:
            target_track = (unconfirmed_tracks + [confirmed_tracks[i] for i in unmatched_tracks])[track_idx]
            target_track.update(self.kf, detections[det_idx], features[det_idx])

        # Mark missed for unmatched tracks
        for track in self.tracks:
            if track.time_since_update > 0:
                track.mark_missed()

        # 6. Initialize new tracks for remaining unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            feat = features[det_idx]
            measurement = bbox_to_xyah(det['bbox'])
            mean, covariance = self.kf.initiate(measurement)
            new_track = Track(
                mean, covariance, self._next_id, self.n_init, self.max_age,
                det['class_name'], det['confidence'], feat
            )
            self._next_id += 1
            self.tracks.append(new_track)

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.state == TrackState.Deleted]

        return self._get_active_tracks()

    def _match(self, tracks, detections, features):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Build Cosine Distance Matrix (len(tracks) x len(detections))
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            track_feat = track.features[-1]
            for j, feat in enumerate(features):
                # Cosine distance = 1 - dot_product
                cost_matrix[i, j] = 1.0 - np.dot(track_feat, feat)

        # Gating distance
        cost_matrix[cost_matrix > self.max_cosine_distance] = 1e5

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(set(range(len(tracks))) - set(row_ind))
        unmatched_detections = list(set(range(len(detections))) - set(col_ind))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.max_cosine_distance:
                matches.append((r, c))
            else:
                unmatched_tracks.append(r)
                unmatched_detections.append(c)

        return matches, unmatched_tracks, unmatched_detections

    def _match_iou(self, tracks, detections, unmatched_det_indices):
        if len(tracks) == 0 or len(unmatched_det_indices) == 0:
            return [], list(range(len(tracks))), unmatched_det_indices

        cost_matrix = np.zeros((len(tracks), len(unmatched_det_indices)), dtype=np.float32)
        for i, track in enumerate(tracks):
            t_box = track.to_tlbr()
            for j_idx, d_idx in enumerate(unmatched_det_indices):
                d_box = detections[d_idx]['bbox']
                cost_matrix[i, j_idx] = 1.0 - compute_iou(t_box, d_box)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        remaining_dets = list(unmatched_det_indices)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.7:  # IoU distance threshold
                matches.append((r, unmatched_det_indices[c]))
                if unmatched_det_indices[c] in remaining_dets:
                    remaining_dets.remove(unmatched_det_indices[c])

        return matches, [], remaining_dets

    def _get_active_tracks(self):
        active = []
        for track in self.tracks:
            if track.is_confirmed() or track.hits >= 2:
                tlbr = [int(v) for v in track.to_tlbr()]
                active.append({
                    'track_id': track.track_id,
                    'bbox': tlbr,
                    'class_name': track.class_name,
                    'confidence': track.confidence,
                    'trajectory': track.trajectory,
                    'hits': track.hits,
                    'age': track.age
                })
        return active


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou
