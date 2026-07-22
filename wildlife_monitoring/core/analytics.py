import cv2
import numpy as np
import time
import pandas as pd
import json
import os

class WildlifeAnalytics:
    """
    Analytics & Spatial Intrusion engine for Intelligent Wildlife Monitoring.
    Features:
    - Persistent species census counting (eliminates duplicate counts via Track IDs)
    - Danger Zone / Virtual Fence polygon intrusion alerts
    - Trajectory polyline rendering and speed vector estimation
    - Exportable CSV and JSON reports
    """
    def __init__(self, danger_zone_poly=None):
        # Dict of seen track_id -> {'class_name': str, 'first_seen': float, 'last_seen': float, 'intruded': bool}
        self.tracked_entities = {}
        self.danger_zone_poly = danger_zone_poly  # List of (x, y) coordinates
        self.intrusion_events = []
        self.class_counts = {}
        self.start_time = time.time()

    def set_danger_zone(self, polygon_coords):
        """
        Set danger zone polygon. Expected: numpy array of shape (N, 1, 2) and dtype int32.
        """
        if polygon_coords is not None and len(polygon_coords) >= 3:
            self.danger_zone_poly = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))

    def process_tracks(self, tracks, frame_width, frame_height):
        """
        Process frame tracks, check polygon intrusions, update persistent count dictionary.
        Returns dict containing current frame active stats & alert logs.
        """
        current_time = time.time()
        frame_alerts = []

        # Convert normalized polygon ratios to frame pixel coordinates if needed
        if self.danger_zone_poly is None and frame_width > 0 and frame_height > 0:
            # Default polygon setup (top right zone)
            default_pts = [
                (int(frame_width * 0.65), int(frame_height * 0.10)),
                (int(frame_width * 0.98), int(frame_height * 0.10)),
                (int(frame_width * 0.98), int(frame_height * 0.85)),
                (int(frame_width * 0.65), int(frame_height * 0.85))
            ]
            self.danger_zone_poly = np.array(default_pts, dtype=np.int32).reshape((-1, 1, 2))

        for trk in tracks:
            tid = trk['track_id']
            cname = trk['class_name']
            bbox = trk['bbox']

            # Centroid calculation
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)

            if tid not in self.tracked_entities:
                self.tracked_entities[tid] = {
                    'class_name': cname,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'intruded': False,
                    'max_confidence': trk['confidence']
                }
                # Increment persistent global census count for this species
                self.class_counts[cname] = self.class_counts.get(cname, 0) + 1
            else:
                self.tracked_entities[tid]['last_seen'] = current_time
                self.tracked_entities[tid]['max_confidence'] = max(
                    self.tracked_entities[tid]['max_confidence'], trk['confidence']
                )

            # Point-in-polygon Virtual Fence Intrusion Check
            if self.danger_zone_poly is not None:
                dist = cv2.pointPolygonTest(self.danger_zone_poly, (float(cx), float(cy)), False)
                if dist >= 0:  # Inside or on polygon boundary
                    if not self.tracked_entities[tid]['intruded']:
                        self.tracked_entities[tid]['intruded'] = True
                        alert_item = {
                            'track_id': tid,
                            'species': cname,
                            'timestamp': time.strftime("%H:%M:%S", time.localtime(current_time)),
                            'message': f"CRITICAL: {cname} (ID #{tid}) entered Danger Zone Boundary!"
                        }
                        self.intrusion_events.append(alert_item)
                        frame_alerts.append(alert_item)

        return {
            'total_unique_animals': len(self.tracked_entities),
            'species_breakdown': self.class_counts,
            'active_tracks_count': len(tracks),
            'recent_alerts': self.intrusion_events[-5:],
            'frame_alerts': frame_alerts
        }

    def annotate_frame(self, frame, tracks, active_stats):
        """
        Draw rich high-aesthetic overlays on video frames:
        - DeepSORT Bounding Boxes with track IDs, species name, confidence
        - Motion Trajectory tails (glowing color lines)
        - Virtual Fence Danger Zone polygon highlight
        - HUD Overlay with census stats & alerts
        """
        annotated = frame.copy()
        h, w, _ = annotated.shape

        # 1. Draw Danger Zone Polygon
        if self.danger_zone_poly is not None:
            # Draw semi-transparent danger zone overlay
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [self.danger_zone_poly], (0, 0, 180))  # Crimson red fill
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
            cv2.polylines(annotated, [self.danger_zone_poly], isClosed=True, color=(0, 50, 255), thickness=3)

            # Zone label
            moments = cv2.moments(self.danger_zone_poly)
            if moments["m00"] != 0:
                mcx = int(moments["m10"] / moments["m00"])
                mcy = int(moments["m01"] / moments["m00"])
                cv2.putText(annotated, "DANGER / VIRTUAL FENCE ZONE", (mcx - 110, mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 2. Draw Active Track Bounding Boxes & Trajectories
        color_palette = [
            (0, 230, 255), (0, 255, 128), (255, 128, 0), (255, 0, 128),
            (128, 0, 255), (0, 128, 255), (255, 255, 0), (0, 255, 255)
        ]

        for trk in tracks:
            tid = trk['track_id']
            cname = trk['class_name']
            conf = trk['confidence']
            bbox = trk['bbox']
            trajectory = trk.get('trajectory', [])

            color = color_palette[tid % len(color_palette)]

            x1, y1, x2, y2 = bbox

            # Draw Motion Trajectory Polyline
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    thickness = int(np.sqrt(float(i + 1)) * 1.5)
                    cv2.line(annotated, trajectory[i - 1], trajectory[i], color, thickness)

            # Draw Bounding Box with rounded corners look
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Header Label Box
            label_text = f"#{tid} {cname} [{conf:.2f}]"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 10)), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 5, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 3. Draw Modern Top HUD (Heads-Up Display) Panel
        hud_bg = annotated.copy()
        cv2.rectangle(hud_bg, (0, 0), (w, 55), (20, 24, 33), -1)
        cv2.addWeighted(hud_bg, 0.75, annotated, 0.25, 0, annotated)

        hud_text = f"YOLOv11+DeepSORT | Active Targets: {len(tracks)} | Total Unique Census: {active_stats['total_unique_animals']}"
        cv2.putText(annotated, hud_text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

        return annotated

    def generate_report_df(self):
        """
        Generate pandas DataFrame of recorded census and spatial intrusion history.
        """
        records = []
        for tid, info in self.tracked_entities.items():
            records.append({
                'Track_ID': tid,
                'Species': info['class_name'],
                'First_Detected': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info['first_seen'])),
                'Last_Detected': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info['last_seen'])),
                'Duration_Sec': round(info['last_seen'] - info['first_seen'], 1),
                'Max_Confidence': info['max_confidence'],
                'Zone_Intrusion_Triggered': info['intruded']
            })
        return pd.DataFrame(records)
