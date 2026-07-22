/**
 * Intelligent Wildlife Monitoring System - Main Frontend Interactivity & Visualization Script
 */

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initStatsPolling();
    initImageUploader();
});

// Chart instances container
let speciesChart = null;
let prChart = null;

function initCharts() {
    const speciesCtx = document.getElementById('speciesChart');
    if (speciesCtx) {
        speciesChart = new Chart(speciesCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Elephant', 'Zebra', 'Giraffe', 'Bear', 'Wolf / Canid', 'Bird'],
                datasets: [{
                    data: [12, 18, 8, 5, 9, 22],
                    backgroundColor: [
                        '#00f2fe', '#4facfe', '#10b981', '#f59e0b', '#f43f5e', '#8b5cf6'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } }
                    }
                },
                cutout: '70%'
            }
        });
    }
}

function initStatsPolling() {
    // Poll stats every 3 seconds for dynamic updates
    setInterval(async () => {
        try {
            const resp = await fetch('/api/stats');
            if (!resp.ok) return;
            const data = await resp.json();

            // Update DOM counters
            const uniqueElem = document.getElementById('stat-total-unique');
            if (uniqueElem) uniqueElem.textContent = data.total_unique || 0;

            const intrusionsElem = document.getElementById('stat-intrusions');
            if (intrusionsElem) intrusionsElem.textContent = data.intrusions_count || 0;

            // Update Doughnut Chart if species breakdown present
            if (speciesChart && data.species_counts) {
                const labels = Object.keys(data.species_counts);
                const counts = Object.values(data.species_counts);
                if (labels.length > 0) {
                    speciesChart.data.labels = labels;
                    speciesChart.data.datasets[0].data = counts;
                    speciesChart.update();
                }
            }
        } catch (err) {
            console.log("Stats polling heartbeat error:", err);
        }
    }, 3000);
}

function initImageUploader() {
    const uploadInput = document.getElementById('imageFileInput');
    const uploadForm = document.getElementById('imageUploadForm');
    const resultBox = document.getElementById('imageResultBox');
    const resultImg = document.getElementById('resultAnnotatedImg');

    if (uploadForm && uploadInput) {
        uploadInput.addEventListener('change', async () => {
            if (!uploadInput.files || uploadInput.files.length === 0) return;

            const formData = new FormData();
            formData.append('image_file', uploadInput.files[0]);

            if (resultBox) {
                resultBox.style.display = 'block';
                resultBox.innerHTML = '<div class="alert alert-info">Processing frame with YOLOv11 + DeepSORT...</div>';
            }

            try {
                const resp = await fetch('/upload_image', {
                    method: 'POST',
                    body: formData
                });
                const resData = await resp.json();

                if (resData.status === 'success') {
                    resultBox.innerHTML = `
                        <div style="margin-top: 1rem;">
                            <h4 style="color: var(--accent-emerald); margin-bottom: 0.5rem;">YOLOv11 Detection Output:</h4>
                            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                                Found <strong>${resData.total_detected}</strong> wildlife target(s): 
                                ${resData.species_list.join(', ')}
                            </p>
                            <img src="${resData.result_image}" style="width: 100%; border-radius: 12px; border: 1px solid var(--border-color); margin-top: 10px;" />
                        </div>
                    `;
                } else {
                    resultBox.innerHTML = `<div class="alert alert-danger">Error: ${resData.error}</div>`;
                }
            } catch (err) {
                if (resultBox) resultBox.innerHTML = `<div class="alert alert-danger">Failed to connect to detection server.</div>`;
            }
        });
    }
}
