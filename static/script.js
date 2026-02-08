document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('predictionForm');
    const clearBtn = document.getElementById('clearBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    
    const initialState = document.getElementById('initialState');
    const resultsDashboard = document.getElementById('resultsDashboard');
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    // Result Elements
    const predictionText = document.getElementById('predictionText');
    const probabilityText = document.getElementById('probabilityText');
    const riskBadge = document.getElementById('riskBadge');
    const shapPlot = document.getElementById('shapPlot');
    const limePlot = document.getElementById('limePlot');
    const analysisId = document.getElementById('analysisId');

    // Toast
    const errorToastEl = document.getElementById('errorToast');
    const errorToast = new bootstrap.Toast(errorToastEl);
    const errorMessage = document.getElementById('errorMessage');

    // Store latest results for PDF generation
    let currentData = null;

    // --- Event Listeners ---

    // 1. Handle Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        showLoading(true);
        
        // Collect form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Prediction failed');
            }
            
            const result = await response.json();
            currentData = {
                features: data,
                prediction: result.prediction,
                explanations: result.explanations
            };
            
            displayResults(result);
            
        } catch (error) {
            console.error('Error:', error);
            showError(error.message);
        } finally {
            showLoading(false);
        }
    });

    // 2. Handle Clear Button
    clearBtn.addEventListener('click', () => {
        form.reset();
        // Reset range inputs output
        document.querySelectorAll('input[type=range]').forEach(input => {
            input.nextElementSibling.value = input.defaultValue;
        });
        
        // Hide results, show initial state
        resultsDashboard.classList.add('d-none');
        initialState.classList.remove('d-none');
        currentData = null;
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // 3. Handle PDF Download
    downloadBtn.addEventListener('click', async () => {
        if (!currentData) return;
        
        try {
            const originalText = downloadBtn.innerHTML;
            downloadBtn.disabled = true;
            downloadBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Generating...';
            
            const response = await fetch('/download_report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(currentData),
            });
            
            if (!response.ok) {
                throw new Error('Failed to generate report');
            }
            
            // Trigger download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Heart_Disease_Report_${new Date().toISOString().slice(0,19).replace(/[:T]/g, '')}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
            
        } catch (error) {
            console.error('Download Error:', error);
            showError('Failed to download report. Please try again.');
        } finally {
            downloadBtn.disabled = false;
            downloadBtn.innerHTML = '<i class="fas fa-file-pdf me-2"></i> Download Full Report';
        }
    });

    // --- Helper Functions ---

    function showLoading(show) {
        if (show) {
            loadingOverlay.classList.remove('d-none');
        } else {
            loadingOverlay.classList.add('d-none');
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorToast.show();
    }

    function displayResults(data) {
        const pred = data.prediction;
        const exp = data.explanations;
        
        // Hide initial state, show dashboard
        initialState.classList.add('d-none');
        resultsDashboard.classList.remove('d-none');
        
        // Update Prediction Card
        predictionText.textContent = pred.prediction.toUpperCase();
        predictionText.className = `display-6 fw-bold mb-2 text-${pred.risk_class}`;
        
        probabilityText.textContent = `Confidence: ${(pred.probability * 100).toFixed(2)}%`;
        
        // Update Badge
        riskBadge.textContent = pred.risk_level.toUpperCase();
        // Remove all color classes
        riskBadge.classList.remove('text-bg-danger', 'text-bg-warning', 'text-bg-success', 'text-bg-info');
        // Add new class
        riskBadge.classList.add(`text-bg-${pred.risk_class}`);
        
        // Update ID
        analysisId.textContent = `#${Math.random().toString(36).substr(2, 9).toUpperCase()}`;
        
        // Update Images
        if (exp.shap_plot_url) {
            shapPlot.src = exp.shap_plot_url + `?t=${new Date().getTime()}`; // Prevent caching
        }
        
        if (exp.lime_plot_url) {
            limePlot.src = exp.lime_plot_url + `?t=${new Date().getTime()}`;
        }
        
        // Activate SHAP tab by default
        const shapTab = new bootstrap.Tab(document.getElementById('shap-tab'));
        shapTab.show();
        
        // Scroll to results
        // resultsDashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        // Better scroll for mobile
        const yOffset = -80; // Navbar height
        const y = resultsDashboard.getBoundingClientRect().top + window.pageYOffset + yOffset;
        window.scrollTo({top: y, behavior: 'smooth'});
    }
});
