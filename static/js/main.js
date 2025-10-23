document.addEventListener('DOMContentLoaded', function () {
    const spectrumForm = document.getElementById('spectrumForm'); 
    const generateBtn = document.getElementById('generateBtn');
    const downloadCsvBtn = document.getElementById('downloadCsvBtn');
    const toggleScaleBtn = document.getElementById('toggleScaleBtn');
    const statusMessage = document.getElementById('statusMessage');
    const chartCanvas = document.getElementById('spectrumChart');
    
    const enableDetectorCheckbox = document.getElementById('enable_detector_response');
    const detectorParamsContainer = document.getElementById('detector_params_container');

    let spectrumChart = null;
    let currentChartData = null;
    let isLogScale = false;

    // --- Event Listeners ---
    enableDetectorCheckbox.addEventListener('change', function() {
        detectorParamsContainer.classList.toggle('hidden', !this.checked);
    });

    const voltageSlider = document.getElementById('voltage_kv');
    const voltageValueDisplay = document.getElementById('voltage_kv_val');
    voltageSlider.addEventListener('input', function() {
        voltageValueDisplay.textContent = this.value;
    });


    const chartConfigBase = {
        type: 'line',
        data: {
            labels: [], 
            datasets: [{
                label: 'Spectrum Intensity',
                data: [], 
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1,
                fill: false, 
                pointRadius: 0, 
                borderWidth: 1.5 
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false, 
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Energy (keV)',
                        font: { size: 14, weight: '500' },
                        color: '#4A5568' 
                    },
                    ticks: {
                        maxTicksLimit: 15, 
                        autoSkip: true,
                        color: '#718096' 
                    },
                    grid: {
                        color: '#E2E8F0' 
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Intensity (photons/s/sr/mA/keV)', 
                        font: { size: 14, weight: '500' },
                        color: '#4A5568'
                    },
                    type: 'linear', 
                    ticks: {
                        color: '#718096',
                        callback: function(value, index, values) {
                            if (value === 0) return '0';
                            if (Math.abs(value) < 1e-3 && Math.abs(value) > 0 || Math.abs(value) >= 1e4) {
                                return value.toExponential(1);
                            }
                            return Number(value.toFixed(3)); 
                        }
                    },
                    grid: {
                        color: '#E2E8F0'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: { size: 12 },
                        color: '#4A5568'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0,0,0,0.7)',
                    titleFont: { size: 14, weight: 'bold'},
                    bodyFont: { size: 12 },
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toExponential(3); 
                            }
                            return label;
                        }
                    }
                }
            },
            animation: {
                duration: 200 
            }
        }
    };

    function updateChart(data) {
        currentChartData = data; 
        if (spectrumChart) {
            spectrumChart.destroy();
        }
        const config = JSON.parse(JSON.stringify(chartConfigBase)); 
        config.data.labels = data.labels;
        config.data.datasets[0].data = data.datasets[0].data;
        config.data.datasets[0].label = data.datasets[0].label || 'Spectrum Intensity';
        config.options.scales.y.type = isLogScale ? 'logarithmic' : 'linear';
        
        if (isLogScale) {
            const positiveData = data.datasets[0].data.filter(d => d > 0);
            if (positiveData.length > 0) {
                config.options.scales.y.min = Math.max(1e-9, Math.min(...positiveData) / 10); 
            } else {
                 config.options.scales.y.min = 1e-9; 
            }
        } else {
            config.options.scales.y.min = 0; 
        }

        spectrumChart = new Chart(chartCanvas, config);
    }
    
    function showMessage(message, type = 'info') {
        statusMessage.textContent = message;
        const baseClasses = 'text-center mb-4 h-6 '; // Ensure h-6 is always there for consistent height
        if (type === 'error') {
            statusMessage.className = baseClasses + 'text-red-600';
        } else if (type === 'success') {
            statusMessage.className = baseClasses + 'text-green-600';
        } else { // info
            statusMessage.className = baseClasses + 'text-blue-600';
        }
    }


    generateBtn.addEventListener('click', async function () {
        showMessage('Generating spectrum...', 'info');
        generateBtn.disabled = true;
        downloadCsvBtn.disabled = true;

        const formData = new FormData(spectrumForm);

        try {
            const response = await fetch('/calculate_spectrum', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            if (result.success) {
                updateChart(result.data);
                showMessage('Spectrum generated successfully.', 'success');
            } else {
                showMessage(`Error: ${result.error}`, 'error');
                console.error("Spectrum generation error:", result.error);
            }
        } catch (error) {
            showMessage(`Network or server error: ${error.message}`, 'error');
            console.error("Fetch error:", error);
        } finally {
            generateBtn.disabled = false;
            downloadCsvBtn.disabled = false;
        }
    });

    downloadCsvBtn.addEventListener('click', function () {
        const tempForm = document.createElement('form');
        tempForm.method = 'POST';
        tempForm.action = '/download_csv';
        
        const formData = new FormData(spectrumForm); 
        for (const pair of formData.entries()) {
            const input = document.createElement('input');
            input.type = 'hidden';
            input.name = pair[0];
            input.value = pair[1];
            tempForm.appendChild(input);
        }
        document.body.appendChild(tempForm);
        tempForm.submit();
        document.body.removeChild(tempForm);
    });

    toggleScaleBtn.addEventListener('click', function () {
        if (!spectrumChart || !currentChartData) {
            showMessage('Please generate a spectrum first.', 'error');
            return;
        }
        isLogScale = !isLogScale;
        updateChart(currentChartData); 
        toggleScaleBtn.innerHTML = isLogScale ? 
            '<svg xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" class="h-5 w-5 inline-block mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" /></svg>Set Linear Scale' : 
            '<svg xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" class="h-5 w-5 inline-block mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" /></svg>Set Log Scale';
    });
    
    updateChart({ 
        labels: Array.from({length: 10}, (_, i) => (i * 0.1).toFixed(1)), 
        datasets: [{ data: Array(10).fill(0), label: "No data generated"}] 
    });
    enableDetectorCheckbox.dispatchEvent(new Event('change'));
});

