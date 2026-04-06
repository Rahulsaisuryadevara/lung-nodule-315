// main.js: handles file selection, AJAX predict, chart and UI toggles
document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const previewArea = document.getElementById("previewArea");
    const previewImg = document.getElementById("previewImg");
    const predictBtn = document.getElementById("predictBtn");
    const removeBtn = document.getElementById("removeBtn");
    const loading = document.getElementById("loading");
    const resultCard = document.getElementById("resultCard");
    const resultLabel = document.getElementById("resultLabel");
    const confCanvas = document.getElementById("confChart");
    const dropZone = document.getElementById("dropZone");
    let confChart = null;
    let selectedFile = null;

    // File input change handler
    if (fileInput) {
        fileInput.addEventListener("change", function(e) {
            if (e.target.files && e.target.files[0]) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    // Remove image button
    if (removeBtn) {
        removeBtn.addEventListener("click", function() {
            selectedFile = null;
            previewArea.classList.add("hidden");
            resultCard.classList.add("hidden");
            fileInput.value = "";
        });
    }

    // Function to handle file selection
    function handleFileSelection(file) {
        selectedFile = file;
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(selectedFile.type)) {
            alert("Please upload a valid image file (JPEG or PNG)");
            selectedFile = null;
            return;
        }
        // Validate file size (10MB max)
        const maxSize = 10 * 1024 * 1024; // 10MB in bytes
        if (selectedFile.size > maxSize) {
            alert("File size must be less than 10MB");
            selectedFile = null;
            return;
        }
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            previewArea.classList.remove("hidden");
            resultCard.classList.add("hidden");
        };
        reader.readAsDataURL(selectedFile);
    }

    // Predict button click handler
    if (predictBtn) {
        predictBtn.addEventListener("click", async function() {
            console.log("=".repeat(50));
            console.log("PREDICT BUTTON CLICKED");
            console.log("=".repeat(50));
            if (!selectedFile) {
                alert("Please select an image first.");
                console.log("ERROR: No file selected");
                return;
            }
            console.log("Selected file:", selectedFile.name, selectedFile.type, selectedFile.size);
            // Show loading state
            loading.classList.remove("hidden");
            resultCard.classList.add("hidden");
            predictBtn.disabled = true;
            // Create form data
            const formData = new FormData();
            formData.append("file", selectedFile);
            console.log("FormData created, sending request to /predict");
            try {
                // Send prediction request
                console.log("Fetching /predict endpoint...");
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
                console.log("Response received:", response.status, response.statusText);
                // Check if response is ok
                if (!response.ok) {
                    const errorText = await response.text();
                    console.error("Response not OK:", errorText);
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                console.log("Response data:", data);
                // Hide loading
                loading.classList.add("hidden");
                predictBtn.disabled = false;
                // Handle error response
                if (data.error) {
                    console.error("API returned error:", data.error);
                    alert("Error: " + data.error);
                    return;
                }
                // Validate response data
                if (!data.prediction || !data.labels || !data.confidences) {
                    console.error("Invalid response format:", data);
                    alert("Invalid response from server");
                    return;
                }
                // Display results
                console.log("Displaying results:", data.prediction);
                resultLabel.textContent = `Prediction: ${data.prediction}`;
                resultCard.classList.remove("hidden");
                // Render confidence chart
                console.log("Rendering chart with:", data.labels, data.confidences);
                renderChart(data.labels, data.confidences);
                // Update preview image with server image if available
                if (data.img_url) {
                    console.log("Updating preview image:", data.img_url);
                    previewImg.src = data.img_url;
                }
                // Scroll to results
                resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                console.log("Prediction completed successfully!");
            } catch (error) {
                console.error("=".repeat(50));
                console.error("PREDICTION ERROR:", error);
                console.error("Error stack:", error.stack);
                console.error("=".repeat(50));
                alert("Prediction failed. Please check your internet connection and try again.\n\nError: " + error.message);
                loading.classList.add("hidden");
                predictBtn.disabled = false;
            }
        });
    }

    // Function to render confidence chart
    function renderChart(labels, confidences) {
        // Destroy existing chart if it exists
        if (confChart) {
            confChart.destroy();
        }
        // Create new chart
        const ctx = confCanvas.getContext('2d');
        confChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence',
                    data: confidences,
                    backgroundColor: [
                        'rgba(52, 211, 153, 0.8)',  // Green
                        'rgba(96, 165, 250, 0.8)',   // Blue
                        'rgba(248, 113, 113, 0.8)'   // Red
                    ],
                    borderColor: [
                        'rgb(52, 211, 153)',
                        'rgb(96, 165, 250)',
                        'rgb(248, 113, 113)'
                    ],
                    borderWidth: 2,
                    borderRadius: 8,
                    borderSkipped: false,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Confidence Scores',
                        font: {
                            size: 16,
                            weight: 'bold'
                        },
                        color: '#1f2937'
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        padding: 12,
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + (context.parsed.y * 100).toFixed(2) + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            },
                            color: '#6b7280'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#6b7280'
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    // Drag and drop functionality
    if (dropZone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        function highlight(e) {
            dropZone.classList.add('border-blue-500', 'bg-blue-100');
        }
        function unhighlight(e) {
            dropZone.classList.remove('border-blue-500', 'bg-blue-100');
        }
        dropZone.addEventListener('drop', handleDrop, false);
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleFileSelection(files[0]);
            }
        }
    }

    // Handle mobile menu toggle if needed
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mobileMenu = document.getElementById('mobileMenu');
    if (mobileMenuBtn && mobileMenu) {
        mobileMenuBtn.addEventListener('click', function() {
            mobileMenu.classList.toggle('hidden');
        });
    }
});
