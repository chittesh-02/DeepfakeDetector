// Add event listener to the upload form
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    // Prevent default form submission
    e.preventDefault();
    
    // Get DOM elements
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const statusSpan = document.getElementById('status');
    const confidenceSpan = document.getElementById('confidence');
    
    // Hide previous results and errors
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
    
    // Validate file input
    if (!fileInput.files.length) {
        showError('Please select an image file');
        return;
    }
    
    // Create FormData object for file upload
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        // Send POST request to the server
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        
        // Parse JSON response
        const data = await response.json();
        
        if (response.ok) {
            // Display results
            statusSpan.textContent = data.is_deepfake ? 'Deepfake Detected' : 'Real Image';
            confidenceSpan.textContent = `${(data.confidence * 100).toFixed(2)}%`;
            resultDiv.classList.remove('hidden');
        } else {
            // Display error message
            showError(data.error || 'An error occurred during analysis');
        }
    } catch (error) {
        // Handle network errors
        showError('Failed to connect to the server');
    }
});

/**
 * Display error message to the user
 * @param {string} message - The error message to display
 */
function showError(message) {
    const errorDiv = document.getElementById('error');
    const errorMessage = errorDiv.querySelector('.error-message');
    errorMessage.textContent = message;
    errorDiv.classList.remove('hidden');
} 