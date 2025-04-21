document.getElementById('upload-form').addEventListener('submit', function(e) {
    e.preventDefault();

    let formData = new FormData();
    let pdfFile = document.getElementById('pdf').files[0];

    // Check if a file is selected
    if (!pdfFile) {
        alert("Please select a PDF file to upload.");
        return;
    }

    // Optional: Check for file size limit (e.g., 10MB)
    let maxSizeMB = 10;
    if (pdfFile.size / 1024 / 1024 > maxSizeMB) {
        alert("File size exceeds " + maxSizeMB + "MB limit. Please upload a smaller file.");
        return;
    }

    formData.append('pdf', pdfFile);
    

    // Show loading spinner and hide the upload section
    document.getElementById('loading').style.display = 'block';
    document.getElementById('upload-section').style.display = 'none';

    let submitButton = document.querySelector('button[type="submit"]');
    submitButton.disabled = true;  // Disable the button to prevent multiple submissions

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading spinner
        document.getElementById('loading').style.display = 'none';
        submitButton.disabled = false;  // Re-enable the button

        // Check for errors
        if (data.error) {
            document.getElementById('result').innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
        } else {
            // Display document type
            document.getElementById('doc-info').innerHTML =
                `<strong>Document Type:</strong> ${data.document_type}`;
            
            // Display match percentage
            document.getElementById('match-percentage').innerHTML =
                `<strong>Match Percentage:</strong> ${data.match_percentage.toFixed(2)}%`;

            // Display summary
            document.getElementById('summary').innerHTML =
                `<strong>Summary:</strong> ${data.summary ? data.summary : 'No summary available.'}`;

            // Display the top keywords
            let keywordList = document.getElementById('keyword-list');
            keywordList.innerHTML = '';  // Clear previous keywords
            if (data.keywords && data.keywords.length > 0) {
                data.keywords.forEach(function(keyword) {
                    let listItem = document.createElement('li');
                    listItem.innerText = keyword;
                    keywordList.appendChild(listItem);
                });
            } else {
                let noKeywords = document.createElement('li');
                noKeywords.innerText = "No keywords found.";
                keywordList.appendChild(noKeywords);
            }

            // Show the analysis result section
            document.getElementById('analysis-result').style.display = 'grid';
        }
    })
    .catch(error => {
        // Hide loading spinner
        document.getElementById('loading').style.display = 'none';
        submitButton.disabled = false;  // Re-enable the button

        document.getElementById('result').innerHTML = `<div class="alert alert-danger">Error occurred while processing the document.</div>`;
        console.error("Error:", error);  // Log the error to the console for debugging
    });
});
