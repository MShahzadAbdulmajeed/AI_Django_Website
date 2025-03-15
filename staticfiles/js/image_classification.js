document.getElementById("uploadForm").addEventListener("submit", function(event) {
    event.preventDefault();
    
    let formData = new FormData(this);
    
    fetch('/image-classification/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.image) {
            let outputImage = document.getElementById("outputImage");
            outputImage.src = "data:image/jpeg;base64," + data.image;
            outputImage.style.display = "block"; // Show the image after processing
        }
    })
    .catch(error => console.error("Error:", error));
});

function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function () {
            const preview = document.getElementById("previewImage");
            preview.src = reader.result;
            preview.style.display = "block";
        };
        reader.readAsDataURL(file);
    }
}

// Drag and Drop functionality
const uploadArea = document.getElementById("uploadArea");
const imageInput = document.getElementById("imageInput");

uploadArea.addEventListener("click", () => imageInput.click());

uploadArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadArea.classList.remove("dragover");

    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = function () {
            const preview = document.getElementById("previewImage");
            preview.src = reader.result;
            preview.style.display = "block";
        };
        reader.readAsDataURL(file);

        // Assign file to input element
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        imageInput.files = dataTransfer.files;
    }
});