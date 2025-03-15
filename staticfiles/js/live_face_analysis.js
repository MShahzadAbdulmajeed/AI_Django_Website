const video = document.getElementById("video");
        const outputImage = document.getElementById("outputImage");

        // Access the webcam
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => video.srcObject = stream)
            .catch(err => console.error("Webcam access error:", err));

        function captureFrame() {
            let canvas = document.createElement("canvas");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            let ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob(blob => {
                let formData = new FormData();
                formData.append("image", blob, "frame.jpg");

                fetch("/live-face-analysis/", {
                    method: "POST",
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.image) {
                        outputImage.src = "data:image/jpeg;base64," + data.image;
                    }
                })
                .catch(error => console.error("Error:", error));
            }, "image/jpeg");
        }

        // Capture a frame every 2 seconds
        setInterval(captureFrame, 2000);