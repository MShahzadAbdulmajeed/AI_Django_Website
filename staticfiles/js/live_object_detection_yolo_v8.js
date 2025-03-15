const video = document.getElementById("video");
        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");
        const processed_img = document.getElementById("processed_img");

        navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
            video.srcObject = stream;
        });

        async function sendFrame() {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append("image", blob, "frame.jpg");

                try {
                    const response = await fetch("/live-object-detection/", {
                        method: "POST",
                        body: formData
                    });

                    const result = await response.json();
                    console.log("Detections:", result.detections);

                    // Display processed image from FastAPI
                    processed_img.src = `data:image/jpeg;base64,${result.image}`;

                } catch (error) {
                    console.error("Failed to fetch:", error);
                }

                setTimeout(sendFrame, 100); // Adjust for performance
            }, "image/jpeg");
        }

        video.addEventListener("play", sendFrame);