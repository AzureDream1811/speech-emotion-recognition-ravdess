async function sendAudio(blob, filename) {
    const formData = new FormData();
    formData.append("file", blob, filename);

    document.getElementById("result").textContent = "Predicting...";
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    document.getElementById("result").textContent = "Emotion: " + data.emotion;
}

document.getElementById("fileInput").onchange = (e) => {
    const file = e.target.files[0];
    if (file) sendAudio(file, file.name);
};