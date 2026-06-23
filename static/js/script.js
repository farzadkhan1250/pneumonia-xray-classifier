const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const resultBox = document.getElementById("result");

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
        preview.innerHTML = `<img src="${URL.createObjectURL(file)}">`;
    }
});

async function predict() {
    if (!fileInput.files.length) {
        alert("Please choose an image first");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultBox.innerText = "Predicting...";

    try {
        const res = await fetch("/predict", { method: "POST", body: formData });
        const data = await res.json();
        resultBox.innerText = `${data.prediction} (confidence: ${data.confidence})`;
    } catch (err) {
        resultBox.innerText = "Error: could not get prediction";
    }
}