import os
import json
from flask import Flask, request, render_template_string
from textblob import TextBlob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import librosa

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

# ----------------------------------------------------------
# TEXT SENTIMENT
# ----------------------------------------------------------
def predict_text_sentiment(text: str):
    if not text or not text.strip():
        return None, None

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        arr = [0.1, 0.1, 0.8]
    elif polarity < -0.1:
        arr = [0.8, 0.1, 0.1]
    else:
        arr = [0.2, 0.7, 0.1]

    return arr, max(arr)


# ----------------------------------------------------------
# FINAL AUDIO SENTIMENT (Librosa-based - NO TF, NO TRANSFORMERS)
# ----------------------------------------------------------
def predict_audio_sentiment(file_path):
    if not file_path:
        return None, None

    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=16000)

        # Extract intensity & pitch
        energy = float(np.mean(np.abs(y)))

        pitch, _ = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitch[pitch > 0]
        pitch_mean = float(np.mean(pitch_vals)) if pitch_vals.size > 0 else 0

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Simple rule-based emotions
        if energy < 0.02 and pitch_mean < 120:
            arr = [0.8, 0.15, 0.05]  # negative
        elif pitch_mean > 180 and energy > 0.05:
            arr = [0.7, 0.2, 0.1]   # angry -> negative
        elif tempo > 120 or pitch_mean > 160:
            arr = [0.1, 0.1, 0.8]   # happy -> positive
        else:
            arr = [0.1, 0.8, 0.1]   # neutral

        return arr, max(arr)

    except Exception as e:
        print("❌ AUDIO ERROR:", e)
        return None, None


# ----------------------------------------------------------
# IMAGE SENTIMENT (your trained CNN)
# ----------------------------------------------------------
NUM_CLASSES = 7
IMG_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

class MediumEmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Load model
IMG_MODEL_PATH = os.path.join(BASE_DIR, "emotion_cnn.pth")
image_device = "cuda" if torch.cuda.is_available() else "cpu"

image_model = MediumEmotionCNN().to(image_device)
IMAGE_MODEL_OK = False

try:
    image_model.load_state_dict(torch.load(IMG_MODEL_PATH, map_location=image_device))
    image_model.eval()
    IMAGE_MODEL_OK = True
    print("🟢 Image CNN loaded successfully!")
except Exception as e:
    print("❌ Image model failed:", e)

img_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image_sentiment(path):
    if not (IMAGE_MODEL_OK and path and os.path.exists(path)):
        return None, None

    try:
        img = Image.open(path).convert("RGB")
        x = img_transform(img).unsqueeze(0).to(image_device)

        with torch.no_grad():
            logits = image_model(x)
            probs7 = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx = {l: i for i, l in enumerate(IMG_LABELS)}
        pos = float(probs7[idx["happy"]] + probs7[idx["surprise"]])
        neu = float(probs7[idx["neutral"]])
        neg = float(probs7[idx["angry"]] + probs7[idx["disgust"]] +
                    probs7[idx["fear"]] + probs7[idx["sad"]])

        return [neg, neu, pos], max([neg, neu, pos])

    except Exception as e:
        print("❌ Image error:", e)
        return None, None


# ----------------------------------------------------------
# FUSION
# ----------------------------------------------------------
def fuse_sentiments(*items):
    probs = [arr for arr, conf in items if arr]
    if not probs:
        return None

    avg = torch.tensor(probs).mean(dim=0).tolist()
    sent = ["negative", "neutral", "positive"][int(np.argmax(avg))]
    emoji = {"negative": "😡", "neutral": "😐", "positive": "😊"}[sent]

    return {"sentiment": sent, "emoji": emoji, "probs": avg}


# ----------------------------------------------------------
# HTML (unchanged)
# ----------------------------------------------------------
HTML = """
<!doctype html>
<html><head>
<meta charset="utf-8" />

<title>🎭 Multimodal Sentiment Analyzer</title>

<style>
body{
    margin:0;
    font-family:Poppins, sans-serif;
    background: linear-gradient(135deg, #161616, #1f0033, #33001a);
    background-size: 200% 200%;
    animation: gradientShift 8s ease infinite;
    color:#f5f5f5;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.wrap{
    max-width:900px;
    margin:40px auto;
    padding:20px;
}

.card{
    background:rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
    border-radius:16px;
    padding:28px;
    box-shadow:0 0 18px rgba(0,0,0,0.5);
    margin-top:22px;
    border:1px solid rgba(255,255,255,0.15);
}

h1{
    text-align:center;
    font-size:36px;
    font-weight:700;
    color:#ffca5a;
    margin-bottom:10px;
    text-shadow:0 0 12px rgba(255, 204, 102,0.4);
}

input,textarea{
    width:100%;
    padding:14px;
    border-radius:12px;
    background:rgba(255,255,255,0.15);
    border:1px solid rgba(255,255,255,0.25);
    color:#fff;
    margin-top:8px;
    margin-bottom:18px;
    outline:none;
    resize:none;
    box-sizing: border-box;
}

.btn{
    width:100%;
    padding:16px;
    border-radius:12px;
    background:linear-gradient(90deg,#ff9933,#ff5500);
    border:0;
    font-weight:bold;
    color:white;
    margin-top:6px;
    cursor:pointer;
    box-shadow:0 0 12px rgba(255,153,51,0.5);
    transition: transform .2s ease;
}

.btn:hover{
    transform:scale(1.04);
}

.preview img,.preview audio{
    margin-top:12px;
    max-width:100%;
    border-radius:12px;
    box-shadow:0 0 14px rgba(255,153,51,0.4);
}

.result-emoji{
    font-size:60px;
    margin-bottom:10px;
    animation: pop 0.7s ease;
}

@keyframes pop {
    0%{transform:scale(0.2);}
    100%{transform:scale(1);}
}

pre{
    background:rgba(0,0,0,0.4);
    padding:16px;
    border-radius:12px;
    color:#7fffd4;
    overflow:auto;
}

label{
    font-size:15px;
    opacity:0.9;
    margin-top:12px;
    display:block;
}
</style>


<script>
function preview(input,id,type){
    let file = input.files[0];
    if(!file) return;
    let url = URL.createObjectURL(file);

    if(type==="img")
        document.getElementById(id).innerHTML = `<img src="${url}">`;
    else
        document.getElementById(id).innerHTML = `<audio controls src="${url}"></audio>`;
}
</script>

</head>
<body>

<div class="wrap">
    <h1>🎯 Multimodal Sentiment Analyzer</h1>

    <form method="POST" enctype="multipart/form-data" class="card">
        <label>Enter Text:</label>
        <textarea name="text" rows="4" placeholder="Write something..."></textarea>

        <label>Upload Face Image:</label>
        <input type="file" name="image" accept="image/*" onchange="preview(this,'imgprev','img')">
        <div class="preview" id="imgprev"></div>

        <label>Upload Audio:</label>
        <input type="file" name="audio" accept="audio/*" onchange="preview(this,'audprev','aud')">
        <div class="preview" id="audprev"></div>

        <button class="btn">🚀 Analyze</button>
    </form>

    {% if result %}
    <div class="card" style="text-align:center;">
        <div class="result-emoji">{{ result['fused']['emoji'] }}</div>
        <h2>{{ result['fused']['sentiment'] | capitalize }}</h2>
    </div>

    <div class="card">
        <pre>{{ result_json }}</pre>
    </div>
    {% endif %}
</div>

</body></html>
"""



# ----------------------------------------------------------
# ROUTE
# ----------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form.get("text", "")

        audio_file = request.files.get("audio")
        image_file = request.files.get("image")

        audio_path = None
        img_path = None

        if audio_file and audio_file.filename:
            audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
            audio_file.save(audio_path)

        if image_file and image_file.filename:
            img_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(img_path)

        t = predict_text_sentiment(text)
        a = predict_audio_sentiment(audio_path)
        i = predict_image_sentiment(img_path)

        fused = fuse_sentiments(t, a, i)

        result = {"text": t, "audio": a, "image": i, "fused": fused}
        result_json = json.dumps(result, indent=2)

        return render_template_string(HTML, result=result, result_json=result_json)

    return render_template_string(HTML)


# ----------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
