from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nltk

# NLTK 다운로드 (punkt)
nltk.download("punkt")
nltk.download("punkt_tab")

app = Flask(__name__)
CORS(app)

# 🔥 BERT 모델 로드 (CPU 환경에서도 OK)
MODEL_NAME = "textattack/roberta-base-CoLA"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def analyze_sentence(sentence):
    """문장 하나를 BERT로 분석해서 acceptability 점수 반환"""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    score = torch.softmax(outputs.logits, dim=1)[0][1].item()
    return score

def generate_feedback(text):
    """전체 문단을 문장 단위로 분석"""
    sentences = nltk.sent_tokenize(text)
    results = []

    for s in sentences:
        score = analyze_sentence(s)
        feedback = "문장이 자연스럽습니다." if score > 0.5 else "문장이 부자연스럽습니다. 개선이 필요합니다."
        results.append({"sentence": s, "score": score, "feedback": feedback})

    return results

@app.route("/analyze", methods=["POST"])
def analyze_text():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "텍스트가 없습니다."}), 400

    result = generate_feedback(text)
    return jsonify({"result": result})

@app.route("/", methods=["GET"])
def home():
    return "BERT Feedback Server Running!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

