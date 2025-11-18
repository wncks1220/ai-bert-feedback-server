from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch
import re

app = Flask(__name__)
CORS(app)


# ============================================================
# 🔥 자연스러움 모델
# ============================================================
FLU_MODEL_NAME = "heegyu/korean-sentence-similarity"
flu_tokenizer = AutoTokenizer.from_pretrained(FLU_MODEL_NAME)
flu_model = AutoModelForSequenceClassification.from_pretrained(FLU_MODEL_NAME)


def analyze_fluency(sentence):
    """문장 자연스러움 점수 0~1"""
    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = flu_model(**inputs)

    score = torch.sigmoid(outputs.logits.squeeze()).item()

    if score >= 0.8:
        label = "4 stars"
        comment = "문장이 매우 자연스럽습니다."
    elif score >= 0.6:
        label = "3 stars"
        comment = "대체로 자연스럽습니다."
    elif score >= 0.4:
        label = "2 stars"
        comment = "조금 어색합니다. 개선해보세요."
    else:
        label = "1 star"
        comment = "문장이 많이 부자연스럽습니다."

    return round(score, 4), label, comment


# ============================================================
# 🔥 감정 모델
# ============================================================
SENTI_MODEL_NAME = "brainbert/korean-sentiment-analysis"
senti_tokenizer = AutoTokenizer.from_pretrained(SENTI_MODEL_NAME)
senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_MODEL_NAME)

SENTI_LABELS = ["부정", "중립", "긍정"]


def analyze_sentiment(sentence):
    inputs = senti_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = senti_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    label_id = torch.argmax(probs).item()
    score = probs[label_id].item()

    return SENTI_LABELS[label_id], round(score, 4)


# ============================================================
# 🔥 NLTK 없이 문장 분리
# ============================================================
def split_sentences(text):
    # 한국어 문장 분리: . ? ! \n 기준
    raw = re.split(r'(?<=[\.\?\!])\s+|\n+', text)
    # 빈 문자열 제거
    return [s.strip() for s in raw if s.strip()]


# ============================================================
# 🔥 전체 문장 분석
# ============================================================
def analyze_text_all(text):
    sentences = split_sentences(text)
    results = []

    for s in sentences:
        flu_score, flu_label, flu_comment = analyze_fluency(s)
        senti_label, senti_score = analyze_sentiment(s)

        results.append({
            "sentence": s,
            "fluency_score": flu_score,
            "fluency_label": flu_label,
            "fluency_comment": flu_comment,
            "senti_label": senti_label,
            "senti_score": senti_score,
        })

    return results


# ============================================================
# 🔥 API
# ============================================================
@app.route("/feedback/bert", methods=["POST"])
def bert_feedback():
    data = request.get_json()
    essay = data.get("essay", "").strip()

    if not essay:
        return jsonify({"error": "내용이 비어 있습니다."}), 400

    feedback = analyze_text_all(essay)
    return jsonify({"feedback": feedback})


@app.route("/")
def home():
    return "Korean Fluency + Sentiment Server Running without NLTK!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)





