from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
import torch
import nltk

# NLTK
nltk.download("punkt")

app = Flask(__name__)
CORS(app)

# -----------------------------
# 🔥 MODEL 1: 한국어 자연스러움 평가
# -----------------------------
FLU_MODEL_NAME = "snunlp/KR-ELECTRA-discriminator"
flu_tokenizer = AutoTokenizer.from_pretrained(FLU_MODEL_NAME)
flu_model = AutoModelForSequenceClassification.from_pretrained(FLU_MODEL_NAME)


def analyze_fluency(sentence):
    """한국어 자연스러움 평가 (Acceptability Score)"""
    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = flu_model(**inputs)
    score = torch.softmax(outputs.logits, dim=1)[0][1].item()  # 자연스러움 확률
    
    # 점수 → 라벨
    label = (
        "4 stars" if score > 0.8 else
        "3 stars" if score > 0.6 else
        "2 stars" if score > 0.4 else
        "1 star"
    )

    comment = (
        "문장이 자연스럽습니다." if score > 0.7 else
        "조금 부자연스럽습니다. 개선이 필요합니다."
    )

    return score, label, comment


# -----------------------------
# 🔥 MODEL 2: 한국어 감정/문장 분류
# -----------------------------
SENTI_MODEL_NAME = "klue/roberta-base"
senti_tokenizer = AutoTokenizer.from_pretrained(SENTI_MODEL_NAME)
senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_MODEL_NAME)


def analyze_sentiment(sentence):
    """한국어 감정/문장 분류"""
    inputs = senti_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = senti_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    label_id = torch.argmax(probs).item()
    score = probs[label_id].item()

    # KLUE 감정 라벨 (기본: 3개 또는 7개)
    klue_labels = {
        0: "중립",
        1: "부정",
        2: "긍정"
    }

    label = klue_labels.get(label_id, "기타")

    return label, score


# -----------------------------
# 🔥 문장 분석 통합 함수
# -----------------------------
def analyze_text_all(text):
    sentences = nltk.sent_tokenize(text)
    results = []

    for s in sentences:
        # 자연스러움 평가
        flu_score, flu_label, flu_comment = analyze_fluency(s)

        # 감정/문장 분류
        senti_label, senti_score = analyze_sentiment(s)

        results.append({
            "sentence": s,

            # 자연스러움
            "fluency_score": round(flu_score, 4),
            "fluency_label": flu_label,
            "fluency_comment": flu_comment,

            # 감정/문장 분류
            "senti_label": senti_label,
            "senti_score": round(senti_score, 4)
        })

    return results


# -----------------------------
# 🔥 API 엔드포인트
# -----------------------------
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
    return "Korean Multi-BERT Feedback Server Running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


