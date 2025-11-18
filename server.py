from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch
import nltk

# NLTK 다운로드

app = Flask(__name__)
CORS(app)


# ============================================================
# 🔥 1) 자연스러움 모델 - heegyu/korean-sentence-similarity
# ============================================================
FLU_MODEL_NAME = "heegyu/korean-sentence-similarity"
flu_tokenizer = AutoTokenizer.from_pretrained(FLU_MODEL_NAME)
flu_model = AutoModelForSequenceClassification.from_pretrained(FLU_MODEL_NAME)


def analyze_fluency(sentence):
    """문장 자연스러움(유창성) 0~1 점수"""
    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = flu_model(**inputs)

    # similarity 모델 → sigmoid를 통해 0~1 범위
    score = torch.sigmoid(outputs.logits.squeeze()).item()

    # 점수 → 라벨 변환
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
# 🔥 2) 감정 모델 - brainbert/korean-sentiment-analysis
# ============================================================
SENTI_MODEL_NAME = "brainbert/korean-sentiment-analysis"
senti_tokenizer = AutoTokenizer.from_pretrained(SENTI_MODEL_NAME)
senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_MODEL_NAME)

SENTI_LABELS = ["부정", "중립", "긍정"]


def analyze_sentiment(sentence):
    """감정 분석: 부정 / 중립 / 긍정"""
    inputs = senti_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = senti_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    label_id = torch.argmax(probs).item()
    score = probs[label_id].item()

    label = SENTI_LABELS[label_id]

    return label, round(score, 4)


# ============================================================
# 🔥 문장 분석 통합 함수
# ============================================================
def analyze_text_all(text):
    sentences = nltk.sent_tokenize(text)
    results = []

    for s in sentences:

        # 자연스러움
        flu_score, flu_label, flu_comment = analyze_fluency(s)

        # 감정
        senti_label, senti_score = analyze_sentiment(s)

        results.append({
            "sentence": s,

            # 자연스러움
            "fluency_score": flu_score,
            "fluency_label": flu_label,
            "fluency_comment": flu_comment,

            # 감정
            "senti_label": senti_label,
            "senti_score": senti_score,
        })

    return results


# ============================================================
# 🔥 API 엔드포인트
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
    return "Korean Fluency + Sentiment Feedback Server Running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)




