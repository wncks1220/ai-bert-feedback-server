from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
import torch

# Flask 기본 설정
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
    score = torch.softmax(outputs.logits, dim=1)[0][1].item()

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

    klue_labels = {
        0: "중립",
        1: "부정",
        2: "긍정"
    }

    label = klue_labels.get(label_id, "기타")

    return label, score


# -----------------------------
# 🔥 문장 분석 통합 함수 (NLTK 제거)
# -----------------------------
def split_korean_sentences(text):
    """NLTK 없이 한국어 문장 분리"""
    # ?, !, . 을 모두 마침표 처리
    tmp = text.replace("?", ".").replace("!", ".")
    sentences = [s.strip() for s in tmp.split(".") if s.strip()]
    return sentences


def analyze_text_all(text):
    sentences = split_korean_sentences(text)
    results = []

    for s in sentences:
        flu_score, flu_label, flu_comment = analyze_fluency(s)
        senti_label, senti_score = analyze_sentiment(s)

        results.append({
            "sentence": s,

            "fluency_score": round(flu_score, 4),
            "fluency_label": flu_label,
            "fluency_comment": flu_comment,

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



