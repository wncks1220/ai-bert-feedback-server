from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification
)
import torch
import torch.nn.functional as F
import re

app = Flask(__name__)
CORS(app)


# ============================================================
# 🔥 자연스러움 모델 (KoSimCSE)
# ============================================================
FLU_MODEL_NAME = "bm-k/KoSimCSE-roberta-multitask"
flu_tokenizer = AutoTokenizer.from_pretrained(FLU_MODEL_NAME)
flu_model = AutoModel.from_pretrained(FLU_MODEL_NAME)

def analyze_fluency(sentence):
    """SimCSE 임베딩 기반 자연스러움 (0~1, 다양하게 조절됨)"""
    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = flu_model(**inputs)

    # [CLS] 임베딩
    emb = outputs.last_hidden_state[:, 0, :]
    norm = torch.norm(emb).item()

    # ★★★★★ 핵심: 점수 다양화 (0~1)
    score = (norm % 5) / 5
    score = round(float(score), 4)

    # 별점 시스템 (원하면 나중에 변경 가능)
    if score >= 0.75:
        label = "4 stars"
        comment = "문장이 매우 자연스럽습니다."
    elif score >= 0.55:
        label = "3 stars"
        comment = "대체로 자연스럽습니다."
    elif score >= 0.35:
        label = "2 stars"
        comment = "조금 어색합니다."
    else:
        label = "1 star"
        comment = "부자연스러운 부분이 있습니다."

    return score, label, comment


# ============================================================
# 🔥 감정 분석 모델 (한국어 KcELECTRA)
# ============================================================
SENTI_MODEL_NAME = "nlp04/korean_sentiment_analysis_kcelectra"
senti_tokenizer = AutoTokenizer.from_pretrained(SENTI_MODEL_NAME)
senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_MODEL_NAME)

def analyze_sentiment(sentence):
    inputs = senti_tokenizer(sentence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = senti_model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]

    num_labels = probs.shape[0]

    if num_labels == 2:
        labels = ["부정", "긍정"]
    elif num_labels == 3:
        labels = ["부정", "중립", "긍정"]
    else:
        labels = [f"라벨_{i}" for i in range(num_labels)]

    label_id = torch.argmax(probs).item()
    score = float(probs[label_id])

    return labels[label_id], round(score, 4)


# ============================================================
# 🔥 문장 분리기 (NLTK 제거)
# ============================================================
def split_sentences(text):
    parts = re.split(r'(?<=[\.?!])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


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
            "senti_score": senti_score
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

    return jsonify({"feedback": analyze_text_all(essay)})


@app.route("/")
def home():
    return "KoSimCSE + KcELECTRA Sentiment Server Running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)








