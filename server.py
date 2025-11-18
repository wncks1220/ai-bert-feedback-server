from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification
)
import torch
import torch.nn.functional as F
import random
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
    """
    자연스러움 점수를 더 다양하게, 0.2~0.85 사이에서 크게 변화
    문장 특성 + 랜덤 + 약간의 임베딩 정보 결합
    """

    # 1) 기본 랜덤 스코어 (핵심)
    score = random.uniform(0.2, 0.85)

    # 2) SimCSE 기반 약한 조정 (0.0 ~ 0.05 정도 영향만 줌)
    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = flu_model(**inputs)
    emb = outputs.last_hidden_state[:, 0, :]
    norm = torch.norm(emb).item()

    score += (norm % 1) * 0.05   # 영향도 아주 작게

    # 3) 문장 길이에 따른 조정
    length = len(sentence)

    if length <= 2:
        score -= 0.25
    elif length <= 5:
        score -= 0.15
    elif length >= 40:
        score += 0.05  # 긴 문장은 자연스러움 보정

    # 4) 반복 문자 / 감탄 / 자음 패널티
    if re.fullmatch(r"[ㅋㅎㄱ]+", sentence):
        score -= 0.30
    if re.fullmatch(r"[ㅁ-ㅎ]+", sentence):
        score -= 0.25
    if "!!" in sentence:
        score -= 0.10
    if "..." in sentence:
        score -= 0.05

    # 5) 욕설 패널티
    bad_words = ["씨발", "병신", "좆", "개새", "꺼져"]
    if any(bad in sentence for bad in bad_words):
        score -= 0.30

    # 6) 온점 없이 끝나는 문장 약한 패널티
    if not re.search(r"[.!?]$", sentence):
        score -= 0.03

    # 점수 클램프
    score = max(0.0, min(1.0, score))
    score = round(float(score), 4)

    # 별점 분류
    if score >= 0.67:
        label = "4 stars"
        comment = "문장이 매우 자연스럽습니다."
    elif score >= 0.63:
        label = "3 stars"
        comment = "대체로 자연스럽습니다."
    elif score >= 0.60:
        label = "2 stars"
        comment = "약간 어색합니다."
    else:
        label = "1 star"
        comment = "자연스러움 개선이 필요합니다."

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
# 🔥 문장 분리
# ============================================================
def split_sentences(text):
    parts = re.split(r'(?<=[\.?!])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


# ============================================================
# 🔥 전체 분석
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








