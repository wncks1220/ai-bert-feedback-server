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
    KoSimCSE 기반 + 커스텀 자연스러움 점수:
    - 기본 분포: 0.2 ~ 0.7
    - 문장 특성 반영
    - 랜덤 편차 추가
    """
    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = flu_model(**inputs)

    emb = outputs.last_hidden_state[:, 0, :]
    base_norm = torch.norm(emb).item()

    # 기본 점수 0.2~0.7 범위
    base_score = ((base_norm % 5) / 5)
    base_score = 0.2 + base_score * 0.5

    # 랜덤 편차
    random_noise = random.uniform(-0.08, 0.08)
    score = base_score + random_noise

    # ------------------------------
    # 특정 유형 문장 패널티
    # ------------------------------

    # (1) 너무 짧은 문장
    if len(sentence) <= 2:
        score -= 0.15

    # (2) "ㅋㅋㅋ", "ㅎㅎㅎ", "ㅇㅇ", "ㄱㄱ"
    if re.fullmatch(r"[ㅋㅎㅇㄱ]+", sentence):
        score -= 0.20

    # (3) 욕설 포함
    bad_words = ["병신", "씨발", "좆", "미친", "꺼져", "개새"]
    if any(bad in sentence for bad in bad_words):
        score -= 0.25

    # (4) 자모만 반복
    if re.fullmatch(r"[ㄱ-ㅎㅏ-ㅣ]+", sentence):
        score -= 0.20

    # (5) 마침표 없이 끝나면 약간 낮게
    if not re.search(r"[.!?]$", sentence):
        score -= 0.05

    # ------------------------------
    # 점수 클램프
    # ------------------------------
    score = max(0.0, min(1.0, score))
    score = round(float(score), 4)

    # 별점 라벨
    if score >= 0.65:
        label = "4 stars"
        comment = "문장이 매우 자연스럽습니다."
    elif score >= 0.50:
        label = "3 stars"
        comment = "대체로 자연스럽습니다."
    elif score >= 0.35:
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








