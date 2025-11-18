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
    """SimCSE 임베딩 기반 자연스러움 (0~1)"""
    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = flu_model(**inputs)

    # [CLS] 벡터 취득
    emb = outputs.last_hidden_state[:, 0, :]
    norm = torch.norm(emb).item()

    # tanh로 0~1 스케일링
    score = float(torch.tanh(torch.tensor(norm / 8)))
    score = round(score, 4)

    # 별점 (너가 기준 조정 가능)
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
        comment = "다소 부자연스럽습니다."

    return score, label, comment


# ============================================================
# 🔥 감정 분석 모델 (한국어 Electra 기반)
# ============================================================
SENTI_MODEL_NAME = "nlp04/korean_sentiment_analysis_kcelectra"

senti_tokenizer = AutoTokenizer.from_pretrained(SENTI_MODEL_NAME)
senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_MODEL_NAME)

# 해당 모델은 3개 라벨 사용
# 0 = 부정, 1 = 중립, 2 = 긍정

def analyze_sentiment(sentence):
    inputs = senti_tokenizer(sentence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = senti_model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)[0]

    num_labels = probs.shape[0]

    # 라벨 자동 생성 (모델이 보내주는 logits 개수에 맞춤)
    if num_labels == 2:
        labels = ["부정", "긍정"]
    elif num_labels == 3:
        labels = ["부정", "중립", "긍정"]
    else:
        labels = [f"라벨_{i}" for i in range(num_labels)]

    # argmax가 num_labels보다 클 때 보정
    label_id = torch.argmax(probs).item()
    if label_id >= num_labels:
        label_id = num_labels - 1

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
# 🔥 API 엔드포인트
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
    return "KoSimCSE + Korean KcELECTRA Sentiment Server Running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)








