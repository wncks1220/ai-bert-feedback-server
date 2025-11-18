from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)
import torch
import torch.nn.functional as F
import re

app = Flask(__name__)
CORS(app)

# ============================================================
# 🔥 1) 자연스러움 모델 (paust/pko-t5-base-fluency)
# ============================================================
FLU_MODEL_NAME = "paust/pko-t5-base-fluency"

flu_tokenizer = AutoTokenizer.from_pretrained(FLU_MODEL_NAME)
flu_model = AutoModelForSeq2SeqLM.from_pretrained(FLU_MODEL_NAME)


def analyze_fluency(sentence):
    """
    한국어 자연스러움 점수 (1~5점)
    모델이 직접 점수를 생성함.
    """
    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        output = flu_model.generate(
            **inputs,
            max_length=4,
        )

    score_text = flu_tokenizer.decode(output[0], skip_special_tokens=True)

    try:
        score_val = float(score_text)
    except:
        score_val = 2.5  # fallback value

    score_val = max(1.0, min(5.0, score_val))
    score_01 = round(score_val / 5.0, 4)  # 0~1 스케일

    if score_val >= 4.0:
        label = "4 stars"
        comment = "문장이 매우 자연스럽습니다."
    elif score_val >= 3.0:
        label = "3 stars"
        comment = "대체로 자연스럽습니다."
    elif score_val >= 2.0:
        label = "2 stars"
        comment = "조금 어색합니다."
    else:
        label = "1 star"
        comment = "다소 부자연스럽습니다."

    return score_01, label, comment


# ============================================================
# 🔥 2) 감정 분석 모델 (한국어 Electra)
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
# 🔥 문장 분리기
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
    return "T5 Fluency + Korean Electra Sentiment Server Running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)








