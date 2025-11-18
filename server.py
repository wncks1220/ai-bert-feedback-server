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
# 🔥 자연스러움(유창성) 모델 — heegyu/KLUE-FLUENCY-v1
# ============================================================
FLU_MODEL_NAME = "heegyu/KLUE-FLUENCY-v1"

flu_tokenizer = AutoTokenizer.from_pretrained(FLU_MODEL_NAME)
flu_model = AutoModelForSeq2SeqLM.from_pretrained(FLU_MODEL_NAME)


def analyze_fluency(sentence):
    """KLUE Fluency 모델: 자연스러움을 0~5 점수로 출력"""
    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        output_ids = flu_model.generate(
            **inputs,
            max_length=4
        )

    # 출력 예: "2.7" 또는 "4.3"
    score_text = flu_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    try:
        raw_score = float(score_text)
    except:
        raw_score = 2.5   # fallback

    # 0~1 스케일
    score01 = max(0.0, min(1.0, raw_score / 5.0))

    # 별점 라벨링 (원하면 조절 가능)
    if raw_score >= 4.0:
        label = "4 stars"
        comment = "문장이 매우 자연스럽습니다."
    elif raw_score >= 3.0:
        label = "3 stars"
        comment = "대체로 자연스럽습니다."
    elif raw_score >= 2.0:
        label = "2 stars"
        comment = "조금 어색합니다."
    else:
        label = "1 star"
        comment = "다소 부자연스럽습니다."

    return round(score01, 4), label, comment


# ============================================================
# 🔥 감정 분석 모델 — nlp04/korean_sentiment_analysis_kcelectra
# ============================================================
SENTI_MODEL_NAME = "nlp04/korean_sentiment_analysis_kcelectra"

senti_tokenizer = AutoTokenizer.from_pretrained(SENTI_MODEL_NAME)
senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_MODEL_NAME)


def analyze_sentiment(sentence):
    inputs = senti_tokenizer(sentence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = senti_model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]

    labels = ["부정", "중립", "긍정"]
    label_id = torch.argmax(probs).item()

    return labels[label_id], float(probs[label_id])


# ============================================================
# 🔥 문장 분리기
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
            "senti_score": round(senti_score, 4)
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
    return "KLUE-FLUENCY + KcELECTRA Sentiment Server Running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)








