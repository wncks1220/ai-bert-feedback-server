from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = Flask(__name__)
CORS(app)

# -----------------------------
# MODEL 1: 자연스러움 평가
# -----------------------------
FLU_MODEL_NAME = "heegyu/korean-sentence-score"
flu_tokenizer = AutoTokenizer.from_pretrained(FLU_MODEL_NAME)
flu_model = AutoModelForSequenceClassification.from_pretrained(FLU_MODEL_NAME)

def analyze_fluency(sentence):
    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = flu_model(**inputs)

    # 모델은 1~5점 회귀 형태로 출력됨 → 그대로 사용
    score = outputs.logits.squeeze().item()

    # 1~5점 → 별 라벨 변환
    if score >= 4.0:
        label = "4 stars"
        comment = "문장이 매우 자연스럽습니다."
    elif score >= 3.0:
        label = "3 stars"
        comment = "대체로 자연스럽습니다."
    elif score >= 2.0:
        label = "2 stars"
        comment = "조금 어색합니다. 개선해보세요."
    else:
        label = "1 star"
        comment = "문장이 많이 부자연스럽습니다."

    return score, label, comment

# -----------------------------
# MODEL 2: 감정 분석
# -----------------------------
SENTI_MODEL_NAME = "klue/roberta-base"
senti_tokenizer = AutoTokenizer.from_pretrained(SENTI_MODEL_NAME)
senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_MODEL_NAME)

def analyze_sentiment(sentence):
    inputs = senti_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = senti_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    label_id = torch.argmax(probs).item()
    score = probs[label_id].item()

    klue_labels = {0: "중립", 1: "부정", 2: "긍정"}

    return klue_labels.get(label_id, "기타"), score

# -----------------------------
# 한국어 문장 분리 — 개선된 버전
# -----------------------------
def split_korean_sentences(text):
    # 마침표, 물음표, 느낌표 기준 분리
    sentences = re.split(r"[\.!\?]\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# -----------------------------
# 전체 분석 함수
# -----------------------------
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
# API 엔드포인트
# -----------------------------
@app.route("/feedback/bert", methods=["POST"])
def bert_feedback():
    data = request.get_json()
    print("📥 받은 값:", data)

    essay = data.get("essay", "").strip()
    if not essay:
        return jsonify({"error": "내용이 비어 있습니다."}), 400

    feedback = analyze_text_all(essay)
    print("🔍 분석 결과:", feedback)

    return jsonify({"feedback": feedback})

@app.route("/")
def home():
    return "Korean Multi-BERT Feedback Server Running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)



