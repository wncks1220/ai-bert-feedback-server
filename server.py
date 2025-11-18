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
FLU_MODEL_NAME = "snunlp/KR-ELECTRA-discriminator"
flu_tokenizer = AutoTokenizer.from_pretrained(FLU_MODEL_NAME)
flu_model = AutoModelForSequenceClassification.from_pretrained(FLU_MODEL_NAME)

def analyze_fluency(sentence):
    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = flu_model(**inputs)
    score = torch.softmax(outputs.logits, dim=1)[0][1].item()

    #새 별점 기준 (자기소개서 최적화)
    if score >= 0.5:
        label = "4"
    elif score >= 0.3:
        label = "3"
    elif score >= 0.1:
        label = "2"
    else:
        label = "1"

    # 코멘트도 조정
    if score >= 0.5:
        comment = "문장이 자연스럽습니다."
    elif score >= 0.3:
        comment = "대체로 자연스럽지만 약간의 개선이 가능합니다."
    elif score >= 0.1:
        comment = "조금 부자연스럽습니다. 개선이 필요합니다."
    else:
        comment = "문장이 부자연스럽습니다. 재작성하는 것이 좋습니다."

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



