from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import re

app = Flask(__name__)
CORS(app)


# ============================================================
#  🔥 자연스러움 모델: cointegrated/LaBSE-en-kr
# ============================================================
FLU_MODEL_NAME = "cointegrated/LaBSE-en-kr"

flu_tokenizer = AutoTokenizer.from_pretrained(FLU_MODEL_NAME)
flu_model = AutoModel.from_pretrained(FLU_MODEL_NAME)


def analyze_fluency(sentence):
    """
    문장 자연스러움 점수 생성 (임베딩 기반)
    결과는 0 ~ 1 범위이며, 너가 원하는 대로 나중에 스케일 조정하면 됨.
    """

    inputs = flu_tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = flu_model(**inputs)

    # CLS 임베딩 가져오기
    emb = outputs.last_hidden_state[:, 0, :]   # (1, hidden)

    # 벡터 L2 norm 기반 naturalness score (0~1)
    score = torch.tanh(emb.norm() / 10).item()  # 스케일 조정 가능
    
    score = round(score, 4)

    # 라벨은 너가 점수 기준 수정할 수 있으므로 기본값만 제공
    if score >= 0.75:
        label = "4 stars"
        comment = "문장이 매우 자연스럽습니다."
    elif score >= 0.55:
        label = "3 stars"
        comment = "대체로 자연스럽습니다."
    elif score >= 0.35:
        label = "2 stars"
        comment = "조금 어색합니다. 개선해보세요."
    else:
        label = "1 star"
        comment = "문장이 많이 부자연스럽습니다."

    return score, label, comment


# ============================================================
#  🔥 감정 분석 모델 (한국어 감정 분류)
# ============================================================
SENTI_MODEL_NAME = "jason9693/ko-sentiment-roberta"

senti_tokenizer = AutoTokenizer.from_pretrained(SENTI_MODEL_NAME)
senti_model = AutoModel.from_pretrained(SENTI_MODEL_NAME)

SENTI_LABELS = ["부정", "중립", "긍정"]


def analyze_sentiment(sentence):
    inputs = senti_tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = senti_model(**inputs)

    emb = outputs.last_hidden_state[:, 0, :]
    logits = torch.nn.Linear(emb.size(-1), 3)(emb)  # 감정 3분류

    probs = F.softmax(logits, dim=1)[0]
    label_id = torch.argmax(probs).item()
    score = probs[label_id].item()

    return SENTI_LABELS[label_id], round(score, 4)


# ============================================================
#  🔥 문장 분리 (NLTK 없이)
# ============================================================
def split_sentences(text):
    raw = re.split(r'(?<=[\.\?\!])\s+|\n+', text)
    return [s.strip() for s in raw if s.strip()]


# ============================================================
#  🔥 전체 분석
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
#  🔥 API
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
    return "LaBSE + Sentiment Server Running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)






