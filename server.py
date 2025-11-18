from transformers import pipeline
from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")

app = Flask(__name__)

analyzer = pipeline("text-classification", model="textattack/roberta-base-CoLA")

@app.route("/analyze", methods=["POST"])
def analyze_text():
    data = request.get_json()
    text = data.get("essay", "")
    sentences = sent_tokenize(text)

    results = []
    for s in sentences:
        out = analyzer(s)[0]
        label = out["label"]
        score = round(out["score"], 2)

        if label == "LABEL_1":
            comment = "문장이 문법적으로 자연스럽습니다."
        else:
            comment = "문장이 비문법적이거나 어색할 수 있습니다."

        results.append({
            "sentence": s,
            "label": label,
            "score": score,
            "comment": comment
        })

    return jsonify({"feedback": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
