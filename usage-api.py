from flask import Flask, request
from langdetect import detect, DetectorFactory
from transformers import pipeline

sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="src/heBERT_sentiment_analysis",
    tokenizer="src/heBERT_sentiment_analysis",
    return_all_scores=True
)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/predict', methods=['POST'])
def prepare_text():
    request_data = request.get_json()

    sent = request_data['text']

    # Detect language of text
    try:
        DetectorFactory.seed = 0
        lang = detect(sent)
    except:
        return "No text enter", 400

    if lang != "he":
        return "the language is not Hebrew, please type Hebrew language to detect topic.", 400

    if len(sent) < 40:
        return "your text is too small, please type more words to detect", 400

    final_dict = {"dependency": [],
                  "probability": {}}

    label_dict = sentiment_analysis(sent)
    label_dict = sorted(label_dict[0], key=lambda d: d['score'], reverse=True)

    final_dict["dependency"].append(label_dict[0]["label"])

    for prob in label_dict:
        final_dict["probability"][prob["label"]] = float('%.2f' % prob["score"])

    # Return on a JSON format
    return final_dict


@app.route('/check', methods=['GET'])
def check():
    return "every things right! "


if __name__ == '__main__':
    app.run(host='0.0.0.0')
