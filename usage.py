from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd
# tokenizer = AutoTokenizer.from_pretrained("src/heBERT_sentiment_analysis")  # same as 'avichr/heBERT' tokenizer
# model = AutoModel.from_pretrained("src/heBERT_sentiment_analysis")

sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="src/heBERT_sentiment_analysis",
    tokenizer="src/heBERT_sentiment_analysis",
    return_all_scores=True
)
c = 1
def add_senti(sent):
    global c
    label_dict = sentiment_analysis(sent)
    label_dict = sorted(label_dict[0], key=lambda d: d['score'], reverse=True)
    label = label_dict[0]["label"]
    print(c,"   ", label)
    c += 1

    return label

test_df = pd.read_csv("heb_test2.csv", sep=",", encoding="utf-8",quoting=1)

test_df["new_senti"] = test_df["tweet"].apply(lambda x: add_senti(x))

test_df.to_csv("new_senti_test_full.csv", index=False, encoding="utf-8", quoting=1)
