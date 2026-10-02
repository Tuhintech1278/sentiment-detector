"""
Simple sentiment checker: type a sentence, get POSITIVE / NEGATIVE / NEUTRAL.

Run:  python ask.py
Needs sentiment_classifier.py in the same folder.
Trains a model automatically the first time if none exists.
"""
import __main__

import joblib
from sklearn.linear_model import LogisticRegression

import sentiment_classifier as sc

# lets the saved model load (it was pickled while running as a script)
__main__.TextPreprocessor = sc.TextPreprocessor
__main__._FallbackLemmatizer = sc._FallbackLemmatizer


def get_model():
    if sc.MODEL_PATH.exists():
        return joblib.load(sc.MODEL_PATH)
    print("First run: training the model, please wait a few seconds...")
    df = sc.make_demo_dataset()
    model = sc.build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(df.text, df.label)
    joblib.dump(model, sc.MODEL_PATH)
    return model


def main():
    model = get_model()
    print("=" * 50)
    print(" Sentiment Checker  (type 'quit' to exit)")
    print("=" * 50)
    while True:
        text = input("\nEnter a sentence: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not text:
            print("Please type something.")
            continue
        label = model.predict([text])[0]
        print(f"  -> {label.upper()}")


if __name__ == "__main__":
    main()
