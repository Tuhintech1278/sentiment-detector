# Sentiment Detector

A small NLP project that classifies a sentence as **positive**, **negative** or **neutral**.

- Preprocessing: stop-word removal (negations kept) and lemmatization
- Features: TF-IDF (word 1-2 grams)
- Models compared: Logistic Regression, LinearSVC, Complement Naive Bayes (best chosen by macro F1)
- Evaluation: precision / recall / F1 per class, macro / weighted / micro F1, confusion matrix
- Web app: Flask page where you type a sentence and get the result

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Train the model

```bash
python sentiment_classifier.py                    # built-in demo data
python sentiment_classifier.py --data mydata.csv  # your own CSV with columns: text,label
```

## Run the website

```bash
python app.py
```

Open http://127.0.0.1:5000. If no trained model exists, the app trains one on first start.

## Other ways to use it

```bash
python ask.py                                          # terminal version
python sentiment_classifier.py --predict "Great job"   # one-off prediction
```
