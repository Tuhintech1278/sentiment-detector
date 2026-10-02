#!/usr/bin/env python3
"""
Intelligent Multi-Class Sentiment Classifier
============================================
Classes: negative | neutral | positive

Pipeline
    raw text -> clean -> tokenize -> stop-word removal (negations kept)
             -> lemmatization -> TF-IDF (word 1-2 grams) -> classifier
Evaluation
    Stratified train/test split, cross-validated model comparison,
    per-class precision/recall/F1, macro / weighted / micro F1, confusion matrix.

Usage
    python sentiment_classifier.py                       # built-in demo dataset
    python sentiment_classifier.py --data reviews.csv    # CSV with columns: text,label
    python sentiment_classifier.py --predict "Not good at all"   # uses saved model

Dependencies
    pip install scikit-learn pandas numpy joblib nltk
    (NLTK is optional: without it a built-in lemmatizer and sklearn's stop-word
     list are used automatically.)
"""
import argparse
import random
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
MODEL_PATH = Path("sentiment_model.joblib")
LABELS = ["negative", "neutral", "positive"]


# ----------------------------------------------------------------------------
# 1. TEXT PREPROCESSING  (stop-word removal + lemmatization)
# ----------------------------------------------------------------------------
# Negations/intensifiers carry sentiment ("not good" != "good"), so they are
# kept even though standard stop-word lists contain them.
KEEP_WORDS = {"not", "no", "nor", "never", "n't", "cannot", "without",
              "very", "too", "against", "but", "only", "hardly", "barely"}


def _load_nltk():
    """Return (stopwords, lemmatizer) from NLTK, or (None, None) if unavailable."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        for pkg, path in [("stopwords", "corpora/stopwords"),
                          ("wordnet", "corpora/wordnet"),
                          ("omw-1.4", "corpora/omw-1.4")]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
        lem = WordNetLemmatizer()
        lem.lemmatize("tests")  # raises if wordnet is missing
        return set(stopwords.words("english")), lem
    except Exception:
        return None, None


class _FallbackLemmatizer:
    """Small rule-based lemmatizer used when NLTK/WordNet is unavailable."""
    IRREGULAR = {"was": "be", "were": "be", "is": "be", "are": "be", "been": "be",
                 "had": "have", "has": "have", "did": "do", "does": "do",
                 "went": "go", "gone": "go", "bought": "buy", "better": "good",
                 "best": "good", "worse": "bad", "worst": "bad", "ran": "run",
                 "broke": "break", "broken": "break", "made": "make",
                 "loved": "love", "hated": "hate", "liked": "like",
                 "children": "child", "people": "person", "men": "man"}

    def lemmatize(self, w, pos=None):
        if w in self.IRREGULAR:
            return self.IRREGULAR[w]
        if len(w) <= 3:
            return w
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith(("sses", "shes", "ches", "xes", "zes")):
            return w[:-2]
        if w.endswith("ing") and len(w) > 5:
            stem = w[:-3]
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "ls":
                stem = stem[:-1]               # running -> run
            return stem
        if w.endswith("ed") and len(w) > 4:
            stem = w[:-2]
            if stem.endswith("i"):
                return stem[:-1] + "y"         # tried -> try
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "ls":
                stem = stem[:-1]               # stopped -> stop
            return stem if not w.endswith(("ated", "ized", "ised")) else w[:-1]
        if w.endswith("s") and not w.endswith(("ss", "us", "is")):
            return w[:-1]
        return w


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer: list[str] -> list[cleaned str]."""

    _URL = re.compile(r"https?://\S+|www\.\S+")
    _HTML = re.compile(r"<[^>]+>")
    _CONTRACTIONS = [(r"won't", "will not"), (r"can't", "can not"),
                     (r"n't", " not"), (r"'re", " are"), (r"'ve", " have"),
                     (r"'ll", " will"), (r"'m", " am"), (r"'d", " would"),
                     (r"'s", "")]
    _TOKEN = re.compile(r"[a-z]+")

    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

    def fit(self, X, y=None):
        nltk_stop, nltk_lem = _load_nltk()
        base = nltk_stop if nltk_stop else set(ENGLISH_STOP_WORDS)
        self.stopwords_ = (base - KEEP_WORDS) | {"u", "ur", "im"}
        self.lemmatizer_ = nltk_lem if nltk_lem else _FallbackLemmatizer()
        self.backend_ = "nltk-wordnet" if nltk_lem else "built-in fallback"
        return self

    def _clean(self, text):
        text = str(text).lower()
        text = self._URL.sub(" ", text)
        text = self._HTML.sub(" ", text)
        for pat, rep in self._CONTRACTIONS:
            text = re.sub(pat, rep, text)
        tokens = self._TOKEN.findall(text)
        out = []
        for tok in tokens:
            if len(tok) < 2 or (self.remove_stopwords and tok in self.stopwords_):
                continue
            if self.lemmatize:
                # try verb form first (loved->love), then noun form (movies->movie)
                try:
                    lem = self.lemmatizer_.lemmatize(tok, pos="v")
                    if lem == tok:
                        lem = self.lemmatizer_.lemmatize(tok, pos="n")
                except TypeError:
                    lem = self.lemmatizer_.lemmatize(tok)
                tok = lem
            out.append(tok)
        return " ".join(out)

    def transform(self, X):
        return [self._clean(t) for t in X]


# ----------------------------------------------------------------------------
# 2. DATA  (CSV loader + synthetic demo corpus so the script runs out of the box)
# ----------------------------------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path)
    if not {"text", "label"} <= set(df.columns):
        sys.exit("CSV must contain 'text' and 'label' columns.")
    return df.dropna(subset=["text", "label"]).reset_index(drop=True)


def make_demo_dataset(n_per_class=2500, seed=SEED):
    """Synthetic multi-class corpus with a wide vocabulary and negation handling.
    Demo only - for best results train on real data via --data."""
    rng = random.Random(seed)
    subjects = ["The movie", "This phone", "The service", "The hotel", "My order", "The app",
                "The food", "The staff", "This laptop", "The delivery", "The product", "The show",
                "The teacher", "The restaurant", "The game", "This book", "The camera",
                "The experience", "The store", "The class", "The meal", "Everything", "It", "This",
                "The music", "The weather", "My day", "The trip", "The party", "The doctor"]
    nouns = ["movie", "phone", "service", "meal", "day", "product", "trip", "show", "purchase",
             "place", "book", "game", "film", "experience", "job"]
    pos_adj = ("good great excellent amazing awesome fantastic wonderful brilliant superb outstanding "
               "perfect lovely delightful fabulous terrific incredible beautiful nice best enjoyable "
               "helpful friendly fast delicious tasty fresh impressive pleasant charming cool fun funny "
               "smart clever elegant stunning gorgeous marvelous magnificent splendid satisfying reliable "
               "comfortable cozy clean smooth easy happy glad thrilled excited delighted joyful cheerful "
               "proud grateful pleased satisfied relaxed peaceful favorite worthwhile valuable generous "
               "kind polite warm affordable exceptional flawless sweet fine-quality refreshing "
               "inspiring entertaining hilarious magical epic top-notch first-class").split()
    neg_adj = ("bad terrible awful horrible poor dreadful useless annoying disappointing worst boring "
               "rude slow broken disgusting pathetic frustrating unacceptable overpriced worthless "
               "unreliable sad ugly dirty cold stale bland noisy uncomfortable difficult confusing "
               "painful stressful miserable angry upset furious unhappy depressed anxious worried "
               "scared annoyed irritated disgusted nasty mean cruel lazy weak faulty defective shoddy "
               "unfriendly unhelpful dull awkward horrid lousy mediocre tiresome exhausting "
               "depressing gross creepy hopeless regrettable awful-tasting trash garbage "
               "unbearable infuriating insulting").split()
    neu_adj = ["okay", "average", "fine", "acceptable", "standard", "ordinary", "decent enough",
               "unremarkable", "as expected", "adequate", "normal", "typical", "so-so", "moderate",
               "plain", "regular", "alright", "passable", "neither good nor bad", "just okay"]
    pos_verbs = "love like enjoy adore appreciate admire recommend treasure".split()
    neg_verbs = "hate dislike despise regret loathe detest resent".split()
    pos_phrases = ["Great job!", "Well done!", "Thank you so much!", "Highly recommended.", "Love it!",
                   "Can't wait to come back.", "Made my day.", "Worth every penny.", "Five stars.",
                   "Exceeded my expectations.", "I am so happy today.", "What a wonderful day.",
                   "Absolutely loved it.", "Best decision ever.", "Totally satisfied.", "Good job.",
                   "Everything was perfect.", "I feel great.", "Couldn't be happier."]
    neg_phrases = ["Never again.", "Total waste of money.", "I want a refund.", "Waste of time.",
                   "Complete disaster.", "Do not buy this.", "Very disappointed.", "I am never coming back.",
                   "Worst experience ever.", "I feel terrible.", "This ruined my day.", "Zero stars.",
                   "Not worth it.", "Not good at all.", "I regret buying this.", "What a waste.",
                   "Absolutely horrible.", "I am so angry right now.", "Totally unacceptable.", "Not happy."]
    days = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
    cities = ["London", "Delhi", "Paris", "Tokyo", "the city centre", "the airport", "the station"]

    def neutral_fact():
        k = rng.randint(2, 500)
        return rng.choice([
            f"The {rng.choice(nouns)} arrived on {rng.choice(days)}.",
            f"The meeting is at {rng.randint(1, 12)} pm.",
            f"I bought it {rng.choice(['last week', 'yesterday', 'in March', 'two days ago'])}.",
            f"The store opens at {rng.randint(7, 11)} am.",
            f"The {rng.choice(nouns)} costs {k} dollars.",
            f"We went to {rng.choice(cities)} on {rng.choice(days)}.",
            f"The report has {rng.randint(3, 40)} pages.",
            f"He works in {rng.choice(cities)}.",
            f"The train leaves from platform {rng.randint(1, 9)}.",
            f"I will check it on {rng.choice(days)}.",
            f"It comes in {rng.randint(2, 5)} sizes.",
            f"The class starts next {rng.choice(days)}.",
            f"The {rng.choice(nouns)} weighs {rng.randint(1, 20)} kilograms.",
            f"She lives near {rng.choice(cities)}.",
            f"The package contains {rng.randint(1, 6)} items.",
            f"I need to submit the form by {rng.choice(days)}.",
            f"The event is held in {rng.choice(cities)}.",
            f"They are talking about the schedule.",
            f"The {rng.choice(nouns)} was delivered to my office.",
            f"What time does the {rng.choice(nouns)} start?",
        ])

    def pos_sentence():
        s_, a, n = rng.choice(subjects), rng.choice(pos_adj), rng.choice(nouns)
        v = rng.choice(pos_verbs)
        return rng.choice([
            f"{s_} was {a}!", f"{s_} is {a}.", f"{s_} was really {a}.", f"I think {s_.lower()} is {a}.",
            f"What a {a} {n}!", f"Best {n} ever!", f"So {a}!", f"Absolutely {a}.", f"I am so {a}.",
            f"I {v} it.", f"I really {v} this {n}.", f"We {v} the {n} so much.", f"I {v} this!",
            f"{s_} was not {rng.choice(neg_adj)} at all." if rng.random() < .3 else f"Truly {a}, thank you.",
            f"{s_} was {a} and {rng.choice(pos_adj)}.", f"A {a} {n}, highly recommend.",
            f"I feel {a} today.", f"Very {a} {n}.", rng.choice(pos_phrases), rng.choice(pos_phrases)])

    def neg_sentence():
        s_, a, n = rng.choice(subjects), rng.choice(neg_adj), rng.choice(nouns)
        v = rng.choice(neg_verbs)
        return rng.choice([
            f"{s_} was {a}.", f"{s_} is {a}!", f"{s_} was really {a}.", f"I think {s_.lower()} is {a}.",
            f"What a {a} {n}!", f"Worst {n} ever!", f"So {a}.", f"Absolutely {a}.", f"I am so {a}.",
            f"I {v} it.", f"I really {v} this {n}.", f"We {v} the {n}.", f"I {v} this!",
            f"{s_} was not {rng.choice(pos_adj)}.", f"{s_} is not {rng.choice(pos_adj)} at all.",
            f"I do not {rng.choice(pos_verbs)} it.", f"I don't {rng.choice(pos_verbs)} this {n}.",
            f"{s_} was {a} and {rng.choice(neg_adj)}.", f"A {a} {n}, do not recommend.",
            f"I feel {a} today.", f"Very {a} {n}.", rng.choice(neg_phrases), rng.choice(neg_phrases)])

    def neu_sentence():
        if rng.random() < 0.6:
            return neutral_fact()
        s_, a = rng.choice(subjects), rng.choice(neu_adj)
        return rng.choice([
            f"{s_} was {a}.", f"{s_} is {a}, nothing special.", f"It was {a}, not great and not bad.",
            f"{s_} did the job, {a} overall.", f"I have mixed feelings; {s_.lower()} was {a}.",
            f"{s_} is just {a}.", f"{s_} was not {rng.choice(neg_adj)}.",
            f"{s_} was neither {rng.choice(pos_adj)} nor {rng.choice(neg_adj)}."])

    rows = []
    for _ in range(n_per_class):
        rows += [(pos_sentence(), "positive"), (neg_sentence(), "negative"), (neu_sentence(), "neutral")]
    df = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates().reset_index(drop=True)
    # ~2% label noise so metrics are realistic rather than trivially 1.0
    noisy = df.sample(frac=0.02, random_state=seed).index
    df.loc[noisy, "label"] = [rng.choice([l for l in LABELS if l != df.label[i]]) for i in noisy]
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. MODEL SELECTION + EVALUATION
# ----------------------------------------------------------------------------
def build_pipeline(clf):
    return Pipeline([
        ("prep", TextPreprocessor()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", clf),
    ])


def compare_models(X_train, y_train):
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced", random_state=SEED),
        "ComplementNB": ComplementNB(),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print("\n=== 5-fold CV (macro F1) ===")
    scores = {}
    for name, clf in candidates.items():
        s = cross_val_score(build_pipeline(clf), X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        scores[name] = s.mean()
        print(f"{name:<20} {s.mean():.4f} (+/- {s.std():.4f})")
    return max(scores, key=scores.get)


def tune(best_name, X_train, y_train):
    grids = {
        "LogisticRegression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                               {"clf__C": [0.5, 1, 5, 10]}),
        "LinearSVC": (LinearSVC(class_weight="balanced", random_state=SEED),
                      {"clf__C": [0.1, 0.5, 1, 2]}),
        "ComplementNB": (ComplementNB(), {"clf__alpha": [0.1, 0.5, 1.0]}),
    }
    clf, grid = grids[best_name]
    grid.update({"tfidf__ngram_range": [(1, 1), (1, 2)]})
    gs = GridSearchCV(build_pipeline(clf), grid, scoring="f1_macro",
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"\nBest model: {best_name} | params: {gs.best_params_} | CV macro-F1: {gs.best_score_:.4f}")
    return gs.best_estimator_


def evaluate(model, X_test, y_test, plot=True):
    pred = model.predict(X_test)
    print("\n=== Test-set evaluation ===")
    print(classification_report(y_test, pred, labels=LABELS, digits=4, zero_division=0))
    print(f"Macro F1   : {f1_score(y_test, pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_test, pred, average='weighted'):.4f}")
    print(f"Micro F1   : {f1_score(y_test, pred, average='micro'):.4f}")
    cm = confusion_matrix(y_test, pred, labels=LABELS)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues", values_format="d")
            plt.title("Sentiment confusion matrix")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png", dpi=150)
            print("Saved confusion_matrix.png")
        except ImportError:
            pass


def show_top_features(model, k=8):
    clf, vec = model.named_steps["clf"], model.named_steps["tfidf"]
    if not hasattr(clf, "coef_"):
        return
    names = np.array(vec.get_feature_names_out())
    print("\n=== Most influential terms per class ===")
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        print(f"{cls:<9}: {', '.join(names[np.argsort(coefs)[-k:][::-1]])}")


# ----------------------------------------------------------------------------
# 4. INFERENCE
# ----------------------------------------------------------------------------
def predict_texts(texts):
    if not MODEL_PATH.exists():
        sys.exit(f"{MODEL_PATH} not found - train first by running without --predict.")
    model = joblib.load(MODEL_PATH)
    clf = model.named_steps["clf"]
    for t in texts:
        label = model.predict([t])[0]
        if hasattr(model, "predict_proba"):
            p = model.predict_proba([t])[0]
            conf = f"  (confidence {p.max():.2f})"
        elif hasattr(clf, "decision_function"):
            d = model.decision_function([t])[0]
            e = np.exp(d - d.max())
            conf = f"  (confidence ~{(e / e.sum()).max():.2f})"
        else:
            conf = ""
        print(f"[{label.upper():<8}] {t}{conf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV with 'text' and 'label' columns")
    ap.add_argument("--predict", nargs="+", help="classify one or more sentences with the saved model")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    if args.predict:
        return predict_texts(args.predict)

    df = load_csv(args.data) if args.data else make_demo_dataset()
    if not args.data:
        print("No --data given: using synthetic demo dataset.")
    print(f"Samples: {len(df)}\n{df.label.value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df.text, df.label, test_size=args.test_size, stratify=df.label, random_state=SEED)

    prep = TextPreprocessor().fit([])
    print(f"\nPreprocessing backend: {prep.backend_}")
    for raw in X_train.iloc[:3]:
        print(f"  RAW  : {raw}\n  CLEAN: {prep._clean(raw)}")

    best = compare_models(X_train, y_train)
    model = tune(best, X_train, y_train)
    evaluate(model, X_test, y_test)
    show_top_features(model)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    print("\n=== Demo predictions ===")
    predict_texts(["I absolutely love this, it's brilliant!",
                   "This was a complete waste of money.",
                   "It arrived on time and works as described.",
                   "Not good at all, I'm very disappointed."])


if __name__ == "__main__":
    main()
