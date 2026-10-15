# BanglaNLP: Bangla Text Classification and Sentiment Analysis

An end-to-end pipeline that preprocesses Bangla text and compares classical ML, deep learning and transformer models for 3-class sentiment analysis (positive / negative / neutral). The code is label-agnostic, so the same pipeline also works for topic classification, hate-speech detection and similar tasks.

## Project structure

```
banglanlp/
├── main.py                    # run experiments (ML / DL / transformer) and compare them
├── predict.py                 # predict on new text with any saved model
├── requirements.txt
├── data/
│   └── sample_bangla_sentiment.csv   # 45 hand-written sentences, for smoke-testing only
└── src/
    ├── preprocessing.py       # normalisation, tokenisation, stopwords, stemming
    ├── data_utils.py          # loading, de-duplication, stratified splits, label maps
    ├── features.py            # BoW, word TF-IDF, word + char n-gram TF-IDF
    ├── ml_models.py           # LogReg, Linear SVM, Naive Bayes, Random Forest
    ├── dl_models.py           # BiLSTM + attention, TextCNN (PyTorch)
    ├── transformer_model.py   # BanglaBERT / XLM-R fine-tuning (Hugging Face)
    └── evaluation.py          # accuracy, precision, recall, F1, confusion matrices
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install git+https://github.com/csebuetnlp/normalizer   # optional, for BanglaBERT
```

## Data format

A CSV (or TSV / XLSX / JSONL) file with a text column and a label column:

```csv
text,label
খাবারটা অসাধারণ ছিল,positive
খাবারের মান একদম ভালো না,negative
দোকানটি শুক্রবার বন্ধ থাকে,neutral
```

The sample file is only for checking that everything runs; with 5 test sentences the scores are meaningless. For real results, use a public Bangla sentiment dataset such as SentNoB (noisy Bangla social-media comments, 3 classes). If a dataset uses numeric labels, map them with `--label-map`, after checking the dataset's own documentation for which number means which class.

## Running

```bash
# Classical ML only (fast, CPU)
python main.py --data data/sample_bangla_sentiment.csv --models ml

# ML + BiLSTM + TextCNN
python main.py --data data/your_data.csv --models ml dl

# Everything, including BanglaBERT (a GPU is strongly recommended)
python main.py --data data/your_data.csv --models ml dl transformer

# Pre-split files with different column names and numeric labels
python main.py --train train.csv --val val.csv --test test.csv \
    --text-col Data --label-col Label --label-map "0:neutral,1:positive,2:negative"

# Initialise LSTM/CNN embeddings with fastText Bangla vectors (cc.bn.300.vec)
python main.py --data data/your_data.csv --models dl --embeddings cc.bn.300.vec

# Try a different transformer
python main.py --data data/your_data.csv --models transformer --transformer-model xlm-roberta-base

# Preprocessing ablations
python main.py --data data/your_data.csv --models ml --remove-stopwords --stem
```

Prediction on new text:

```bash
python predict.py --model results/models/tfidf_word_char__logreg.joblib --text "খাবারটা দারুণ ছিল"
python predict.py --model results/models/dl__bilstm.pt --file sentences.txt
python predict.py --model results/models/transformer__banglabert/best      # interactive mode
```

## Outputs (`results/`)

`summary.csv` ranks every model by test macro-F1 and `model_comparison.png` charts it. For each model, `eval/` holds a classification report (per-class precision, recall and F1), a confusion matrix image, and an `_errors.csv` file of misclassified sentences for error analysis. `class_distribution.png` shows the label balance of each split, and `models/` contains every trained model.

## Bangla-specific design decisions

**Tokenisation.** Python's `\w` does not match Bangla vowel signs (া, ো, ি…), so scikit-learn's default tokenizer splits "ভালো" into `['ভ', 'ল']`. All vectorizers here use a custom Bangla-aware tokenizer instead.

**Unicode normalisation.** Text is converted to NFC so that the different ways of typing the same character (for example য় as one code point or as য + nukta, or ো as one sign or as ে + া) become identical. The old khanda-ta encoding (ত + ্ + ZWJ) is mapped to ৎ, zero-width characters are removed, and ৷ (a currency sign often typed by mistake instead of the dari ।) is handled.

**Negation is preserved.** Words like না, নয়, নেই and নাই are never removed as stopwords, and word bigrams let models learn that "ভালো না" is negative.

**Noisy text.** Repeated characters are squashed ("ভালোোোো" → "ভালো"), Bangla digits are normalised, URLs and mentions are removed, and emoji are kept as tokens because they carry strong sentiment signal. Character n-grams make the ML models robust to spelling variation and inflection.

**Imbalanced classes.** All models use class-balanced weights, and macro-F1 (not accuracy) is used for model selection and early stopping.

**Transformers get natural text.** BanglaBERT receives only light normalisation (punctuation and function words kept), plus the official BanglaBERT normaliser when installed.
