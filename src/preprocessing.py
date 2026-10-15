"""
Bangla text preprocessing.

Full pipeline (used by classical ML and LSTM/CNN models):

    raw text
      -> Unicode normalisation (NFC, khanda-ta, zero-width chars)
      -> noise removal (HTML, URLs, e-mails, @mentions; #hashtag -> word)
      -> digit normalisation (Bangla digits -> ASCII -> <num> placeholder)
      -> repeated-character squashing ("ভালোোোো" -> "ভালো", "😂😂😂" -> "😂")
      -> punctuation removal (dari ।, ॥ and the commonly mistyped ৷ included)
      -> Bangla-aware tokenisation (emoji kept as separate tokens)
      -> optional stopword removal (negations are NEVER removed)
      -> optional light suffix stripping (stemming)

For transformer models use `light_normalize`, which keeps punctuation and
function words because pretrained models were trained on natural text.

Why a custom tokenizer?  Python's `\\w` does not match Bangla vowel signs
(e.g. া, ো are Unicode category Mc), so scikit-learn's default token_pattern
splits "ভালো" into ['ভ', 'ল']. Every vectorizer in this project therefore
uses the tokenizer defined here.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional

from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------- #
# Character classes and regexes
# --------------------------------------------------------------------------- #
BN_RANGE = "\u0980-\u09FF"                               # Bengali Unicode block
EMOJI_RANGE = "\U0001F300-\U0001FAFF\u2600-\u27BF"        # pictographs + dingbats

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MENTION_RE = re.compile(r"@\S+")
_HASHTAG_RE = re.compile(r"#(\S+)")
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\ufe0f]")
_KHANDA_TA_RE = re.compile("\u09a4\u09cd\u200d")         # ত + ্ + ZWJ  ->  ৎ
_DARI_RE = re.compile("[\u0964\u0965\u09f7]")             # ।  ॥  ৷
_NUMBER_RE = re.compile(r"\d+(?:[.,:/]\d+)*")
_REPEAT_RE = re.compile(r"([^\d\s])\1{2,}")
_EMOJI_RE = re.compile(f"([{EMOJI_RANGE}])")
_NON_TOKEN_RE = re.compile(f"[^{BN_RANGE}a-zA-Z0-9{EMOJI_RANGE}<>\\s]")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(f"<num>|[{BN_RANGE}]+|[a-zA-Z]+|\\d+|[{EMOJI_RANGE}]")


def normalize_unicode(text: str) -> str:
    """Canonical Unicode form for Bangla.

    NFC composes split vowel signs (ে + া -> ো) and, because U+09DC/09DD/09DF
    are composition exclusions, maps both encodings of ড়/ঢ়/য় to one form.
    """
    text = unicodedata.normalize("NFC", text)
    text = _KHANDA_TA_RE.sub("\u09ce", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return text


# --------------------------------------------------------------------------- #
# Stopwords and negations
# --------------------------------------------------------------------------- #
# Negation words flip sentiment ("ভালো না" = "not good"), so they are always kept.
NEGATIONS = {normalize_unicode(w) for w in ["না", "নয়", "নেই", "নাই", "নি", "নহে", "নাহ"]}

_RAW_STOPWORDS = """
অতএব অথচ অথবা অনুযায়ী অন্য অবধি আর আগে আপনার আপনি আবার আমরা আমাকে আমাদের আমার আমি
ইত্যাদি ইহা উনি উপর উপরে এ এই এক একই একটি একটা এখন এখানে এটা এটাই এটি এদের এবং এর এরা
ও ওই ওকে ওখানে ওদের ওর ওরা কখন করতে করবে করবেন করা করি করে করেছে করেছেন করেন
কাছে কারণ কি কিংবা কিছু কী কে কেউ কেন কোন কোনো কোথায় গিয়ে গেছে গেল গেলে চলে ছিল ছিলেন
জন্য জন্যে তখন তবে তা তাই তাকে তাদের তার তারপর তারা তাহলে তিনি তুমি তো তোমার থাকবে থাকা
থাকে থেকে দিয়ে দিয়েছে দিতে দেয় দ্বারা নিয়ে নিজে নিজের পর পরে পর্যন্ত বা বিভিন্ন যখন যদি যা যাকে
যাতে যাদের যার যারা যিনি যে যেমন যেন সঙ্গে সব সবার সমস্ত সহ সাথে সুতরাং সে সেই সেখানে সেটা
সেটি হতে হয় হয়ে হয়েছে হয়েছিল হবে হলে হলো হল
"""
STOPWORDS = {normalize_unicode(w) for w in _RAW_STOPWORDS.split()} - NEGATIONS

# --------------------------------------------------------------------------- #
# Light rule-based stemmer (inflectional suffixes only)
# --------------------------------------------------------------------------- #
# Deliberately conservative and lossy; compare results with stem on/off.
# Negation suffixes such as "নি" (যাইনি) are NOT stripped.
_SUFFIXES = sorted(
    {normalize_unicode(s) for s in [
        "গুলোকে", "গুলোতে", "গুলোর", "গুলো", "গুলিকে", "গুলির", "গুলি",
        "দেরকে", "দের", "টিকে", "টাকে", "টিতে", "টাতে", "টির", "টার", "টি", "টা",
        "কে", "তে", "ের",
    ]},
    key=len, reverse=True,
)
_BN_WORD_RE = re.compile(f"^[{BN_RANGE}]+$")


def light_stem(token: str, min_stem_len: int = 3) -> str:
    if not _BN_WORD_RE.match(token):
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= min_stem_len:
            return token[: -len(suf)]
    return token


# --------------------------------------------------------------------------- #
# Main preprocessor
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessConfig:
    remove_urls: bool = True
    remove_mentions: bool = True
    keep_emoji: bool = True          # emoji carry strong sentiment signal
    replace_numbers: bool = True
    collapse_repeats: bool = True
    remove_stopwords: bool = False
    stem: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class BanglaPreprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def clean(self, text: str) -> str:
        """Normalise and clean a raw string (no tokenisation)."""
        c = self.config
        text = normalize_unicode(str(text))
        text = _HTML_RE.sub(" ", text)
        if c.remove_urls:
            text = _URL_RE.sub(" ", text)
            text = _EMAIL_RE.sub(" ", text)
        if c.remove_mentions:
            text = _MENTION_RE.sub(" ", text)
        text = _HASHTAG_RE.sub(r"\1", text)
        text = text.translate(_BN_DIGITS).lower()
        if c.replace_numbers:
            text = _NUMBER_RE.sub(" <num> ", text)
        text = _DARI_RE.sub(" ", text)
        if c.collapse_repeats:
            text = _REPEAT_RE.sub(r"\1", text)
        text = _EMOJI_RE.sub(r" \1 " if c.keep_emoji else " ", text)
        text = _NON_TOKEN_RE.sub(" ", text)
        return _WHITESPACE_RE.sub(" ", text).strip()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Bangla-aware tokenizer: Bangla words, Latin words, numbers, emoji."""
        return _TOKEN_RE.findall(text)

    def process(self, text: str) -> List[str]:
        tokens = self.tokenize(self.clean(text))
        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if self.config.stem:
            tokens = [light_stem(t) for t in tokens]
        return tokens

    def __call__(self, text: str) -> str:
        return " ".join(self.process(text))

    def batch_tokens(self, texts: Iterable[str]) -> List[List[str]]:
        return [self.process(t) for t in texts]


def light_normalize(text: str) -> str:
    """Minimal cleaning for transformer inputs (keeps punctuation & stopwords)."""
    text = normalize_unicode(str(text))
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = text.replace("\u09f7", "\u0964")      # ৷ (mistyped) -> ।
    text = _REPEAT_RE.sub(r"\1", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def whitespace_tokenizer(text: str) -> List[str]:
    """Module-level so that fitted sklearn pipelines can be pickled."""
    return text.split()


class BanglaTextCleaner(BaseEstimator, TransformerMixin):
    """scikit-learn step: raw text -> cleaned, space-joined tokens."""

    def __init__(self, remove_stopwords: bool = False, stem: bool = False,
                 keep_emoji: bool = True, replace_numbers: bool = True):
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.keep_emoji = keep_emoji
        self.replace_numbers = replace_numbers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pp = BanglaPreprocessor(PreprocessConfig(
            remove_stopwords=self.remove_stopwords, stem=self.stem,
            keep_emoji=self.keep_emoji, replace_numbers=self.replace_numbers,
        ))
        return [pp(t) for t in X]


if __name__ == "__main__":
    demo = [
        "খাবারটা অসাধারণ ছিল!!! আবার যাব 😍😍😍 https://example.com",
        "ভালোোোো না, দাম ৫০০ টাকা কিন্তু মান একদম খারাপ৷ @rahim",
        "#বাংলাদেশ দল আজকের ম্যাচে দুর্দান্ত খেলেছে",
    ]
    pp = BanglaPreprocessor(PreprocessConfig(remove_stopwords=True))
    for s in demo:
        print("RAW   :", s)
        print("TOKENS:", pp.process(s))
        print()
