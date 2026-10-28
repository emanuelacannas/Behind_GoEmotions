"""
demo.py: Gradio demo for Behind GoEmotions.

Loads all four trained models and runs inference on a user-supplied text,
reproducing the same preprocessing and vectorization pipeline used during
training (notebook 01 / 02).

Run from the project root:
    python demo/demo.py

Requires:
    pip install gradio tensorflow spacy
    python -m spacy download en_core_web_sm
"""

import os
import sys
import re
import json

import numpy as np
import tensorflow as tf
import spacy
import gradio as gr

# paths 
# demo.py lives in  <project_root>/demo/demo.py
PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VECTORIZED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "vectorized")
EXPERIMENTS     = os.path.join(PROJECT_ROOT, "experiments")
TRANSFORMER_DIR = os.path.join(EXPERIMENTS, "transformer")

# make transformer_layers importable
if TRANSFORMER_DIR not in sys.path:
    sys.path.append(TRANSFORMER_DIR)
from transformer_layers import PositionalEmbedding, EncoderBlock  # noqa: E402

# experiment date used as checkpoint prefix (matches the run kept in the repo)
DATE = "2026-06-06"

# constants (must match training)
VOCAB_SIZE = 2500
SEQ_LEN    = 128

LABEL_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise",
]
NUM_CLASSES = len(LABEL_NAMES)


#  preprocessing (mirrors preprocess_text in notebook 01) 
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    """Lowercase, remove punctuation, lemmatize, and strip stopwords using spaCy."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    doc  = nlp(text)
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_space
    ]
    return " ".join(lemmas)


# vocabularies
def _load_vocab(path: str) -> list:
    """Load a JSON vocabulary file and return it as a list."""
    with open(path, encoding="utf-8") as f:
        v = json.load(f)
    return v if isinstance(v, list) else list(v)

tfidf_vocab = _load_vocab(os.path.join(VECTORIZED_PATH, "vocab_tfidf.json"))
int_vocab   = _load_vocab(os.path.join(VECTORIZED_PATH, "vocab_int.json"))

# build lookup tables: token -> index
tfidf_token_to_id = {token: i for i, token in enumerate(tfidf_vocab)}
int_token_to_id   = {token: i for i, token in enumerate(int_vocab)}


# vectorizers (pure-Python, no Keras TextVectorization needed at inference)
def vectorize_tfidf(processed: str) -> np.ndarray:
    """
    Reproduce the TF-IDF vectorizer output for a single preprocessed string.

    Uses unigrams and bigrams with raw term-frequency weighting, matching the
    TextVectorization layer used in training (output_mode='tf_idf').
    Unknown tokens are mapped to index 0 ([UNK]).
    Returns a float32 array of shape (1, VOCAB_SIZE).
    """
    tokens   = processed.split()
    bigrams  = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    ngrams   = tokens + bigrams

    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for ng in ngrams:
        idx = tfidf_token_to_id.get(ng, 0)   # 0 = [UNK]
        vec[idx] += 1.0

    return vec.reshape(1, -1)


def vectorize_int(processed: str) -> np.ndarray:
    """
    Reproduce the integer-sequence vectorizer output for a single preprocessed string.

    Truncates or zero-pads to SEQ_LEN, matching the TextVectorization layer
    used in training (output_mode='int', output_sequence_length=SEQ_LEN).
    Returns an int32 array of shape (1, SEQ_LEN).
    """
    tokens = processed.split()
    ids = [int_token_to_id.get(t, 0) for t in tokens]   # 0 = [UNK]
    ids = ids[:SEQ_LEN]                                   # truncate
    ids += [0] * (SEQ_LEN - len(ids))                     # zero-pad
    return np.array(ids, dtype=np.int32).reshape(1, -1)


# model loading
def _ckpt(experiment: str, name: str) -> str:
    """Return the path to a .keras checkpoint file given the experiment folder and model name."""
    return os.path.join(EXPERIMENTS, experiment, "checkpoints", f"{DATE}_{name}.keras")

print("Loading models…")

model_linear = tf.keras.models.load_model(_ckpt("tfidf_linear", "tfidf_linear"))
model_mlp    = tf.keras.models.load_model(_ckpt("tfidf_mlp",    "tfidf_mlp"))
model_bilstm = tf.keras.models.load_model(_ckpt("bilstm",       "bilstm"))
model_trans  = tf.keras.models.load_model(
    _ckpt("transformer", "transformer"),
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "EncoderBlock":        EncoderBlock,
    },
)

print("Models loaded.")


# inference 
def predict(text: str) -> dict:
    """
    Run all four models on a raw English text and return a dict of
    {model_name: {label: probability}} suitable for Gradio Label components.

    Steps: preprocess -> vectorize -> predict -> format as label-prob dicts.
    """
    processed = preprocess_text(text)
    x_tfidf   = vectorize_tfidf(processed)
    x_int     = vectorize_int(processed)

    probs_linear = model_linear.predict(x_tfidf, verbose=0)[0]
    probs_mlp    = model_mlp.predict(x_tfidf,    verbose=0)[0]
    probs_bilstm = model_bilstm.predict(x_int,   verbose=0)[0]
    probs_trans  = model_trans.predict(x_int,    verbose=0)[0]

    def to_label_dict(probs):
        return {LABEL_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    return (
        to_label_dict(probs_mlp),
        to_label_dict(probs_linear),
        to_label_dict(probs_bilstm),
        to_label_dict(probs_trans),
    )


# Gradio interface
_examples = [
    ["Thank you so much, I really appreciate your help!"],
    ["This is absolutely ridiculous, I can't believe they did that."],
    ["I'm not sure how to feel about this, it's all so confusing."],
    ["I wanna die. I don't wanna live anymore."],
    ["Congratulations! I'm so proud of what you've accomplished."],
]

with gr.Blocks(title="Behind GoEmotions") as demo:
    gr.Markdown(
        """
        # Behind GoEmotions
        Emotion classifier trained on the [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
        dataset. Enter an English text and compare the predictions of four architectures trained from scratch.
        """
    )

    with gr.Row():
        text_input = gr.Textbox(
            lines=3,
            placeholder="Type an English sentence here…",
            label="Input text",
        )

    submit_btn = gr.Button("Classify", variant="primary")

    with gr.Row():
        out_mlp    = gr.Label(num_top_classes=5, label="TF-IDF + MLP")
        out_linear = gr.Label(num_top_classes=5, label="TF-IDF Linear")

    with gr.Row():
        out_bilstm = gr.Label(num_top_classes=5, label="BiLSTM")
        out_trans  = gr.Label(num_top_classes=5, label="Transformer")

    submit_btn.click(
        fn=predict,
        inputs=text_input,
        outputs=[out_mlp, out_linear, out_bilstm, out_trans],
    )

    gr.Examples(
        examples=_examples,
        inputs=text_input,
        label="Example inputs",
    )

    gr.Markdown(
        """
        ---
        **Note on preprocessing.** The input text is lowercased, punctuation is removed,
        and it is lemmatized and stopword-filtered with spaCy (`en_core_web_sm`) before
        being fed to the models, the same pipeline used during training.

        TF-IDF Linear and MLP receive a bag-of-ngrams vector (unigrams + bigrams, vocab size 2500).
        BiLSTM and Transformer receive an integer token sequence padded to length 128.
        """
    )


if __name__ == "__main__":
    demo.launch(share=False)
