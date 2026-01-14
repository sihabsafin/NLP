import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import os
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="🎭 LSTM Next-Word Generator",
    page_icon="🎭",
    layout="centered"
)

# ======================================================
# DARK / LIGHT MODE
# ======================================================
dark_mode = st.toggle("🌗 Dark / Light Mode", value=True)

if dark_mode:
    bg = "linear-gradient(135deg, #0f2027, #203a43, #2c5364)"
    card_bg = "rgba(255,255,255,0.08)"
    text_color = "#ffffff"
    muted_text = "#d1d5db"
else:
    bg = "linear-gradient(135deg, #f8fafc, #e2e8f0)"
    card_bg = "rgba(255,255,255,0.95)"
    text_color = "#0f172a"
    muted_text = "#334155"

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown(f"""
<style>
.stApp {{
    background: {bg};
    color: {text_color};
}}

.card {{
    background: {card_bg};
    backdrop-filter: blur(16px);
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    margin-bottom: 25px;
}}

.title {{
    font-size: 42px;
    font-weight: 800;
    text-align: center;
}}

.subtitle {{
    text-align: center;
    color: {muted_text};
    margin-bottom: 30px;
}}

footer {{
    visibility: hidden;
}}
</style>
""", unsafe_allow_html=True)

# ======================================================
# FILE CHECKS
# ======================================================
MODEL_PATH = "next_word_lstm.h5"
TOKENIZER_PATH = "tokenizer.pickle"

missing = []
if not os.path.exists(MODEL_PATH):
    missing.append(MODEL_PATH)
if not os.path.exists(TOKENIZER_PATH):
    missing.append(TOKENIZER_PATH)

if missing:
    st.error(f"❌ Missing required files: {', '.join(missing)}")
    st.stop()

# ======================================================
# LOAD MODEL & TOKENIZER
# ======================================================
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()
MAX_LEN = model.input_shape[1] + 1

# ======================================================
# CORE FUNCTIONS
# ======================================================
def predict_next_word(text, temperature=1.0):
    tokens = tokenizer.texts_to_sequences([text])[0]
    tokens = tokens[-(MAX_LEN - 1):]
    padded = pad_sequences([tokens], maxlen=MAX_LEN - 1, padding="pre")

    preds = model.predict(padded, verbose=0)[0]
    preds = np.log(preds + 1e-9) / temperature
    probs = np.exp(preds) / np.sum(np.exp(preds))

    top_indices = probs.argsort()[-3:][::-1]
    top_words = []
    for idx in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                top_words.append((word, probs[idx]))

    return top_words, len(tokens)

def generate_text(seed, n_words, temperature):
    text = seed
    for _ in range(n_words):
        top_words, _ = predict_next_word(text, temperature)
        next_word = top_words[0][0]
        text += " " + next_word
    return text

# ======================================================
# SESSION STATE
# ======================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='title'>🎭 LSTM Next-Word Generator</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Shakespeare-style Language Modeling using LSTM + Embeddings</div>",
    unsafe_allow_html=True
)

# ======================================================
# MAIN CARD
# ======================================================
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # ---------------- EXAMPLES ----------------
    c1, c2, c3 = st.columns(3)
    if c1.button("📜 To be or not to"):
        st.session_state.seed = "to be or not to"
    if c2.button("🎭 O Romeo Romeo"):
        st.session_state.seed = "o romeo romeo"
    if c3.button("🌍 All the world’s a"):
        st.session_state.seed = "all the world's a"

    seed_text = st.text_area(
        "✍️ Enter seed text:",
        value=st.session_state.get("seed", ""),
        placeholder="e.g. to be or not to",
        height=140
    )

    st.caption(f"📝 Characters: {len(seed_text.strip())}")

    # ---------------- CONTROLS ----------------
    n_words = st.slider("🔢 Number of words to generate", 1, 30, 10)
    temperature = st.slider("🎨 Creativity (Temperature)", 0.5, 1.5, 1.0, 0.1)

    # ---------------- GENERATE ----------------
    if st.button("🚀 Generate Text", use_container_width=True):

        if seed_text.strip() == "":
            st.warning("Please enter some seed text.")
        else:
            top_words, token_len = predict_next_word(seed_text, temperature)
            generated = generate_text(seed_text, n_words, temperature)

            st.markdown("---")
            st.subheader("📖 Generated Text")
            st.write(generated)

            # ---------------- CONFIDENCE ----------------
            st.subheader("📊 Next-Word Prediction Confidence")
            for w, p in top_words:
                st.write(f"**{w}** — {int(p*100)}%")

            # ---------------- LIMITATION ----------------
            st.info(
                "⚠️ This LSTM model is trained on limited Shakespeare text and may not generalize "
                "well to modern language or long contexts."
            )

            # ---------------- EXPLAINABILITY ----------------
            with st.expander("🧠 How the model processes text"):
                st.write(f"🔢 Tokenized length: {token_len}")
                st.write(f"📏 Padded length: {MAX_LEN - 1}")
                st.write("Text → Tokens → Padding → Embedding → LSTM → Softmax")

            # ---------------- HISTORY ----------------
            st.session_state.history.insert(
                0,
                {
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Seed": seed_text,
                    "Words": n_words,
                    "Output": generated[:60] + "..."
                }
            )
            st.session_state.history = st.session_state.history[:5]

            # ---------------- EXPORT ----------------
            export_df = pd.DataFrame([{
                "Timestamp": datetime.now(),
                "Seed Text": seed_text,
                "Generated Text": generated,
                "Words Generated": n_words
            }])

            st.download_button(
                "⬇️ Download Generated Text (CSV)",
                export_df.to_csv(index=False),
                file_name="generated_text.csv",
                mime="text/csv"
            )

            st.download_button(
                "⬇️ Download Generated Text (TXT)",
                generated,
                file_name="generated_text.txt",
                mime="text/plain"
            )

    # ---------------- HISTORY TABLE ----------------
    if st.session_state.history:
        st.subheader("🕒 Recent Generations")
        st.table(pd.DataFrame(st.session_state.history))

    st.markdown("</div>", unsafe_allow_html=True)
