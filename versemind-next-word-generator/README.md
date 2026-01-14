# 🎭 VerseMind — LSTM-Powered Next-Word Language Generator

An end-to-end NLP project that uses a **Simple LSTM neural network with word embeddings**
to predict and generate the next word in a text sequence.

The model is trained on Shakespeare’s *Hamlet* and deployed as an interactive
**Streamlit web application** with a focus on usability, interpretability, and clean design.

---

## 🚀 Live Demo
👉 Deployed on Streamlit Cloud  

---

## 🧠 Project Overview

This project demonstrates the complete lifecycle of a deep learning NLP system:

- Text preprocessing & tokenization
- Sequence modeling using LSTM
- Next-word probability prediction
- Multi-word text generation
- Interactive deployment using Streamlit

Rather than focusing only on model accuracy, this project emphasizes **end-to-end execution**
and **product-style thinking**, including UI design, explainability, and deployment.

---

## 🧩 Model Architecture

- **Embedding Layer** – learns word representations
- **LSTM Layer** – captures sequential dependencies in text
- **Dense + Softmax Output** – predicts the probability of the next word

The model is trained using categorical cross-entropy on sliding text windows.

---

## ✨ Key Features

- ✍️ Clean text input with character counter
- 🔮 Single next-word prediction with confidence
- 📜 Multi-word text generation (sequence modeling)
- 🎨 Adjustable creativity using temperature sampling
- 🧠 Tokenization & padding explainability
- ⚠️ Model limitation disclosure (professional transparency)
- 📜 Example Shakespeare prompts (one-click)
- 🕒 Session-based generation history
- ⬇️ Export generated text as CSV or TXT
- 🌗 Dark / Light mode UI
- 🪟 Glassmorphism-inspired modern design

---

## ⚠️ Model Limitations

This model is trained on a **limited Shakespeare corpus** and:

- May struggle with modern language
- Has limited long-range context understanding
- Is intended for educational and demonstration purposes

These limitations are explicitly acknowledged to reflect real-world ML considerations.

---

## 🛠 Tech Stack

- Python 3.12
- TensorFlow / Keras (LSTM)
- NumPy, Pandas
- Streamlit (deployment & UI)

---

## 📂 Repository Structure
```bash
├── app.py
├── next_word_lstm.h5
├── tokenizer.pickle
├── data/
│ └── hamlet.txt
├── notebooks/
│ └── experiments.ipynb
├── requirements.txt
├── runtime.txt
└── README.md
