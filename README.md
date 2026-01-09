# Sherlock Holmes Next Word Predictor 🕵️‍♂️📖

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](____)

**👆 Click the Streamlit logo above to use the live web app!**

A Deep Learning language model trained on the complete canon of Sherlock Holmes stories to generate text in the style of Arthur Conan Doyle.

> **⚠️ Note:** This is an experimental learning project exploring Long Short-Term Memory (LSTM) networks. While the model captures the vocabulary and general tone of the dataset, long-term coherence is varying. It sometimes produces coherent sentences and other times struggles with grammar—a known limitation of LSTM architectures on limited datasets.

## 🧠 Project Overview

This project builds a **Next Word Prediction** engine using TensorFlow/Keras. Unlike simple Markov chains, this model uses deep neural networks to understand context from previous words to predict the most likely next word.

Key features implemented:

- **Data Pipeline:** Aggregates 5+ novels from Project Gutenberg (~2 million characters).
- **Distributed Training:** Utilizes `tf.distribute.MirroredStrategy` to train on dual Tesla T4 GPUs.
- **Custom Sampling:** Implements **Top-K Sampling** and **Temperature Scaling** to prevent repetitive loops and control "creativity" during text generation.
- **Robust Preprocessing:** Handles variable sequence lengths and creates a dense vector space for vocabulary.

## 📂 Dataset

The training corpus was dynamically scraped from [Project Gutenberg](https://www.gutenberg.org/) using Python `requests`. It includes the full text of:

- _A Study in Scarlet_
- _The Sign of the Four_
- _The Adventures of Sherlock Holmes_
- _The Memoirs of Sherlock Holmes_
- _The Hound of the Baskervilles_

## 🛠️ Model Architecture

The model uses a **Stacked LSTM** architecture designed to capture sequential dependencies:

| Layer Type         | Parameters | Description                                                              |
| :----------------- | :--------- | :----------------------------------------------------------------------- |
| **Embedding**      | 256 dim    | Converts integer tokens into dense vectors (Masking enabled).            |
| **LSTM**           | 256 units  | First recurrent layer, returns sequences to feed the next layer.         |
| **Dropout**        | 0.3        | Regularization to prevent memorizing specific sentences.                 |
| **LSTM**           | 256 units  | Second recurrent layer, condenses sequence into a single context vector. |
| **Dropout**        | 0.3        | Further regularization.                                                  |
| **Dense**          | 256 units  | Fully connected layer with ReLU activation.                              |
| **Dense (Output)** | Vocab Size | Softmax layer predicting probability of the next word.                   |

**Total Trainable Parameters:** ~9.4 Million

## 🚀 Installation & Usage

### Prerequisites

- Python 3.10+
- TensorFlow 2.x
- NumPy
- Scikit-Learn (for data splitting)

### Running Inference

To generate text, you need the saved model (`Word_Prediction.keras`) and the tokenizer (`tokenizer.pickle`).

📥 **[Click here to download the Model ](____)** <br>
📥 **[Click here to download the Tokenizer ](____)**

```python
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load Model & Tokenizer
model = tf.keras.models.load_model('Word_Prediction.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 2. Define Generation Function (Top-K & Temperature)
def generate_text(seed, next_words, k=10, temperature=1.0):
    # ... (See notebook for full implementation) ...
    pass

# 3. Generate
print(generate_text("Sherlock Holmes sat", next_words=20, k=5, temperature=0.8))
```

## 📊 Performance & Limitations

### Training Metrics

- **Loss:** Optimized using Sparse Categorical Crossentropy.
- **Accuracy:** Reached ~18% training accuracy (typical for character/word-level models on prose where many "next words" are grammatically valid).

### Known Issues

- **Exposure Bias:** The model performs well on short phrases (1-5 words) but tends to lose coherence on longer sequences.
- **Grammar Loops:** Without Top-K sampling, the model tends to repeat common stopwords (e.g., "the the the").
- **Context Window:** The sliding window is set to 15 words; the model cannot remember plot points from previous pages.

## 🔮 Future Improvements

- **Architecture Shift:** Moving from LSTM to a **Transformer (GPT-2)** architecture would significantly improve long-term coherence.
- **Data Augmentation:** Increasing the dataset size to include more Victorian-era literature.
- **Hyperparameter Tuning:** Experimenting with larger embedding dimensions or bidirectional layers.

## 📜 License

This project uses public domain texts from Project Gutenberg.
Code is open for educational use.
