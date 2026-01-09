import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import os
import gdown
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PREDICTION AI",
    page_icon="🕵️‍♂️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS to improve mobile spacing and font sizes
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px !important;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD RESOURCES (Cached for speed)
# ==========================================
@st.cache_resource

@st.cache_resource
def load_resources():
    #Define the Google Drive ID
    file_id = '1UzgXAMyvwOIpe62Wq5qu-CNQxqXZBokB' 
    output_model_file = 'Word_Prediction.keras'
    
    #Check if model file exists locally; if not, download it
    if not os.path.exists(output_model_file):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_model_file, quiet=False)

    #Load the model
    model = tf.keras.models.load_model(output_model_file)
    
    #Load the Tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return model, tokenizer

try:
    model, tokenizer = load_resources()
    # Get sequence length from model input shape (usually 15)
    SEQ_LEN = model.input_shape[1] 
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ==========================================
# 3. GENERATION ENGINE
# ==========================================
def generate_text(model, tokenizer, seed_text, next_words, max_sequence_len, k=10, temperature=1.0):
    output_text = seed_text
    
    # Progress bar for long generation
    my_bar = st.progress(0)
    
    for i in range(next_words):
        # Update progress
        my_bar.progress((i + 1) / next_words)
        
        # 1. Preprocessing
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        # Important: padding='post' prevents CuDNN/Masking errors on GPU logic
        # truncating='pre' ensures we keep the most recent context words
        token_list = pad_sequences([token_list], 
                                   maxlen=max_sequence_len, 
                                   padding='post', 
                                   truncating='pre')
        
        # 2. Predict
        predictions = model.predict(token_list, verbose=0)[0]
        
        # 3. Sampling Logic
        if temperature <= 0.01:
            # Greedy search (Robot mode)
            predicted_id = np.argmax(predictions)
        else:
            # Temperature Math (Creativity factor)
            predictions = np.log(predictions + 1e-7) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)
            
            # Top-K Sampling (Vocabulary focus)
            if k < 1: k = 1
            top_k_indices = predictions.argsort()[-k:][::-1]
            top_k_probs = predictions[top_k_indices]
            # Renormalize to ensure sum is 1.0 for random.choice
            top_k_probs = top_k_probs / np.sum(top_k_probs)
            
            # Roll Dice based on new probabilities
            predicted_id = np.random.choice(top_k_indices, p=top_k_probs)
            
        # 4. Append Word
        output_word = tokenizer.index_word.get(predicted_id, "")
        if output_word:
            output_text += " " + output_word
            
    my_bar.empty() # Clear bar when done
    return output_text

# ==========================================
# 4. USER INTERFACE
# ==========================================

# --- Header Section ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/cd/Sherlock_Holmes_Portrait_Paget.jpg", width=80)
with col2:
    st.title("Predict The Word")

st.markdown("---")

# --- Input Section ---
st.subheader("✍️ Start your story")
seed_text = st.text_area(
    label="Enter text here", 
    height=100, 
    label_visibility="collapsed",
    placeholder="Type the beginning of a sentence..."
)

# --- Basic Settings (Row 1) ---
# We use columns to make it look good on desktop, but they stack on mobile
c1, c2 = st.columns([3, 1])
with c1:
    st.write("") # Spacer
with c2:
    # Just a visual spacer if needed
    pass

# --- Primary Control: Length ---
st.write("LENGTH OF GENERATION")
num_words = st.slider(
    "How many words to generate?", 
    min_value=5, 
    max_value=100, 
    value=20, 
    label_visibility="collapsed"
)

# --- Advanced Settings (Hidden in Expander for Mobile Cleanliness) ---
with st.expander("⚙️ Advanced Settings (Creativity & Vocabulary)", expanded=False):
    st.info("💡 **Tip:** Higher temperature makes the AI more creative but riskier.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        temperature = st.slider(
            "Creativity (Temperature)", 
            min_value=0.1, 
            max_value=1.5, 
            value=0.6, 
            step=0.1
        )
    
    with col_b:
        top_k = st.slider(
            "Vocabulary Size (Top-K)", 
            min_value=1, 
            max_value=50, 
            value=10
        )

st.markdown("---")

# --- Generate Button ---
if st.button("✨ GENERATE WORD/S", type="primary"):
    if not seed_text.strip():
        st.warning("⚠️ Please enter some text to start!")
    else:
        # Create a placeholder for the output so it appears smoothly
        result_container = st.container()
        
        with st.spinner("🕵️‍♂️ Consulting the Mind Palace..."):
            try:
                generated_story = generate_text(
                    model, 
                    tokenizer, 
                    seed_text, 
                    next_words=num_words, 
                    max_sequence_len=SEQ_LEN, 
                    k=top_k, 
                    temperature=temperature
                )
                
                # --- Display Result ---
                with result_container:
                    st.markdown("### 📜 Result")
                    # Using a text area for copy-paste ability, or a styled box
                    st.success(generated_story)
                    
                    # Copy button workaround (Streamlit doesn't have native copy btn yet)
                    st.code(generated_story, language=None)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

# --- Footer ---
st.markdown(
    """
    <div style='text-align: center; margin-top: 50px; color: #666;'>
        <small>Powered by LSTM & TensorFlow • Mobile Optimized</small>
    </div>
    """, 
    unsafe_allow_html=True
)