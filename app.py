import streamlit as st
import pickle
import os

# Load models and vectorizers
with open(os.path.join("models", "model_title.pkl"), "rb") as f:
    model_title = pickle.load(f)

with open(os.path.join("models", "vectorizer_title.pkl"), "rb") as f:
    vectorizer_title = pickle.load(f)

with open(os.path.join("models", "model_content.pkl"), "rb") as f:
    model_content = pickle.load(f)

with open(os.path.join("models", "vectorizer_content.pkl"), "rb") as f:
    vectorizer_content = pickle.load(f)

# Page setup
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.write("📅 **Dataset used:** The model was trained using news data from the years **2016–2017** to detect fake and real news accurately.")
st.write("Enter a news **headline** or **full article** from the years 2016 and 2017 to detect if it's Real or Fake.")

# --- Session state defaults ---
if "input_type" not in st.session_state:
    st.session_state.input_type = "Title Only"
if "title_input" not in st.session_state:
    st.session_state.title_input = ""
if "article_input" not in st.session_state:
    st.session_state.article_input = ""

# --- Input type selection ---
input_type = st.radio("Choose Input Type:", ["Title Only", "Article Only"], index=0)
st.session_state.input_type = input_type  # Store selection

# --- Input based on type ---
if input_type == "Title Only":
    title_input = st.text_input("📝 Enter News Title:", value=st.session_state.title_input)
    st.session_state.title_input = title_input
else:
    article_input = st.text_area("📰 Enter Full Article Text:", height=200, value=st.session_state.article_input)
    st.session_state.article_input = article_input

# --- Check Authenticity ---
if st.button("Check Authenticity"):
    if input_type == "Title Only":
        if st.session_state.title_input.strip() == "":
            st.warning("Please enter the news title.")
        else:
            vec = vectorizer_title.transform([st.session_state.title_input])
            result = model_title.predict(vec)[0]
            if result == 1:
                st.success("✅ This news is likely **REAL**.")
            else:
                st.error("❌ Warning! This news is likely **FAKE**.")
    else:
        if st.session_state.article_input.strip() == "":
            st.warning("Please enter article.")
        else:
            vec = vectorizer_content.transform([st.session_state.article_input])
            result = model_content.predict(vec)[0]
            if result == 1:
                st.success("✅ This news is likely **REAL**.")
            else:
                st.error("❌ Warning! This news is likely **FAKE**.")

# --- Reset button ---
if st.button("Reset"):
    # Reset session state
    st.session_state.title_input = ""
    st.session_state.article_input = ""
    st.rerun()
