import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

@st.cache_resource
def load_model(src_lang='en', tgt_lang='es'):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text, src_lang='en', tgt_lang='es'):
    model, tokenizer = load_model(src_lang, tgt_lang)
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translation = model.generate(**tokens)
    return tokenizer.decode(translation[0], skip_special_tokens=True)

# Streamlit UI
st.title("🌐 Multilingual Muse: AI Language Translator")
text = st.text_area("Enter text to translate:")
src_lang = st.selectbox("Select source language", ['en', 'es', 'fr', 'ta'])
tgt_lang = st.selectbox("Select target language", ['en', 'es', 'fr', 'ta'])

if st.button("Translate"):
    if text:
        result = translate_text(text, src_lang, tgt_lang)
        st.success(f"**Translated Text:** {result}")
    else:
        st.warning("Please enter text to translate!")
