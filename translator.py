from transformers import MarianMTModel, MarianTokenizer

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

# Example Usage
input_text = "Hello, how are you?"
translated_text = translate_text(input_text, 'en', 'es')
print(f"Original: {input_text}\nTranslated: {translated_text}")
