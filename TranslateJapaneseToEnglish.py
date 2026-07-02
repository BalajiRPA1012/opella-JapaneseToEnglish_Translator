import streamlit as st
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# ---------------------------
# PDF TEXT EXTRACTION
# ---------------------------
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()

    return text


# ---------------------------
# TEXT SPLITTING ✅ FIXED
# ---------------------------
def split_text(text, chunk_size=2000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# ---------------------------
# TRANSLATION USING OLLAMA ✅ UPDATED
# ---------------------------
def translate_text(text, model_name="llama3"):
    llm = OllamaLLM(model=model_name)

    prompt = PromptTemplate.from_template("""
You are a professional translator.

Translate the following Japanese text into clear, natural English.
Preserve formatting as much as possible.

{text}
""")

    chain = prompt | llm

    chunks = split_text(text)
    translated_chunks = []

    progress_bar = st.progress(0)

    for i, chunk in enumerate(chunks):
        st.write(f"Translating chunk {i+1}/{len(chunks)}...")

        result = chain.invoke({"text": chunk})

        translated_chunks.append(result)

        progress_bar.progress((i + 1) / len(chunks))

    return "\n".join(translated_chunks)


# ---------------------------
# SAVE TRANSLATED PDF ✅ IMPROVED WRAPPING
# ---------------------------
def create_translated_pdf(text):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    c = canvas.Canvas(temp_file.name, pagesize=letter)
    width, height = letter

    x = 40
    y = height - 40

    for line in text.split("\n"):
        while len(line) > 90:
            if y < 40:
                c.showPage()
                y = height - 40

            c.drawString(x, y, line[:90])
            line = line[90:]
            y -= 15

        if y < 40:
            c.showPage()
            y = height - 40

        c.drawString(x, y, line)
        y -= 15

    c.save()

    return temp_file.name


# ---------------------------
# STREAMLIT UI ✅ FINAL
# ---------------------------
def main():
    st.set_page_config(page_title="PDF Japanese → English Translator")

    st.title("📄 Japanese → English PDF Translator")
    st.markdown("✅ Fully Local using **Ollama (Llama3 / Mistral)**")

    model_option = st.selectbox(
        "Select Model",
        ["llama3", "mistral"]
    )

    uploaded_file = st.file_uploader("Upload Japanese PDF", type=["pdf"])

    if uploaded_file is not None:
        st.success("✅ File uploaded successfully")

        if st.button("Start Translation 🚀"):

            # Step 1: Extract text
            with st.spinner("📖 Extracting text..."):
                text = extract_text_from_pdf(uploaded_file)

            if not text.strip():
                st.error("⚠️ No text found. This might be a scanned PDF.")
                return

            # Step 2: Translate
            with st.spinner("🌍 Translating..."):
                translated_text = translate_text(text, model_option)

            # Step 3: Create PDF
            with st.spinner("📄 Creating translated PDF..."):
                output_pdf = create_translated_pdf(translated_text)

            # Success
            st.success("✅ Translation Complete!")

            # Download button
            with open(output_pdf, "rb") as f:
                st.download_button(
                    label="📥 Download Translated PDF",
                    data=f,
                    file_name="translated_english.pdf",
                    mime="application/pdf"
                )

            # Cleanup
            os.remove(output_pdf)


if __name__ == "__main__":
    main()