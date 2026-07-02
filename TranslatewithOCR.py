import streamlit as st
import fitz
import pytesseract
from PIL import Image
import io
import os
import re
import pandas as pd
import tempfile

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# ---------------------------
# ✅ TESSERACT CONFIG
# ---------------------------
TESS_PATH = r"C:\Users\U1074239\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
TESSDATA = r"C:\Users\U1074239\AppData\Local\Programs\Tesseract-OCR\tessdata"

if os.path.exists(TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH
    os.environ["TESSDATA_PREFIX"] = TESSDATA
else:
    st.error("❌ Tesseract not found. Check path.")


# ---------------------------
# ✅ OCR EXTRACTION
# ---------------------------
def extract_text_from_pdf(pdf_file):
    file_bytes = pdf_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    text = ""

    for page_num, page in enumerate(doc):
        page_text = page.get_text()

        if page_text.strip():
            text += page_text
        else:
            st.warning(f"🔍 OCR page {page_num+1}")

            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            img = img.convert("L")

            ocr_text = pytesseract.image_to_string(
                img,
                lang="jpn",
                config="--oem 3 --psm 6 -c preserve_interword_spaces=0"
            )

            text += ocr_text + "\n"

    return text


# ---------------------------
# ✅ CLEAN JAPANESE TEXT
# ---------------------------
def clean_japanese_text(text):
    text = re.sub(r'(?<=[\u3040-\u30ff\u4e00-\u9fff])\s+(?=[\u3040-\u30ff\u4e00-\u9fff])', '', text)

    lines = []
    for line in text.split("\n"):
        if len(line.strip()) > 3:
            lines.append(line)

    return "\n".join(lines)


# ---------------------------
# ✅ SPLIT TEXT
# ---------------------------
def split_text(text, chunk_size=1500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# ---------------------------
# ✅ TRANSLATE TEXT
# ---------------------------
def translate_text(text, model="llama3"):
    llm = OllamaLLM(model=model)

    prompt = PromptTemplate.from_template("""
Translate Japanese to clear, professional English.

{text}
""")

    chain = prompt | llm

    chunks = split_text(text)

    translated = []
    progress = st.progress(0)

    for i, chunk in enumerate(chunks):
        st.write(f"Translating {i+1}/{len(chunks)}...")
        result = chain.invoke({"text": chunk})
        translated.append(result)
        progress.progress((i+1)/len(chunks))

    return "\n".join(translated)


# ---------------------------
# ✅ CREATE PDF
# ---------------------------
def create_pdf(text):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    c = canvas.Canvas(temp.name, pagesize=letter)
    width, height = letter

    x, y = 40, height - 40

    for line in text.split("\n"):
        if y < 40:
            c.showPage()
            y = height - 40

        c.drawString(x, y, line[:90])
        y -= 15

    c.save()
    return temp.name


# ---------------------------
# ✅ SIMPLE TABLE EXTRACTION
# ---------------------------
def extract_table(text):
    rows = []

    for line in text.split("\n"):
        if any(keyword in line for keyword in ["日", "店", "センター", "TEL", "FAX"]):
            rows.append(line)

    df = pd.DataFrame(rows, columns=["Japanese Text"])
    return df


# ---------------------------
# ✅ TRANSLATE TABLE
# ---------------------------
def translate_table(df, model="llama3"):
    llm = OllamaLLM(model=model)

    prompt = PromptTemplate.from_template("""
Translate Japanese to English:

{text}
""")

    chain = prompt | llm

    translations = []

    for i, val in enumerate(df["Japanese Text"]):
        st.write(f"Translating table row {i+1}/{len(df)}...")
        result = chain.invoke({"text": val})
        translations.append(result)

    df["English"] = translations
    return df


# ---------------------------
# ✅ SAVE EXCEL
# ---------------------------
def save_excel(df):
    file_path = "translated_table.xlsx"
    df.to_excel(file_path, index=False)
    return file_path


# ---------------------------
# ✅ STREAMLIT APP
# ---------------------------
def main():
    st.title("📄 Japanese → English Translator (OCR + Excel)")

    model = st.selectbox("Model", ["llama3", "mistral"])

    file = st.file_uploader("Upload PDF", type=["pdf"])

    if file:
        st.success("✅ Uploaded")

        if st.button("Start Processing 🚀"):

            # STEP 1 OCR
            with st.spinner("Extracting text..."):
                text = extract_text_from_pdf(file)

            # STEP 2 CLEAN
            text = clean_japanese_text(text)

            st.subheader("✅ Cleaned Text Preview")
            st.text_area("", text[:1000], height=200)

            # STEP 3 TRANSLATE
            with st.spinner("Translating text..."):
                translated = translate_text(text, model)

            # STEP 4 PDF
            pdf_file = create_pdf(translated)

            st.success("✅ Translation Complete")

            with open(pdf_file, "rb") as f:
                st.download_button("📥 Download PDF", f, "translated.pdf")

            # STEP 5 TABLE
            df = extract_table(text)

            if len(df) > 0:
                with st.spinner("Processing table..."):
                    df = translate_table(df, model)

                st.dataframe(df)

                excel_file = save_excel(df)

                with open(excel_file, "rb") as f:
                    st.download_button("📥 Download Excel", f, "table.xlsx")


if __name__ == "__main__":
    main()