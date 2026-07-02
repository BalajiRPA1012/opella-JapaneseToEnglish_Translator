import streamlit as st
import fitz
import pytesseract
import pandas as pd
from PIL import Image
import io
import os
import re

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# ---------------------------
# ✅ CONFIG
# ---------------------------
SAVE_PATH = r"C:\AIFoundaryWorkshop"

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\U1074239\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


# ---------------------------
# ✅ CLEAN TEXT
# ---------------------------
def clean_text(text):
    text = re.sub(
        r'(?<=[\u3040-\u30ff\u4e00-\u9fff])\s+(?=[\u3040-\u30ff\u4e00-\u9fff])',
        '',
        text
    )

    text = re.sub(r'[a-zA-Z]{3,}', '', text)

    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 3]

    return "\n".join(lines)


# ---------------------------
# ✅ OCR
# ---------------------------
def ocr_text(image):
    image = image.convert("L")

    return pytesseract.image_to_string(
        image,
        lang="jpn",
        config="--oem 3 --psm 6"
    )


# ---------------------------
# ✅ IMPROVED TABLE EXTRACTION ✅ FIXED
# ---------------------------
def extract_table_improved(text):
    rows = []

    for line in text.split("\n"):
        if any(keyword in line for keyword in [
            "日", "店", "センター", "納品", "発注", "住所", "TEL", "FAX"
        ]):
            row = line.strip()
            if len(row) > 5:
                rows.append(row)

    structured_rows = []

    for row in rows:
        cols = re.split(r'\s{2,}|　|:|｜|\||,', row)
        cols = [c.strip() for c in cols if c.strip() != ""]

        merged = []
        buffer = ""

        for c in cols:
            if len(c) <= 2:
                buffer += c
            else:
                if buffer:
                    c = buffer + c
                    buffer = ""
                merged.append(c)

        structured_rows.append(merged)

    if not structured_rows:
        return pd.DataFrame()  # ✅ important fix

    max_cols = max(len(r) for r in structured_rows)

    normalized = [
        r + [""] * (max_cols - len(r))
        for r in structured_rows
    ]

    df = pd.DataFrame(normalized)

    return df  # ✅ CRITICAL FIX


# ---------------------------
# ✅ ADD COLUMN HEADERS SAFELY
# ---------------------------
def assign_column_headers(df):
    if df is None or df.empty:
        return df

    headers = ["Field", "Value1", "Value2", "Value3", "Value4", "Value5"]

    df.columns = headers[:len(df.columns)]

    return df


# ---------------------------
# ✅ TRANSLATION
# ---------------------------
def translate(text, model="llama3"):
    llm = OllamaLLM(model=model)

    prompt = PromptTemplate.from_template(
        "Translate Japanese to fluent professional English:\n{text}"
    )

    chain = prompt | llm

    chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]

    result = []
    for chunk in chunks:
        result.append(chain.invoke({"text": chunk}))

    return "\n".join(result)


# ---------------------------
# ✅ TRANSLATE TABLE
# ---------------------------
def translate_table(df, model):
    if df is None or df.empty:
        return df

    llm = OllamaLLM(model=model)

    prompt = PromptTemplate.from_template(
        "Translate Japanese to English:\n{text}"
    )

    chain = prompt | llm

    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: chain.invoke({"text": str(x)}) if str(x).strip() else ""
        )

    return df


# ---------------------------
# ✅ PROCESS PDF
# ---------------------------
def process_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")

    full_text = ""

    for i, page in enumerate(doc):
        st.write(f"Processing page {i+1}")

        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)

        img = Image.open(io.BytesIO(pix.tobytes("png")))

        text = ocr_text(img)
        text = clean_text(text)

        full_text += text + "\n"

    table_df = extract_table_improved(full_text)
    table_df = assign_column_headers(table_df)

    return full_text, table_df


# ---------------------------
# ✅ MAIN APP
# ---------------------------
def main():
    st.title("✅ OCR + Translation + Excel (Improved Table Version)")

    model = st.selectbox("Model", ["llama3", "mistral"])

    file = st.file_uploader("Upload PDF", type=["pdf"])

    if file and st.button("Run 🚀"):

        text, table_df = process_pdf(file)

        st.subheader("📄 Extracted Text")
        st.text_area("Extracted Text", text[:2000], height=250, label_visibility="collapsed")

        translated = translate(text, model)

        st.subheader("🌍 Translated Text")
        st.text_area("Translated Text", translated[:2000], height=250, label_visibility="collapsed")

        # Save text
        txt_path = os.path.join(SAVE_PATH, "translated_output.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(translated)

        st.success(f"Saved: {txt_path}")

        # ✅ TABLE SECTION SAFE
        if table_df is not None and not table_df.empty:
            st.subheader("📊 Extracted Table")

            df_translated = translate_table(table_df.copy(), model)

            st.dataframe(df_translated)

            excel_path = os.path.join(SAVE_PATH, "table.xlsx")
            df_translated.to_excel(excel_path, index=False)

            st.success(f"Saved Excel: {excel_path}")
        else:
            st.warning("No table detected.")


if __name__ == "__main__":
    main()