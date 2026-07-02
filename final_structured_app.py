import streamlit as st
import fitz
import pytesseract
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
import tempfile

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ---------------------------
# ✅ CONFIG
# ---------------------------
SAVE_PATH = r"C:\AIFoundaryWorkshop"

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\U1074239\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Users\U1074239\AppData\Local\Programs\Tesseract-OCR\tessdata"


# ---------------------------
# ✅ TABLE DETECTION USING OPENCV
# ---------------------------
def extract_structured_table(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Threshold
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Detect vertical lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Combine
    table_mask = cv2.add(horizontal, vertical)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 50 and h > 20:
            cells.append((x, y, w, h))

    # sort cells row-wise
    cells = sorted(cells, key=lambda x: (x[1], x[0]))

    rows = []
    current_row = []
    last_y = -1

    for (x, y, w, h) in cells:
        if abs(y - last_y) > 20:
            if current_row:
                rows.append(current_row)
            current_row = []
        current_row.append((x, y, w, h))
        last_y = y

    if current_row:
        rows.append(current_row)

    table_data = []

    for row in rows:
        row = sorted(row, key=lambda x: x[0])
        texts = []

        for (x, y, w, h) in row:
            cropped = image.crop((x, y, x + w, y + h))

            text = pytesseract.image_to_string(
                cropped,
                lang="jpn",
                config="--psm 6"
            )
            texts.append(text.strip())

        table_data.append(texts)

    return pd.DataFrame(table_data)


# ---------------------------
# ✅ OCR EXTRACTION
# ---------------------------
def process_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    full_text = ""
    tables = []

    for page_num, page in enumerate(doc):
        st.write(f"Processing page {page_num+1}")

        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # OCR full page text
        text = pytesseract.image_to_string(img, lang="jpn")
        full_text += text

        # TABLE EXTRACTION ✅
        try:
            table_df = extract_structured_table(img)
            if not table_df.empty:
                tables.append(table_df)
        except:
            pass

    return full_text, tables


# ---------------------------
# ✅ TRANSLATION
# ---------------------------
def translate(text, model="llama3"):
    llm = OllamaLLM(model=model)

    prompt = PromptTemplate.from_template("Translate to English:\n{text}")
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
    llm = OllamaLLM(model=model)
    prompt = PromptTemplate.from_template("Translate:\n{text}")
    chain = prompt | llm

    for col in df.columns:
        df[col] = df[col].apply(lambda x: chain.invoke({"text": str(x)}) if x else "")

    return df


# ---------------------------
# MAIN APP
# ---------------------------
def main():
    st.title("🚀 Advanced OCR + Structured Table Extraction")

    model = st.selectbox("Model", ["llama3", "mistral"])

    file = st.file_uploader("Upload PDF", type=["pdf"])

    if file and st.button("Process 🚀"):
        text, tables = process_pdf(file)

        st.subheader("Extracted Text")
        st.text_area("", text[:2000], height=250)

        translated_text = translate(text, model)

        st.subheader("Translated Text")
        st.text_area("", translated_text[:2000], height=250)

        # Save PDF
        pdf_path = os.path.join(SAVE_PATH, "translated_output.txt")
        with open(pdf_path, "w", encoding="utf-8") as f:
            f.write(translated_text)

        st.success(f"Saved text to {pdf_path}")

        # Process Tables ✅
        for i, df in enumerate(tables):
            st.subheader(f"Table {i+1}")

            df_translated = translate_table(df, model)

            st.dataframe(df_translated)

            excel_path = os.path.join(SAVE_PATH, f"table_{i+1}.xlsx")
            df_translated.to_excel(excel_path, index=False)

            st.success(f"Saved Excel: {excel_path}")


if __name__ == "__main__":
    main()