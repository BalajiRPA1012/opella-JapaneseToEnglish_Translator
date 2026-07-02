import streamlit as st
import fitz
import pytesseract
import pandas as pd
from PIL import Image
import io
import os
import re
import time

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ---------------------------
# CONFIG
# ---------------------------
BASE_PATH = r"C:\AIFoundaryWorkshop"
INPUT_FOLDER = os.path.join(BASE_PATH, "Input")
OUTPUT_FOLDER = os.path.join(BASE_PATH, "Translated_Output")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ✅ Tesseract
TESS_PATH = os.path.join(BASE_PATH, "Tesseract-OCR", "tesseract.exe")
if os.path.exists(TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH

# ---------------------------
# UI STYLE
# ---------------------------
st.set_page_config(layout="wide")

st.markdown("""
<style>
body { background-color: #F7EFE6; }
h1 { color: #042B0B; }
.stButton>button { background-color: #042B0B; color: white; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ✅ LOGO
logo_path = os.path.join(BASE_PATH, "Opella_Logo.png")
if os.path.exists(logo_path):
    st.image(logo_path, width=200)

# ---------------------------
# OCR WITH POSITION
# ---------------------------
def ocr_with_pos(img):
    df = pytesseract.image_to_data(img, lang="jpn", output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna()
    df = df[df.text.str.strip() != ""]
    return df

# ---------------------------
# BUILD TABLE
# ---------------------------
def build_table(df):
    rows = []

    df["row_id"] = df["top"] // 20
    grouped = df.groupby("row_id")

    for _, row in grouped:
        words = sorted(zip(row["left"], row["text"]))
        rows.append([w[1] for w in words])

    if not rows:
        return pd.DataFrame()

    max_cols = max(len(r) for r in rows)
    table = [r + [""]*(max_cols-len(r)) for r in rows]

    return pd.DataFrame(table)

# ---------------------------
# CLEAN TEXT
# ---------------------------
def clean_text(text):
    text = re.sub(r'(?<=[\u3040-\u30ff\u4e00-\u9fff])\s+(?=[\u3040-\u30ff\u4e00-\u9fff])', '', text)
    return text

# ---------------------------
# TRANSLATE
# ---------------------------
def translate(text, model):
    llm = OllamaLLM(model=model)

    prompt = PromptTemplate.from_template(
        "Translate Japanese to professional English:\n{text}"
    )

    chain = prompt | llm

    chunks = [text[i:i+1200] for i in range(0, len(text), 1200)]
    return "\n".join([chain.invoke({"text": c}) for c in chunks])

# ---------------------------
# FAST TABLE TRANSLATE
# ---------------------------
def translate_table(df, model):
    flat = df.astype(str).values.flatten().tolist()
    joined = "\n".join(flat)

    translated = translate(joined, model)
    lines = translated.split("\n")

    result = []
    idx = 0

    for _ in range(df.shape[0]):
        row = []
        for _ in range(df.shape[1]):
            row.append(lines[idx] if idx < len(lines) else "")
            idx += 1
        result.append(row)

    return pd.DataFrame(result)

# ---------------------------
# PROCESS SINGLE PDF
# ---------------------------
def process_pdf(path, model):
    start = time.time()

    doc = fitz.open(path)
    full_text = ""

    tables = []

    for i, page in enumerate(doc):
        st.write(f"Processing page {i+1}...")

        pix = page.get_pixmap(matrix=fitz.Matrix(1.4,1.4))
        img = Image.open(io.BytesIO(pix.tobytes()))

        df_pos = ocr_with_pos(img)
        table = build_table(df_pos)

        if not table.empty:
            tables.append(table)

        full_text += " ".join(df_pos["text"].tolist()) + "\n"

    full_text = clean_text(full_text)
    translated_text = translate(full_text, model)

    filename = os.path.splitext(os.path.basename(path))[0]

    # ✅ Save text
    txt_path = os.path.join(OUTPUT_FOLDER, filename + "_EN.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(translated_text)

    # ✅ Save Excel
    excel_path = os.path.join(OUTPUT_FOLDER, filename + ".xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        sheet_index = 1

        for table in tables:
            table_en = translate_table(table, model)
            table_en.to_excel(writer, sheet_name=f"Table_{sheet_index}", index=False)
            sheet_index += 1

        pd.DataFrame({"Translated Text":[translated_text]}).to_excel(
            writer, sheet_name="Full_Text", index=False
        )

    end = time.time()

    return translated_text, excel_path, round(end-start,2)

# ---------------------------
# BATCH PROCESS
# ---------------------------
def process_all(model):
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".pdf")]

    results = []

    for f in files:
        st.write(f"Starting: {f}")
        path = os.path.join(INPUT_FOLDER, f)

        text, excel, t = process_pdf(path, model)

        st.success(f"✅ Completed {f} in {t} sec")
        results.append((f,t))

    return results

# ---------------------------
# UI
# ---------------------------
def main():
    st.title("🚀 Opella Enterprise OCR Translator")

    model = st.selectbox("Model", ["llama3","mistral"])

    col1, col2 = st.columns(2)

    # SINGLE FILE
    with col1:
        st.subheader("📄 Single File")

        file = st.file_uploader("Upload PDF", type=["pdf"])

        if file and st.button("Process File 🚀"):

            st.info("⏳ Processing started...")

            temp_path = os.path.join(BASE_PATH,"temp.pdf")
            with open(temp_path,"wb") as f:
                f.write(file.read())

            text, excel, t = process_pdf(temp_path, model)

            st.success(f"✅ Completed in {t} sec")

            st.text_area("Translated", text[:2000])

            with open(excel,"rb") as f:
                st.download_button("Download Excel", f, file_name=os.path.basename(excel))

    # BATCH
    with col2:
        st.subheader("📂 Batch Processing")

        st.code(INPUT_FOLDER)

        if st.button("Process All Files 🚀"):
            st.info("⏳ Batch started...")
            results = process_all(model)
            st.success("✅ All files processed!")

    st.write(f"📁 Output folder: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()