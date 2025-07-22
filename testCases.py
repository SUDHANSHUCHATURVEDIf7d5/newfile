import streamlit as st
import fitz  # PyMuPDF
import ollama
import pandas as pd
from fpdf import FPDF
from io import BytesIO

# -------------------------------
# Extract PDF text
# -------------------------------
def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

# -------------------------------
# Prompt for LLM
# -------------------------------
def build_prompt(user_story_text):
    return f"""
You are a senior QA engineer. Given the user story below, generate detailed test cases, covering:
- Functional test cases
- Edge cases
- UI validations (if applicable)
- Negative test cases

Use structured format:
Test Case ID | Title | Description | Steps | Expected Result | Type

User Story:
{user_story_text}
"""

# -------------------------------
# Call Ollama LLM
# -------------------------------
def generate_test_cases(prompt, model="deepseek-coder:6.7b-instruct"):
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

# -------------------------------
# Parse LLM output to DataFrame
# -------------------------------
def parse_test_cases_to_df(text):
    rows = []
    for line in text.splitlines():
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 6:
                rows.append(parts)
    return pd.DataFrame(rows, columns=[
        "Test Case ID", "Title", "Description", "Steps", "Expected Result", "Type"
    ])

# -------------------------------
# Export PDF
# -------------------------------
def dataframe_to_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    line_height = 10
    col_widths = [20, 35, 50, 40, 40, 20]

    for i, row in df.iterrows():
        for j, item in enumerate(row):
            pdf.multi_cell(col_widths[j], line_height, str(item), border=1,
                           ln=3 if j == len(row) - 1 else 0, max_line_height=pdf.font_size)
        pdf.ln(line_height)
    return pdf.output(dest='S').encode('latin-1')  # returns PDF as bytes

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Test Case Generator", layout="wide")
st.title("🧪 Test Case Generator from User Story (PDF)")
pdf_file = st.file_uploader("Upload a User Story PDF", type=["pdf"])

if pdf_file:
    user_story_text = extract_text_from_pdf(pdf_file)
    st.subheader("📜 Extracted User Story")
    st.text_area("Text", user_story_text, height=300)

    if st.button("Generate Test Cases"):
        with st.spinner("Generating test cases via LLM..."):
            prompt = build_prompt(user_story_text)
            test_cases = generate_test_cases(prompt)
            st.subheader("✅ Generated Test Cases")
            st.code(test_cases)

            df = parse_test_cases_to_df(test_cases)

            if not df.empty:
                st.dataframe(df, use_container_width=True)

                # CSV download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download as CSV", csv, file_name="test_cases.csv", mime="text/csv")

                # PDF download
                pdf_bytes = dataframe_to_pdf(df)
                st.download_button("📄 Download as PDF", pdf_bytes, file_name="test_cases.pdf", mime="application/pdf")
            else:
                st.warning("⚠️ Could not parse structured test cases from LLM response.")
