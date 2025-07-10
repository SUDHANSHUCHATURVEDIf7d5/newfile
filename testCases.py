import streamlit as st
from pypdf import PdfReader  
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community import *
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import httpx


# openai.api_key = "sk-h4SzToxOqOneSAXq191PXA"

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    return text

def generate_test_cases(pdf_text):
    # prompt = f"Create detailed test cases based on the following requirements:\n\n{text}\n\nTest Cases:\n"
    prompt = PromptTemplate(
    input_variables=["pdf_text"],
    template="Generate test cases from the following text:\n{pdf_text}"
)
    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure_ai/genailab-maas-DeepSeek-R1",  # or "gpt-4", "gpt-3.5-turbo"
        api_key="sk-h4SzToxOqOneSAXq191PXA",
        temperature=0.3,
        max_tokens=500
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    # return chat.invoke(    
    #         {"role": "system", "content": "You are a helpful assistant that creates software test cases."},
    #         {"role": "user", "content": prompt},
    # )
    return chain.run(pdf_text)
    
    # return response.content
    # return response['choices'][0]['message']['content']

def main():
    st.title("PDF to Test Cases Generator")
    uploaded_file = st.file_uploader("Upload a PDF with requirements", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        st.subheader("Extracted Requirements Text")
        st.text_area("Requirements", pdf_text, height=300)
        
        if st.button("Generate Test Cases"):
            with st.spinner("Generating test cases..."):
                test_cases = generate_test_cases(pdf_text)
            st.subheader("Generated Test Cases")
            st.text_area("Test Cases", test_cases, height=400)

if __name__ == "__main__":
    main()
