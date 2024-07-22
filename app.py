import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf_files(pdf_files, embedding_model_name, api_key):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    text_chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("pdf_database")
    return vector_store

def setup_qa_chain(chat_model_name, api_key):
    prompt_template = """
    Give answer to the asked question using the provided custom knowledge or given context only and if there is no related content then simply say "Your document dont contain related context to answer". Make sure to not answer incorrect.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model=chat_model_name, temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_response(user_question, chat_model_name, embedding_model_name, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=api_key)
    vector_store = FAISS.load_local("pdf_database", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = setup_qa_chain(chat_model_name, api_key)
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

def main():
    st.set_page_config(page_title="Talk to PDF", layout="wide")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{get_base64_of_image('image.png')});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Chat using Google Gemini Models")

    st.subheader("Upload your PDF Files")
    pdf_files = st.file_uploader("Upload your files", accept_multiple_files=True)

    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Google API Key:", type="password")

    default_chat_models = ["gemini-pro", "chat-model-2", "chat-model-3"]
    selected_chat_model = st.sidebar.selectbox("Select a chat model", default_chat_models, index=0)
    custom_chat_model = st.sidebar.text_input("Or enter a custom chat model name")

    if custom_chat_model:
        chat_model_name = custom_chat_model
    else:
        chat_model_name = selected_chat_model

    default_embedding_models = ["models/embedding-001", "embedding-model-2", "embedding-model-3"]
    selected_embedding_model = st.sidebar.selectbox("Select an embedding model", default_embedding_models, index=0)
    custom_embedding_model = st.sidebar.text_input("Or enter a custom embedding model name")

    if custom_embedding_model:
        embedding_model_name = custom_embedding_model
    else:
        embedding_model_name = selected_embedding_model

    if st.button("Submit data") and pdf_files:
        if embedding_model_name:
            with st.spinner("Processing the data . . ."):
                process_pdf_files(pdf_files, embedding_model_name, api_key)
                st.success("Files submitted successfully")
        else:
            st.warning("Please select or enter an embedding model.")

    if api_key:
        genai.configure(api_key=api_key)
        
        user_question = st.text_input("Ask questions from your custom knowledge base!")

        if user_question:
            with st.spinner("Generating response..."):
                response = get_response(user_question, chat_model_name, embedding_model_name, api_key)
                st.write("**Reply:** ", response)

    else:
        st.sidebar.warning("Please enter your Google API key!")
        
    st.markdown("---")
    st.write("Happy to Connect:")
    kaggle, linkedin, google_scholar, youtube, github = st.columns(5)

    image_urls = {
            "kaggle": "https://www.kaggle.com/static/images/site-logo.svg",
            "linkedin": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/600px-LinkedIn_logo_initials.png",
            "google_scholar": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Google_Scholar_logo.svg/768px-Google_Scholar_logo.svg.png",
            "youtube": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/YouTube_social_white_square_%282017%29.svg/640px-YouTube_social_white_square_%282017%29.svg.png",
            "github": "https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"
        }

    social_links = {
            "kaggle": "https://www.kaggle.com/muhammadimran112233",
            "linkedin": "https://www.linkedin.com/in/muhammad-imran-zaman",
            "google_scholar": "https://scholar.google.com/citations?user=ulVFpy8AAAAJ&hl=en",
            "youtube": "https://www.youtube.com/@consolioo",
            "github": "https://github.com/Imran-ml"
        }

    kaggle.markdown(f'<a href="{social_links["kaggle"]}"><img src="{image_urls["kaggle"]}" width="50" height="50"></a>', unsafe_allow_html=True)
    linkedin.markdown(f'<a href="{social_links["linkedin"]}"><img src="{image_urls["linkedin"]}" width="50" height="50"></a>', unsafe_allow_html=True)
    google_scholar.markdown(f'<a href="{social_links["google_scholar"]}"><img src="{image_urls["google_scholar"]}" width="50" height="50"></a>', unsafe_allow_html=True)
    youtube.markdown(f'<a href="{social_links["youtube"]}"><img src="{image_urls["youtube"]}" width="50" height="50"></a>', unsafe_allow_html=True)
    github.markdown(f'<a href="{social_links["github"]}"><img src="{image_urls["github"]}" width="50" height="50"></a>', unsafe_allow_html=True)
    st.markdown("---")

def get_base64_of_image(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode()
    return base64_str

if __name__ == "__main__":
    main()
