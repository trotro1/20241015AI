import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from PyPDF2 import PdfReader

# Set up OpenAI API
import openai
openai.api_key = "YOUR_OPENAI_API_KEY"

# Streamlit user interface
st.title("PDFから質問回答するアプリケーション")

# Step 1: Upload a PDF from streamlit
uploaded_file = st.file_uploader("PDFをアップロードしてください", type="pdf")

if uploaded_file is not None:
    # Step 2: Extract text from the PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    if text:
        # Step 3: Pass text to langchain
        # Step 4: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        text_chunks = text_splitter.split_text(text)

        # Step 5: Get embeddings for each chunk using OpenAI Embeddings API
        embeddings = OpenAIEmbeddings()
        chunk_embeddings = [embeddings.embed_text(chunk) for chunk in text_chunks]

        # Step 7: Save embeddings to Faiss Vectorstore
        vectorstore = FAISS.from_documents(text_chunks, embeddings)

        # Step 8: User writes a question
        user_question = st.text_input("質問を入力してください")

        if user_question:
            # Step 9: Pass question to langchain
            # Step 10: Get embedding for the question using OpenAI Embeddings API
            question_embedding = embeddings.embed_text(user_question)

            # Step 12: Use the question embedding to search for similar chunks in the vectorstore
            related_chunks = vectorstore.similarity_search_by_vector(question_embedding, k=5)

            # Step 14: Create prompt with the related chunks
            context = "\n".join([chunk.page_content for chunk in related_chunks])
            prompt = f"あなたの質問に対する回答: {context}\n\n質問: {user_question}\n回答: "

            # Step 15: Ask the LLM API with the prompt
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )

            # Step 17: Display the answer in Streamlit
            st.write("回答:")
            st.write(response.choices[0].text.strip())
    else:
        st.write("PDFからテキストを抽出できませんでした。別のファイルを試してください。")