import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up OpenAI API key from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Function to fetch and extract text from a URL
def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP request errors
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(paragraph.get_text() for paragraph in paragraphs)
        return text
    except Exception as e:
        st.error(f"Error fetching or parsing the URL: {e}")
        return ""

# Streamlit app
def main():
    st.title("Web-based Text Question Answering with LangChain")

    # URL input
    url_input = st.text_input("Enter the URL of the web page:")

    if url_input:
        st.write("Fetching and processing the content from the URL...")
       
        # Fetch text from URL
        web_text = fetch_text_from_url(url_input)
       
        if web_text:
            # Split the text into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, length_function=len)
            documents = text_splitter.split_text(web_text)
     
            # Set up the OpenAI model
            model = ChatOpenAI()
     
            # Set up embeddings and vectorstore
            embeddings = OpenAIEmbeddings()
            vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings)
     
            # Set up the prompt template
            template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            Answer:"""
            prompt = PromptTemplate(template=template, input_variables=["context", "question"])
     
            # Set up the chain
            chain = (
                {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
                | prompt
                | model
            )
     
            # Input question
            user_question = st.text_input("Ask a question about the content of the web page:")
            if st.button("Get Answer"):
                if user_question:
                    response = chain.invoke(user_question)
                    st.write("Answer:", response.content)
                else:
                    st.write("Please ask a question.")
        else:
            st.write("No content found at the URL or there was an error.")

if __name__ == "__main__":
    main()
