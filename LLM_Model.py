import boto3
import streamlit as st

# Using Titan Embeddings Model to generate Embedding
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

# Vector Embeddings and Vector Store
from langchain.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name = "bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", client = bedrock)

# Data Injestion
def dataIngestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)

    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embeddings and Vector Store
def getVectorStore(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def getCluadeLLM():
    # Create the Anthropic Model
    llm = Bedrock(model_id = "ai21.j2-mid-v1", client = bedrock, 
                    model_kwargs = {'maxTokens':512})
    return llm

def getLlama2LLM():
    # Create the Anthropic Model
    llm = Bedrock(model_id = "meta.llama2-70b-chat-v1", client = bedrock, 
                    model_kwargs = {'max_gen_len':512})
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end but use atleast
summarize with 250 words with detailed explanations. If you don't know the answer, just say that you don't know,
don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant: """

PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

def getResponseLLM(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm = llm, 
        chain_type = "stuff", 
        retriever = vectorstore_faiss.as_retriever(
            search_type = "similarity", search_kwargs = {"k":3}
        ), 
        return_source_documents = True,
        chain_type_kwargs = {"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock!")

    user_question = st.text_input("Ask a question from the PDF Files")

    with st.sidebar:
        st.title("Update or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = dataIngestion()
                getVectorStore(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm = getCluadeLLM()

            # faiss_index = getVectorStore(docs)
            st.write(getResponseLLM(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm = getLlama2LLM()

            # faiss_index = getVectorStore(docs)
            st.write(getResponseLLM(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
