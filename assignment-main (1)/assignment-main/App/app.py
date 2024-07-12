from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
 
azure_config = {
    "base_url": "https://dono-rag-demo-resource-instance.openai.azure.com/",
    "model_deployment": "GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO",
    "model_name": "gpt-35-turbo",
    "embedding_deployment": "ADA_RAG_DONO_DEMO",
    "embedding_name": "text-embedding-ada-002",
    "api-key": "f6f4b8aec16b4094bfe0b8e063dbf1a3",
    "api_version": "2024-02-01"
    }

azure_config = {
    "base_url": os.getenv("DONO_AZURE_OPENAI_BASE_URL"),
    "model_deployment": os.getenv("DONO_AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
    "model_name": os.getenv("DONO_AZURE_OPENAI_MODEL_NAME"),
    "embedding_deployment": os.getenv("DONO_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    "embedding_name": os.getenv("DONO_AZURE_OPENAI_EMBEDDING_NAME"),
    "api-key": os.getenv("DONO_AZURE_OPENAI_API_KEY"),
    "api_version": os.getenv("DONO_AZURE_OPENAI_API_VERSION")
    }

model = AzureChatOpenAI(
    temperature=0,
    model= azure_config["model_deployment"],
    api_key= azure_config["api-key"],
    api_version= azure_config["api_version"],
    azure_endpoint= azure_config["base_url"]
    )
 
embeddings = AzureOpenAIEmbeddings(
    model= azure_config["embedding_deployment"],
    api_key= azure_config["api-key"],
    api_version= azure_config["api_version"],
    azure_endpoint= azure_config["base_url"]
    )
 

 
def user_input(user_question):
    new_db = FAISS.load_local("index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    prompt_template = """
    You are a helpful bot.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context kindly say it is not available in context and offer help, don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | model 
    st.write_stream(chain.stream({"context": docs, "question": user_question}))
 
 
 
def main():
    # Main page navigation
    st.title("Papers Assignment")
    st.markdown('''
    The indexes for the following papers are stored in the system. \\
-[Towards Efficient Generative Large Language Model Serving:A Survey from Algorithms to Systems](https://arxiv.org/pdf/2312.15234) \\
-[LLM Task Interference: An Initial Study on the Impact of Task-Switch in Conversational History](https://arxiv.org/pdf/2402.18216) \\
-[Stealing Part of a Production Language Model](https://arxiv.org/pdf/2403.06634) \\
-[TransformerFAM: Feedback attention is working memory](https://arxiv.org/pdf/2404.09173)

Go ahead and ask questions''')
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:  # Ensure user question is provided
        user_input(user_question)
 
if __name__ == "__main__":
    main()