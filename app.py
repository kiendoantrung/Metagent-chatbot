# from chatbot import chatbot
import pickle
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from flask import Flask, render_template, request
from langchain.document_loaders import UnstructuredFileLoader
os.environ["OPENAI_API_KEY"] = "your api key"
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.json_loader import JSONLoader


DRIVE_FOLDER = "data_new/data"

loader = DirectoryLoader(DRIVE_FOLDER, glob='**/*.txt', show_progress=True, loader_cls=UnstructuredFileLoader)

documents = loader.load()



# Text Splitter
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000, chunk_overlap=500, separators=[" ", ",", "\n"]
    )


docs = text_splitter.split_documents(documents)

import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(max_retries=10)
vectorstore = FAISS.from_documents(docs, embeddings)

with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

template = """Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi:
Nếu câu hỏi về thông tin sản phẩm, hãy trả lời về thông tin(chất liệu, kiểu dáng, thiết kế) và giá của sản phẩm đó nhưng không trả về ảnh sản phẩm.
Nếu câu hỏi là cho tôi xem ảnh sản phẩm, hãy trả về link hình ảnh của sản phẩm đó.
Nếu câu hỏi là so sánh 2 sản phẩm, hãy so sánh về chất liệu(loại vải), kiểu dáng, thiết kế.
Nếu khách hàng hỏi về size, hãy tư vấn cho họ.
Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.

{context}

Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

def load_retriever():
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever

chain_type_kwargs = {"prompt": prompt}
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.4), 
    chain_type="stuff", 
    retriever=load_retriever(), 
    chain_type_kwargs=chain_type_kwargs)

app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print(userText)
    return str(qa.run(userText))


if __name__ == "__main__":
    app.run()