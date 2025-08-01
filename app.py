import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, render_template, request
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from dotenv import load_dotenv

load_dotenv()

DRIVE_FOLDER = "data_new/data"
FAISS_INDEX_PATH = "faiss_index"

if not os.path.exists(FAISS_INDEX_PATH):
    print("Creating FAISS index...")
    loader = DirectoryLoader(
        DRIVE_FOLDER,
        glob='**/*.json',
        show_progress=True,
        loader_cls=JSONLoader,
        loader_kwargs={'jq_schema': '.', 'text_content': False}
    )
    documents = loader.load()
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, chunk_overlap=500, separators=[" ", ",", "\n"]
    )
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("FAISS index created and saved.")

template = """Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi:
Nếu câu hỏi về thông tin sản phẩm, hãy trả lời về thông tin(chất liệu, kiểu dáng, thiết kế) và giá của sản phẩm đó nhưng không trả về ảnh sản phẩm.
Nếu câu hỏi là cho tôi xem ảnh sản phẩm, hãy trả về link hình ảnh của sản phẩm đó.
Nếu câu hỏi là so sánh 2 sản phẩm, hãy so sánh về chất liệu(loại vải), kiểu dáng, thiết kế.
Nếu khách hàng hỏi về size, hãy tư vấn cho họ.
Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.

{context}

Question: {input}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def load_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.4)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = load_retriever()
qa_chain = create_retrieval_chain(retriever, document_chain)


app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print(userText)
    response = qa_chain.invoke({"input": userText})
    return response["answer"]


if __name__ == "__main__":
    app.run()
