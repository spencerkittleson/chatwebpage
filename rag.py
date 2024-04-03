from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import os
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


class ChatWebPage:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, model: str, base_url: str = None):
        self.model = ChatOllama(model=model)
        if base_url is not None:
            self.model.base_url = base_url
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100, separators=[" ", ",", "\n", os.linesep])

        # https://ollama.com/library/tinyllama:latest/blobs/af0ddbdaaa26

        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, webpage: str):

        # load the document and split it into chunks
        print(f"webpage {webpage}")
        loader = WebBaseLoader(webpage)
        documents = loader.load()

        # split it into chunks
        docs = self.text_splitter.split_documents(documents)

        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2")

        # load it into Chroma
        db = Chroma.from_documents(
            docs, embedding_function, persist_directory="./chroma_db")
        db.persist()

        # load from disk
        # Note: The following code is demonstrating how to load the Chroma database from disk.
        # db = Chroma(persist_directory="./chroma_db")

        self.retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        print(f'query {query}')
        if not self.chain:
            return "Please, add a webpage first"

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
