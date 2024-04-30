from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader, TextLoader
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
        prompt_model_template_inst_based = ['mistral', 'mixtral']
        prompt_model_template_inst_xml_based = ['phi3', 'tinyllama:chat']

        # Check if any of the options are in the model string
        if any(option in model for option in prompt_model_template_inst_based):
            self.prompt = PromptTemplate.from_template(
                """
                <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context
                to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
                maximum and keep the answer concise. Add citations to source documents inline of the answer. [/INST] </s>
                [INST] Question: {question}
                Context: {context}
                Answer: [/INST]
                """
            )
        elif any(option in model for option in prompt_model_template_inst_xml_based):
            self.prompt = PromptTemplate.from_template(
                """
                <|system|>
                You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
                to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
                maximum and keep the answer concise. </s>
                <|user|>
                Question: {question} 
                Context: {context} 
                Answer: <|assistant|>
                """
            )
        else:
            raise NotImplemented(
                "Unknown RAG template to use. Supported models are mistral, mixtral, tinyllama:chat")

    def ingest(self, webpage: str = None, text: str = None):

        # load the document and split it into chunks
        if webpage is not None:
            print(f"webpage {webpage}")
            loader = WebBaseLoader(webpage)
        elif text is not None:
            print(f"text {text}")
            loader = TextLoader(text)
        else:
            raise Exception("Provider a loader")
        documents = loader.load()

        # split it into chunks
        docs = self.text_splitter.split_documents(documents)

        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2")

        # load it into Chroma
        db = Chroma.from_documents(
            docs, embedding_function)

        # db = Chroma.from_documents(
        #     docs, embedding_function, persist_directory="./chroma_db")
        # db.persist()

        # load from disk
        # Note: The following code is demonstrating how to load the Chroma database from disk.
        # db = Chroma(persist_directory="./chroma_db")

        self.retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.3,
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
