from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

import configuration

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

embedding_func = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

DB_DIR_PATH = configuration.DB_DIR

vectorstore_disk = Chroma(
                        persist_directory=DB_DIR_PATH,
                        embedding_function=embedding_func
                   )

retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

LLM_PROMPT_TEMPLATE = configuration.PROMPT

llm_prompt = PromptTemplate.from_template(LLM_PROMPT_TEMPLATE)

def format_docs(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    return context

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)

def rag_query(question):
    return rag_chain.invoke(question)