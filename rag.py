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

def set_prompt(category):
    if (category == "kb"):
        return configuration.INSURANCE_PROMPT
    elif (category == "student"):
        return configuration.STUDENT_PROMPT
    elif (category == "company"):
        return configuration.COMPANY_PROMPT

def format_docs(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    return context

def rag_query(question, category):
    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3}, filter={"source": category})

    LLM_PROMPT_TEMPLATE = set_prompt(category)

    llm_prompt = PromptTemplate.from_template(LLM_PROMPT_TEMPLATE)

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
    )

    return rag_chain.invoke(question)