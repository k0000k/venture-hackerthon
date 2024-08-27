from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
import os
import configuration

embedding_func = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

SOURCE_PATH = configuration.SOURCE_DIR
DB_DIR_PATH = configuration.DB_DIR

docs = []
for filename in os.listdir(SOURCE_PATH):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(SOURCE_PATH, filename)
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
            doc = Document(page_content=text, metadata={"source": filename})
            docs.append(doc)

# Save to disk
vectorstore = Chroma.from_documents(
                     documents=docs,
                     embedding=embedding_func,
                     persist_directory=DB_DIR_PATH
                     )
