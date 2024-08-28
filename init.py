from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
import os
import shutil
import logging

import configuration

logging.basicConfig(level=logging.INFO)

embedding_func = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

SOURCE_PATH = configuration.SOURCE_DIR
DB_DIR_PATH = configuration.DB_DIR

if os.path.exists(DB_DIR_PATH) and os.path.isdir(DB_DIR_PATH):
    shutil.rmtree(DB_DIR_PATH)
    logging.info("이미 존재하는 DB 폴더가 삭제되었습니다.")

docs = []
for filename in os.listdir(SOURCE_PATH):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(SOURCE_PATH, filename)
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            doc = Document(page_content=page.extract_text(), metadata={"source": filename})
            docs.append(doc)

# Save to disk
vectorstore = Chroma.from_documents(
                     documents=docs,
                     embedding=embedding_func,
                     persist_directory=DB_DIR_PATH
                     )

logging.info("private data 초기화 완료")