from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import uvicorn
import logging

templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = FastAPI (
    title="RAG ChatBot API 문서",
    description="RAG ChatBot의 API 문서입니다.",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    import init
    logging.info("private data 초기화 완료")

@app.post("/chat")
def chat(request: Request, content: str = Form(...)):
    import rag
    return templates.TemplateResponse("index.html", {"request": request, "answer": rag.rag_query(content)})

@app.get("/")
def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
