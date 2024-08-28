from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import model.model as model

templates = Jinja2Templates(directory="templates")

app = FastAPI (
    title="RAG ChatBot API 문서",
    description="RAG ChatBot의 API 문서입니다.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    import init

@app.post("/chat")
def chat(request: Request, content: str = Form(...)):
    import rag
    return templates.TemplateResponse("index.html", {"request": request, "answer": rag.rag_query(content)})

@app.get("/")
def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/insurance-chat", response_model=model.Res)
def api_chat(request: model.Req):
    import rag
    return {"answer": rag.rag_query(request.question, "kb")}

@app.post("/api/student-chat", response_model=model.Res)
def api_chat(request: model.Req):
    import rag
    return {"answer": rag.rag_query(request.question, "student")}

@app.post("/api/company-chat", response_model=model.Res)
def api_chat(request: model.Req):
    import rag
    return {"answer": rag.rag_query(request.question, "company")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
