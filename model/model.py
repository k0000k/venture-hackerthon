from pydantic import BaseModel

class Req(BaseModel):
    question: str

class Res(BaseModel):
    answer: str