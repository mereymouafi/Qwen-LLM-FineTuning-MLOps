from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.copilot.inference import answer

app = FastAPI(title='Qwen AI Engineering Copilot', version='0.1.0')

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=160, ge=16, le=512)

class ChatResponse(BaseModel):
    answer: str

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/chat', response_model=ChatResponse)
def chat(request: ChatRequest):
    return ChatResponse(answer=answer(request.question, request.max_new_tokens))
