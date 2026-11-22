from fastapi.testclient import TestClient
from api.main import app

def test_health():
    assert TestClient(app).get('/health').json() == {'status':'ok'}

def test_invalid_request():
    assert TestClient(app).post('/chat', json={'question':''}).status_code == 422
