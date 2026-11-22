import pytest
from src.copilot.data import validate, to_messages

def test_valid_record():
    record = validate({'instruction':' What is LoRA? ', 'response':'Low-rank adaptation.', 'language':'en', 'topic':'llm'})
    assert record['instruction'] == 'What is LoRA?'
    assert to_messages(record)[-1]['role'] == 'assistant'

def test_invalid_language():
    with pytest.raises(ValueError):
        validate({'instruction':'x','response':'y','language':'fr','topic':'llm'})
