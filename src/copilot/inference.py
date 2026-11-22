"""Lazily load a small instruction model for CPU or GPU inference."""
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_pipeline():
    from transformers import pipeline
    model = os.getenv('MODEL_ID', 'Qwen/Qwen2.5-0.5B-Instruct')
    return pipeline('text-generation', model=model, device_map='auto')

def answer(question: str, max_new_tokens: int = 160) -> str:
    messages = [
        {'role': 'system', 'content': 'You are a careful bilingual AI/ML technical assistant. If unsure, say so.'},
        {'role': 'user', 'content': question},
    ]
    result = get_pipeline()(messages, max_new_tokens=max_new_tokens, do_sample=False)
    generated = result[0]['generated_text']
    if isinstance(generated, list):
        return generated[-1]['content']
    return str(generated)
