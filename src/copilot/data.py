"""Validate and format bilingual AI/ML instruction records."""
import json
from pathlib import Path

REQUIRED = ('instruction', 'response', 'language', 'topic')

def validate(record: dict) -> dict:
    for key in REQUIRED:
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f'Missing or invalid: {key}')
    if record['language'] not in ('en', 'ar'):
        raise ValueError('language must be en or ar')
    return {key: record[key].strip() for key in REQUIRED}

def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    seen = set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = validate(json.loads(line))
        key = (record['language'], record['instruction'].casefold())
        if key not in seen:
            records.append(record)
            seen.add(key)
    return records

def to_messages(record: dict) -> list[dict]:
    return [
        {'role': 'system', 'content': 'You are a careful AI/ML technical assistant. Explain clearly and say when uncertain.'},
        {'role': 'user', 'content': record['instruction']},
        {'role': 'assistant', 'content': record['response']},
    ]
