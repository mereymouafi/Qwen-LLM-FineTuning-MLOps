"""Colab GPU LoRA SFT entrypoint. Supply your own reviewed JSONL."""
import argparse
import json
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from src.copilot.data import load_jsonl, to_messages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Reviewed training JSONL')
    parser.add_argument('--output', default='adapters/qwen-ml-copilot')
    parser.add_argument('--config', default='configs/training.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA GPU; use Colab T4')
    cfg = json.loads(Path(args.config).read_text())
    records = load_jsonl(args.data)
    if len(records) < 2:
        raise ValueError('Need at least two training examples')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def format_row(row):
        messages = to_messages(row)
        return {'text': tokenizer.apply_chat_template(messages, tokenize=False)}
    ds = Dataset.from_list(records).map(format_row)
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], torch_dtype=torch.float16)
    peft_config = LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'], target_modules=['q_proj','v_proj'], task_type=TaskType.CAUSAL_LM)
    trainer = SFTTrainer(model=model, peft_config=peft_config, train_dataset=ds,
        processing_class=tokenizer, args=SFTConfig(output_dir='checkpoints',
        num_train_epochs=cfg['epochs'], per_device_train_batch_size=cfg['batch_size'],
        learning_rate=cfg['learning_rate'], max_length=cfg['max_length'],
        dataset_text_field='text', report_to='none', fp16=True))
    trainer.train()
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print('Saved adapter to', args.output)

if __name__ == '__main__':
    main()
