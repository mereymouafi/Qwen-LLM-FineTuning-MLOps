# Architecture

**Offline:** Reviewed EN/AR instruction dataset → schema validation/deduplication → Qwen2.5-0.5B-Instruct → LoRA SFT on Colab T4 → saved adapter → evaluation against a held-out set.

**Online (starter):** Client → FastAPI `/chat` → Transformers text-generation pipeline → base Qwen model → response. The API currently serves the **base model**. Loading a fine-tuned adapter is a future integration step; do not label this endpoint as fine-tuned until implemented and tested.

**Deployment:** CPU is possible for the 0.5B model, subject to sufficient RAM and latency. For efficient CPU serving, add merged-adapter GGUF quantization and llama.cpp in a later milestone.
