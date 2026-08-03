"""Model loading and SQL generation utilities."""

import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MAX_SEQ_LENGTH
from utils.prompts import extract_sql

load_dotenv()

if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])


def load_model_and_tokenizer(model_name: str, adapter_path: str | None = None):
    """Load a causal LM and tokenizer. Uses 4-bit quantization by default."""
    use_cpu = os.environ.get("USE_CPU", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_cpu or not torch.cuda.is_available():
        # ponytail: CPU path uses float32 to avoid unsupported fp16 ops on CPU.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )

    if adapter_path is not None and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_sql(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens.
    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[0][prompt_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return extract_sql(text)
