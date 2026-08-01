"""QLoRA fine-tuning for Text-to-SQL on Spider."""

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from datasets import load_dataset


class EpochSnapshotCallback(TrainerCallback):
    """Save a clean adapter snapshot after each completed epoch."""

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        epoch = int(round(state.epoch))
        if epoch < 1:
            return control
        dest = self.snapshot_dir.parent / f"qlora-adapter-epoch{epoch}"
        dest.mkdir(parents=True, exist_ok=True)
        if model is not None:
            model.save_pretrained(dest)
        if tokenizer is not None:
            tokenizer.save_pretrained(dest)
        print(f"Epoch snapshot saved to {dest}")
        return control

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ADAPTER_DIR,
    BASE_MODEL,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    PER_DEVICE_BATCH_SIZE,
    PROCESSED_TRAIN,
    TARGET_MODULES,
)
from utils.prompts import format_zero_shot

load_dotenv()


def find_sublist_start(full: list, sub: list) -> int:
    """Return first index where sub appears in full, or -1."""
    if not sub:
        return 0
    for i in range(len(full) - len(sub) + 1):
        if full[i : i + len(sub)] == sub:
            return i
    return -1


def preprocess_dataset(tokenizer, dataset, max_seq_length: int):
    response_template = "\n<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )
    eos_token_id = tokenizer.eos_token_id

    def build_example(ex):
        prompt = format_zero_shot(tokenizer, ex["schema"], ex["question"])
        full_text = prompt + ex["query"] + tokenizer.eos_token
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = list(input_ids)

        start = find_sublist_start(input_ids, response_template_ids)
        if start != -1:
            response_start = start + len(response_template_ids)
            labels[:response_start] = [-100] * response_start
        else:
            # Fall back to training on everything.
            pass

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = dataset.map(build_example, remove_columns=dataset.column_names)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--output_dir", default=str(ADAPTER_DIR))
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--batch_size", type=int, default=PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--target_modules", default=",".join(TARGET_MODULES), help="Comma-separated LoRA target modules")
    parser.add_argument("--wandb_project", default="text-to-sql-qlora")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples for quick tests")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.target_modules.split(",")],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    raw_dataset = load_dataset("json", data_files=str(PROCESSED_TRAIN), split="train")
    if args.max_samples:
        raw_dataset = raw_dataset.select(range(args.max_samples))
    dataset = preprocess_dataset(tokenizer, raw_dataset, args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="wandb",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
        callbacks=[EpochSnapshotCallback(Path(args.output_dir))],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
