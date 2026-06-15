import os
os.environ["HF_HOME"] = "D:\\hf_cache"

import re
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import json

with open("dataset.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

train_data = [q for q in raw["data"] if q["split"] == "train"]

def format_prompt(item: dict) -> str:
    choices_str = "\n".join([
        f"{chr(65+i)}) {c}" for i, c in enumerate(item["choices"])
    ])
    return (
        f"Answer the following multiple choice question. "
        f"Reply with only the answer letter (A, B, C, or D).\n\n"
        f"Question: {item['question']}\n{choices_str}\n\nAnswer:"
    )

hf_dataset = Dataset.from_list([
    {
        "prompt":        format_prompt(q),
        "answer":        q["answer"],
        "answer_letter": chr(65 + q["answer"]),
        "category":      q["category"],
    }
    for q in train_data
])

def extract_answer(text: str) -> str | None:
    text = text.strip()
    patterns = [
        r'^([A-D])[)\.\s]',
        r'[Aa]nswer[:\s]+([A-D])',
        r'\(([A-D])\)',
        r'^([A-D])$',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()
    return None

def reward_mA(prompts, completions, answer_letter, **kwargs) -> list[float]:
    rewards = []
    for completion, correct in zip(completions, answer_letter):
        predicted = extract_answer(completion)
        rewards.append(1.0 if predicted == correct else 0.0)
    return rewards

# 1.5B → 0.5B, VRAM 6GB'a sığar
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

config = GRPOConfig(
    output_dir                  = "D:\\grpo_mA",
    num_train_epochs            = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 16,
    learning_rate               = 1e-5,
    max_completion_length       = 16,
    num_generations             = 2,
    temperature                 = 0.9,
    logging_steps               = 10,
    save_steps                  = 200,
    bf16                        = True,
    gradient_checkpointing      = True,   # ekstra bellek tasarrufu
)

trainer = GRPOTrainer(
    model         = model,
    args          = config,
    train_dataset = hf_dataset,
    reward_funcs  = reward_mA,
)

trainer.train()
trainer.save_model("D:\\grpo_mA_final")
tokenizer.save_pretrained("D:\\grpo_mA_final")
print("mA eğitimi tamamlandı → D:\\grpo_mA_final")