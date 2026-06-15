import os
os.environ["HF_HOME"] = "D:\\hf_cache"

import re
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import json

# ── 1. Veriyi yükle ──────────────────────────────────────────────────────────
with open("dataset.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

with open("output_dataset.json", "r", encoding="utf-8") as f:
    similar_raw = json.load(f)

train_data   = [q for q in raw["data"] if q["split"] == "train"]
similar_data = similar_raw["data"]

min_len = min(len(train_data), len(similar_data))
train_data   = train_data[:min_len]
similar_data = similar_data[:min_len]

print(f"Eğitim çifti sayısı: {min_len}")

# ── 2. Prompt formatı ─────────────────────────────────────────────────────────
def format_prompt(item: dict) -> str:
    choices_str = "\n".join([
        f"{chr(65+i)}) {c}" for i, c in enumerate(item["choices"])
    ])
    return (
        f"Answer the following multiple choice question. "
        f"Reply with only the answer letter (A, B, C, or D).\n\n"
        f"Question: {item['question']}\n{choices_str}\n\nAnswer:"
    )

# ── 3. Dataset hazırla ────────────────────────────────────────────────────────
hf_dataset = Dataset.from_list([
    {
        "prompt":                format_prompt(orig),
        "answer_letter":         chr(65 + orig["answer"]),
        "similar_prompt":        format_prompt(sim),
        "similar_answer_letter": sim["answer_letter"],
        "category":              orig.get("category", ""),
    }
    for orig, sim in zip(train_data, similar_data)
])

# ── 4. Cevap çıkarma ──────────────────────────────────────────────────────────
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

# ── 5. Reward fonksiyonu ──────────────────────────────────────────────────────
def reward_mB(prompts, completions, answer_letter, similar_prompt,
              similar_answer_letter, **kwargs) -> list[float]:
    rewards = []

    for completion, correct, sim_prompt, sim_correct in zip(
        completions, answer_letter, similar_prompt, similar_answer_letter
    ):
        pred_orig    = extract_answer(completion)
        orig_correct = (pred_orig == correct)

        inputs = tokenizer(sim_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        sim_completion = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        pred_sim      = extract_answer(sim_completion)
        sim_correct_bool = (pred_sim == sim_correct)

        if orig_correct and sim_correct_bool:
            reward = 1.0
        elif orig_correct or sim_correct_bool:
            reward = 0.3
        else:
            reward = 0.0

        rewards.append(reward)

    return rewards

# ── 6. Model ve tokenizer ─────────────────────────────────────────────────────
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ── 7. GRPO config ────────────────────────────────────────────────────────────
config = GRPOConfig(
    output_dir                  = "D:\\grpo_mB",
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
    gradient_checkpointing      = True,
)

# ── 8. Trainer ────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model         = model,
    args          = config,
    train_dataset = hf_dataset,
    reward_funcs  = reward_mB,
)

trainer.train()
trainer.save_model("D:\\grpo_mB_final")
tokenizer.save_pretrained("D:\\grpo_mB_final")
print("mB eğitimi tamamlandı → D:\\grpo_mB_final")