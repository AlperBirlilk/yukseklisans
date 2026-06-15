import os
os.environ["HF_HOME"] = "D:\\hf_cache"

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)

# ── 1. Test verisini yükle ───────────────────────────────────────────────────
with open("dataset.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

test_data = [q for q in raw["data"] if q["split"] == "test"]
print(f"Test soru sayısı: {len(test_data)}")

# ── 2. Yardımcı fonksiyonlar ─────────────────────────────────────────────────
def format_prompt(item: dict) -> str:
    choices_str = "\n".join([
        f"{chr(65+i)}) {c}" for i, c in enumerate(item["choices"])
    ])
    return (
        f"Answer the following multiple choice question. "
        f"Reply with only the answer letter (A, B, C, or D).\n\n"
        f"Question: {item['question']}\n{choices_str}\n\nAnswer:"
    )

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

def evaluate(model_path: str, test_data: list, model_label: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  {model_label} değerlendiriliyor: {model_path}")
    print(f"{'='*55}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    y_true, y_pred = [], []
    none_count = 0
    cat_true = defaultdict(list)
    cat_pred = defaultdict(list)

    for i, item in enumerate(test_data):
        prompt  = format_prompt(item)
        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        predicted = extract_answer(completion)
        correct   = chr(65 + item["answer"])

        # Parse edilemeyen cevapları "X" olarak say
        if predicted is None:
            predicted = "X"
            none_count += 1

        y_true.append(correct)
        y_pred.append(predicted)
        cat_true[item["category"]].append(correct)
        cat_pred[item["category"]].append(predicted)

        if (i + 1) % 50 == 0:
            acc = accuracy_score(y_true, y_pred)
            print(f"  [{i+1}/{len(test_data)}] anlık accuracy: {acc:.3f}")

    del model
    torch.cuda.empty_cache()

    # ── Genel metrikler ──────────────────────────────────────────────────────
    labels = ["A", "B", "C", "D"]
    acc       = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, labels=labels, average="macro",  zero_division=0)
    f1_weight = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    prec      = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec       = recall_score(y_true, y_pred, labels=labels, average="macro",  zero_division=0)

    print(f"\n--- {model_label} Genel Metrikler ---")
    print(f"  Accuracy          : {acc:.4f}  ({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)})")
    print(f"  F1 (macro)        : {f1_macro:.4f}")
    print(f"  F1 (weighted)     : {f1_weight:.4f}")
    print(f"  Precision (macro) : {prec:.4f}")
    print(f"  Recall (macro)    : {rec:.4f}")
    print(f"  Parse edilemeyen  : {none_count}  ({none_count/len(y_true)*100:.1f}%)")

    print(f"\n--- {model_label} Sınıf Bazında Rapor ---")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print(f"\n--- {model_label} Kategori Bazında Accuracy ---")
    print(f"{'Kategori':<40} {'Acc':>6} {'n':>4}")
    print("-" * 52)
    for cat in sorted(cat_true.keys()):
        cat_acc = accuracy_score(cat_true[cat], cat_pred[cat])
        print(f"{cat:<40} {cat_acc:>6.3f} {len(cat_true[cat]):>4}")

    return {
        "y_true":     y_true,
        "y_pred":     y_pred,
        "accuracy":   acc,
        "f1_macro":   f1_macro,
        "f1_weighted": f1_weight,
        "precision":  prec,
        "recall":     rec,
        "cat_true":   cat_true,
        "cat_pred":   cat_pred,
    }

# ── 3. Modelleri değerlendir ─────────────────────────────────────────────────
res_mA = evaluate("D:\\grpo_mA_final", test_data, "mA")
res_mB = evaluate("D:\\grpo_mB_final", test_data, "mB")

# ── 4. Karşılaştırma özeti ───────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  KARŞILAŞTIRMA ÖZETİ")
print(f"{'='*55}")
print(f"{'Metrik':<25} {'mA':>8} {'mB':>8} {'Fark':>8}")
print(f"{'-'*55}")
print(f"{'Accuracy':<25} {res_mA['accuracy']:>8.4f} {res_mB['accuracy']:>8.4f} {res_mB['accuracy']-res_mA['accuracy']:>+8.4f}")
print(f"{'F1 (macro)':<25} {res_mA['f1_macro']:>8.4f} {res_mB['f1_macro']:>8.4f} {res_mB['f1_macro']-res_mA['f1_macro']:>+8.4f}")
print(f"{'F1 (weighted)':<25} {res_mA['f1_weighted']:>8.4f} {res_mB['f1_weighted']:>8.4f} {res_mB['f1_weighted']-res_mA['f1_weighted']:>+8.4f}")
print(f"{'Precision':<25} {res_mA['precision']:>8.4f} {res_mB['precision']:>8.4f} {res_mB['precision']-res_mA['precision']:>+8.4f}")
print(f"{'Recall':<25} {res_mA['recall']:>8.4f} {res_mB['recall']:>8.4f} {res_mB['recall']-res_mA['recall']:>+8.4f}")

# Hata analizi
only_mA    = sum(1 for a, b in zip(res_mA["y_true"], res_mA["y_pred"]) if a == b and res_mB["y_pred"][res_mA["y_true"].index(a)] != a)
only_mB_c  = sum(1 for t, pa, pb in zip(res_mA["y_true"], res_mA["y_pred"], res_mB["y_pred"]) if t != pa and t == pb)
only_mA_c  = sum(1 for t, pa, pb in zip(res_mA["y_true"], res_mA["y_pred"], res_mB["y_pred"]) if t == pa and t != pb)
both_right = sum(1 for t, pa, pb in zip(res_mA["y_true"], res_mA["y_pred"], res_mB["y_pred"]) if t == pa and t == pb)
both_wrong = sum(1 for t, pa, pb in zip(res_mA["y_true"], res_mA["y_pred"], res_mB["y_pred"]) if t != pa and t != pb)

print(f"\n--- Hata Analizi ---")
print(f"  İkisi de doğru  : {both_right}")
print(f"  Sadece mA doğru : {only_mA_c}")
print(f"  Sadece mB doğru : {only_mB_c}")
print(f"  İkisi de yanlış : {both_wrong}")