import re
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------
# 1. VERİ HAZIRLIĞI (10 Soru Seçimi)
# ---------------------------------------------------------
print("1. Veri seti yükleniyor...")
dataset = load_dataset("ytu-ce-cosmos/gsm8k_tr", split="train[:10]")

# ---------------------------------------------------------
# 2. MODEL YÜKLEME (Sadece Qwen Modeli Yükleniyor)
# ---------------------------------------------------------
print("2. Qwen modeli yükleniyor...")
model_id = "Qwen/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Padding token ayarı (Batch generation yapabilmek için şarttır)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.float16  # GPU üzerinde yüksek hız için float16 tutulmalı
)


# ---------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------
def extract_answer_from_truth(text):
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = re.sub(r'[^\d.-]', '', ans)
        return ans
    return ""


def extract_answer_from_generation(text):
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    return ""


def calculate_jaccard_difference(text1, text2):
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    if not words1 and not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return 1.0 - (len(intersection) / len(union))


# ---------------------------------------------------------
# 3. HIZLANDIRILMIŞ CEVAP ÜRETİMİ (Batch Processing)
# ---------------------------------------------------------
print("3. Cevaplar paralel olarak üretiliyor (Hızlandırılmış mod)...")
all_prompts_results = []

for i, data in enumerate(dataset):
    question = data['question']
    ground_truth_ans = extract_answer_from_truth(data['answer'])

    prompt = f"Soru: {question}\nCevap:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    # HIZLANDIRMA NOKTASI: 10 cevabı döngüyle tek tek üretmek yerine,
    # 'num_return_sequences=10' ile ekran kartına tek seferde paralel olarak ürettiriyoruz.
    # max_new_tokens değerini de 100'e çekerek gereksiz uzatmaları engelledik.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=10,
            pad_token_id=tokenizer.pad_token_id
        )

    # Üretilen tüm metinleri tek seferde decode et
    generated_texts = [
        tokenizer.decode(out[inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for out in outputs
    ]

    all_prompts_results.append({
        'truth': ground_truth_ans,
        'generations': generated_texts
    })
    print(f"Soru {i + 1}/10 tamamlandı.")

# ---------------------------------------------------------
# 4. METRİK HESAPLAMA (Vektörsüz, Jet Hızında Jaccard)
# ---------------------------------------------------------
print("4. Metrikler hesaplanıyor...")
x_differences = []
y_accuracies = []

for item in all_prompts_results:
    truth = item['truth']
    gens = item['generations']

    accuracies = [1 if extract_answer_from_generation(g) == truth else 0 for g in gens]
    pairs = list(itertools.combinations(range(10), 2))

    for idx1, idx2 in pairs:
        avg_acc = (accuracies[idx1] + accuracies[idx2]) / 2.0
        difference = calculate_jaccard_difference(gens[idx1], gens[idx2])

        x_differences.append(difference)
        y_accuracies.append(avg_acc)

# ---------------------------------------------------------
# 5. GRAFİK ÇİZİMİ VE KAYDETME
# ---------------------------------------------------------
print("5. Grafik çiziliyor ve kaydediliyor...")
plt.figure(figsize=(10, 8))
plt.scatter(x_differences, y_accuracies, alpha=0.5, color='blue', edgecolors='k')

plt.title("Base Model (Qwen2.5-Math-1.5B) - Hızlı Dağılım Analizi")
plt.xlabel("Cevaplar Arası Farklılık (1 - Jaccard Similarity) \n<-- Benzer | Farklı -->")
plt.ylabel("Ortalama Doğruluk \n<-- Yanlış | Doğru -->")

plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.3)

plt.text(0.8, 0.95, 'Hedef (Mükemmel Yer)', fontsize=12, color='green',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

plt.grid(True, alpha=0.3)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

output_image_path = "model_cevap_daagilimi.png"
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
print(f"Grafik kaydedildi: {output_image_path}")

plt.show()
print("Pipeline başarıyla tamamlandı!")