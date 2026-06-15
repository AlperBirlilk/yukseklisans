import json
import statistics
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer, util

with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

with open("output_dataset.json", "r", encoding="utf-8") as f:
    output = json.load(f)

original_list = dataset["data"]
similar_list  = output["data"]

print(f"Orijinal soru sayısı : {len(original_list)}")
print(f"Benzer soru sayısı   : {len(similar_list)}")

model = SentenceTransformer("all-MiniLM-L6-v2")

def make_text(item):
    choices_str = " ".join([f"{chr(65+i)}) {c}" for i, c in enumerate(item.get("choices", []))])
    return item["question"] + " " + choices_str

print("\nEncoding yapılıyor...")
orig_emb = model.encode([make_text(q) for q in original_list], show_progress_bar=True, convert_to_tensor=True)
sim_emb  = model.encode([make_text(q) for q in similar_list],  show_progress_bar=True, convert_to_tensor=True)

scores   = []
qualities = []
cat_scores = defaultdict(list)

for i, (orig, sim) in enumerate(zip(original_list, similar_list)):
    score = util.cos_sim(orig_emb[i], sim_emb[i]).item()
    quality = "kopya" if score > 0.95 else "ideal" if score >= 0.50 else "konu_kayması"
    scores.append(score)
    qualities.append(quality)
    cat_scores[orig.get("category", "unknown")].append(score)

total = len(scores)
dist  = Counter(qualities)

print(f"\n--- Benzerlik İstatistikleri ---")
print(f"Ortalama : {statistics.mean(scores):.4f}")
print(f"Medyan   : {statistics.median(scores):.4f}")
print(f"Min      : {min(scores):.4f}")
print(f"Max      : {max(scores):.4f}")
print(f"Std      : {statistics.stdev(scores):.4f}")

print(f"\n--- Kalite Dağılımı ---")
print(f"ideal        (0.50–0.95) : {dist['ideal']:4d}  ({dist['ideal']/total*100:.1f}%)")
print(f"kopya        (>0.95)     : {dist['kopya']:4d}  ({dist['kopya']/total*100:.1f}%)")
print(f"konu_kayması (<0.50)     : {dist['konu_kayması']:4d}  ({dist['konu_kayması']/total*100:.1f}%)")

print(f"\n--- Kategori Bazında Ortalama Benzerlik ---")
for cat, sc in sorted(cat_scores.items(), key=lambda x: statistics.mean(x[1]), reverse=True):
    print(f"  {cat:<40} {statistics.mean(sc):.4f}  (n={len(sc)})")


# 0.3 altındakileri çıkar, filtered_output_dataset.json üret
THRESHOLD = 0.30

filtered_similar = []
removed = 0

for i, (sim, score) in enumerate(zip(similar_list, scores)):
    if score >= THRESHOLD:
        filtered_similar.append(sim)
    else:
        removed += 1

filtered_output = {"data": filtered_similar}

with open("filtered_output_dataset.json", "w", encoding="utf-8") as f:
    json.dump(filtered_output, f, indent=2, ensure_ascii=False)

print(f"\n--- Filtreleme Sonucu ---")
print(f"Toplam çift     : {len(scores)}")
print(f"Kalan çift      : {len(filtered_similar)}")
print(f"Çıkarılan çift  : {removed}  ({removed/len(scores)*100:.1f}%)")
print(f"Kaydedildi      : filtered_output_dataset.json")