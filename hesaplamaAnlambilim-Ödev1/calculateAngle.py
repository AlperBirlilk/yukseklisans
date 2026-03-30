import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "qa_dataset_1000.json"

with DATA_PATH.open(encoding="utf-8") as f:
    _raw = json.load(f)

if isinstance(_raw, list):
    selected_data = _raw
else:
    selected_data = (_raw.get("train") or []) + (_raw.get("validation") or [])

if not selected_data:
    raise FileNotFoundError(
        f"{DATA_PATH.name} bos veya yok. Once calistirin: python dataSet_parser.py"
    )

questions = [f"query: {item['question']}" for item in selected_data]
answers = [f"passage: {item['answer']}" for item in selected_data]

num_samples = len(questions)

# 2. Modeli Yükleme ve Embedding Çıkarma
print("Model yükleniyor...")
model = SentenceTransformer("ytu-ce-cosmos/turkish-e5-large")

print("Soru ve cevap vektörleri oluşturuluyor... (Bu işlem biraz sürebilir)")
q_embeddings = model.encode(questions, normalize_embeddings=True)
a_embeddings = model.encode(answers, normalize_embeddings=True)


# 3. Açı Hesaplama Fonksiyonu
def get_angles(emb1, emb2):
    # Kosinüs benzerliği
    cos_sim = cosine_similarity(emb1, emb2)
    # Kayan nokta hatalarından dolayı -1 ile 1 arasında sınırlandırıyoruz (Clip)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    # Açı hesaplama (Radyan cinsinden bulup Dereceye çevirebilirsin, ancak sıralama için radyan yeterli)
    angles = np.arccos(cos_sim)
    return angles


# Tüm sorular ve tüm cevaplar arasındaki açı matrisi (Shape: NxN)
# Satırlar sorular, Sütunlar cevaplar
angles_matrix = get_angles(q_embeddings, a_embeddings)

# 4. Top-1 ve Top-5 Başarı Hesaplama (Sorudan -> Cevaba)
top1_correct = 0
top5_correct = 0

for i in range(num_samples):
    # i. soru için tüm cevaplarla olan açıları al
    # Açı ne kadar KÜÇÜKSE, o kadar benzerdir. Bu yüzden küçükten büyüğe sıralıyoruz.
    closest_indices = np.argsort(angles_matrix[i])

    # Gerçek cevap i indeksindedir.
    true_answer_idx = i

    if closest_indices[0] == true_answer_idx:
        top1_correct += 1

    if true_answer_idx in closest_indices[:5]:
        top5_correct += 1

print("Soru -> Cevap Eşleştirme Başarısı:")
print(f"Top-1 Accuracy: {top1_correct / num_samples * 100:.2f}%")
print(f"Top-5 Accuracy: {top5_correct / num_samples * 100:.2f}%")

# 5. Top-1 ve Top-5 Başarı Hesaplama (Cevaptan -> Soruya)
print("\nHesaplamalar yapılıyor: Cevaptan -> Soruya...")

# Matrisin transpozunu alıyoruz. Artık satırlar cevaplar, sütunlar sorular oldu.
angles_matrix_T = angles_matrix.T

top1_correct_aq = 0
top5_correct_aq = 0

for i in range(num_samples):
    # i. cevap için tüm sorularla olan açıları alıp küçükten büyüğe sıralıyoruz
    closest_indices = np.argsort(angles_matrix_T[i])

    # Gerçek soru i indeksindedir.
    true_question_idx = i

    # Top-1 Kontrolü: En yakın açıya sahip (en benzeyen) soru gerçek soru mu?
    if closest_indices[0] == true_question_idx:
        top1_correct_aq += 1

    # Top-5 Kontrolü: Gerçek soru, en benzeyen ilk 5 sorunun içinde mi?
    if true_question_idx in closest_indices[:5]:
        top5_correct_aq += 1

print(f"Cevap -> Soru Eşleştirme Başarısı:")
print(f"Top-1 Accuracy: {top1_correct_aq / num_samples * 100:.2f}%")
print(f"Top-5 Accuracy: {top5_correct_aq / num_samples * 100:.2f}%")
