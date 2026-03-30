import os
import json
import gc
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- KRİTİK DEĞİŞİKLİK: Hugging Face İndirme Klasörünü D Diskine Yönlendirme ---
# Bu satırlar sentence_transformers import edilmeden ÖNCE çalışmalıdır!
CACHE_DIR = r"D:\huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

# --- 1. VERİ YÜKLEME ---
print(">>> ADIM 1: Veri seti yükleniyor...")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "qa_dataset_1000.json"  # Kendi dosya adına göre kontrol et
OUTPUT_IMAGES = Path(r"D:\ödev-1")

with DATA_PATH.open(encoding="utf-8") as f:
    _raw = json.load(f)

if isinstance(_raw, list):
    selected_data = _raw
else:
    selected_data = (_raw.get("train") or []) + (_raw.get("validation") or [])

if not selected_data:
    raise FileNotFoundError(f"{DATA_PATH.name} boş veya yok.")

# Sadece metinleri alalım
raw_questions = [item['question'] for item in selected_data]
raw_answers = [item['answer'] for item in selected_data]
num_samples = len(raw_questions)
print(f"Toplam {num_samples} soru-cevap ikilisi yüklendi.\n")

# --- 2. MODEL LİSTESİ ---
# Hata aldığın 3 modeli test etmen için listeyi senin bıraktığın gibi ayarladım.
# Diğerlerini eklemek istersen daha sonra bu listeye yazabilirsin.
models_to_evaluate = [
    "intfloat/multilingual-e5-base",
]

# Açı hesaplama fonksiyonu
def get_angles(emb1, emb2):
    cos_sim = cosine_similarity(emb1, emb2)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.arccos(cos_sim)


def is_gte_multilingual_base(model_name: str) -> bool:
    return "gte-multilingual-base" in model_name.lower()


def _gte_encode_texts(tf_model, tokenizer, device, texts, batch_size, show_progress_bar, desc):
    """Yuklu GTE modeli ile tek bir metin listesini embed eder."""
    chunks = []
    n = len(texts)
    steps = range(0, n, batch_size)
    if show_progress_bar:
        try:
            from tqdm.auto import tqdm

            steps = tqdm(
                steps,
                total=(n + batch_size - 1) // batch_size,
                desc=desc,
            )
        except Exception:
            pass

    with torch.no_grad():
        for start in steps:
            batch_texts = texts[start : start + batch_size]
            batch_d = tokenizer(
                batch_texts,
                max_length=8192,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_d = {k: v.to(device) for k, v in batch_d.items()}
            outputs = tf_model(**batch_d)
            hidden = outputs.last_hidden_state
            emb = hidden[:, 0, :].float()
            emb = F.normalize(emb, p=2, dim=1)
            chunks.append(emb.cpu().numpy())

    return np.vstack(chunks)


def load_gte_multilingual_base(model_name, cache_dir):
    """
    Alibaba-NLP/gte-multilingual-base icin resmi HF yolu (AutoModel).
    SentenceTransformer.encode bu modelde RoPE/position_ids hatasina dusebiliyor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tf_model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    tf_model.to(device)
    tf_model.eval()
    return tf_model, tokenizer, device


# --- 3. ANA DÖNGÜ: Modelleri Sırasıyla Çalıştırma ---
OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)

for model_idx, model_name in enumerate(models_to_evaluate, 1):
    print("=" * 60)
    print(f"[{model_idx}/{len(models_to_evaluate)}] İŞLENİYOR: {model_name}")
    print("=" * 60)

    # Prefix (Task Text) Ayarlaması
    if "e5" in model_name.lower():
        print("  -> E5 mimarisi tespit edildi, 'query:' ve 'passage:' prefixleri ekleniyor.")
        questions = [f"query: {q}" for q in raw_questions]
        answers = [f"passage: {a}" for a in raw_answers]
    else:
        print("  -> Standart model, prefix eklenmeden saf metin kullanılıyor.")
        questions = raw_questions
        answers = raw_answers

    # Modeli Yükleme ve embedding
    print(f"  -> Model D diskine ({CACHE_DIR}) indiriliyor/yükleniyor...")
    model = None
    if is_gte_multilingual_base(model_name):
        print(
            "  -> GTE multilingual-base: transformers AutoModel yolu "
            "(SentenceTransformer bu checkpoint'te RoPE hatasi verebiliyor)."
        )
        gte_model, gte_tok, gte_dev = load_gte_multilingual_base(model_name, CACHE_DIR)
        print("  -> Soru vektörleri çıkarılıyor...")
        q_embeddings = _gte_encode_texts(
            gte_model,
            gte_tok,
            gte_dev,
            questions,
            batch_size=32,
            show_progress_bar=True,
            desc="GTE sorular",
        )
        print("  -> Cevap vektörleri çıkarılıyor...")
        a_embeddings = _gte_encode_texts(
            gte_model,
            gte_tok,
            gte_dev,
            answers,
            batch_size=32,
            show_progress_bar=True,
            desc="GTE cevaplar",
        )
        del gte_model
        del gte_tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        model = SentenceTransformer(
            model_name,
            cache_folder=CACHE_DIR,
            trust_remote_code=True,
        )
        print("  -> Soru vektörleri çıkarılıyor...")
        q_embeddings = model.encode(
            questions, normalize_embeddings=True, show_progress_bar=True
        )
        print("  -> Cevap vektörleri çıkarılıyor...")
        a_embeddings = model.encode(
            answers, normalize_embeddings=True, show_progress_bar=True
        )

    # --- BAŞARI HESAPLAMALARI ---
    print("  -> Başarı metrikleri hesaplanıyor...")
    angles_matrix = get_angles(q_embeddings, a_embeddings)
    angles_matrix_T = angles_matrix.T  # Cevap -> Soru için transpoz

    top1_qa, top5_qa = 0, 0
    top1_aq, top5_aq = 0, 0

    for i in range(num_samples):
        # Soru -> Cevap
        closest_a = np.argsort(angles_matrix[i])
        if closest_a[0] == i: top1_qa += 1
        if i in closest_a[:5]: top5_qa += 1

        # Cevap -> Soru
        closest_q = np.argsort(angles_matrix_T[i])
        if closest_q[0] == i: top1_aq += 1
        if i in closest_q[:5]: top5_aq += 1

    print("\n  [SONUÇLAR]")
    print(f"    Soru -> Cevap | Top-1: %{top1_qa / num_samples * 100:.2f} | Top-5: %{top5_qa / num_samples * 100:.2f}")
    print(f"    Cevap -> Soru | Top-1: %{top1_aq / num_samples * 100:.2f} | Top-5: %{top5_aq / num_samples * 100:.2f}\n")

    # --- t-SNE GÖRSELLEŞTİRME ---
    print("  -> t-SNE işlemi başlatılıyor (2 boyuta indirgeniyor)...")
    combined_embeddings = np.vstack((q_embeddings, a_embeddings))
    labels = ["Soru"] * num_samples + ["Cevap"] * num_samples

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(combined_embeddings)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=labels,
        palette={"Soru": "blue", "Cevap": "orange"},
        alpha=0.7,
        s=50
    )

    safe_model_name = model_name.replace("/", "_")
    plt.title(f"t-SNE Görselleştirmesi: {model_name}\n(Soru ve Cevap Temsilleri)")
    plt.xlabel("t-SNE Boyut 1")
    plt.ylabel("t-SNE Boyut 2")
    plt.legend(title="Metin Türü")
    plt.grid(True, linestyle='--', alpha=0.5)

    filename = f"tsne_{safe_model_name}.png"
    out_path = OUTPUT_IMAGES / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Grafik kaydedildi: {out_path}")

    # Bellek Temizliği
    if model is not None:
        del model
    del q_embeddings
    del a_embeddings
    gc.collect()

print("\n>>> TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI! <<<")
print(f"PNG grafikleri şuraya kaydedildi: {OUTPUT_IMAGES}")