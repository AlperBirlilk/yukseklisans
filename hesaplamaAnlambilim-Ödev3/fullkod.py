import os, re, gc, time, itertools
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# ====================================================================
# D DİSKİ ALAN VE KLASÖR AYARLARI (YENİ)
# ====================================================================
# 1. HuggingFace indirmelerini D diskine yönlendir
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/models"

# 2. Çıktıların (Grafiklerin) kaydedileceği D diski dizini
OUTPUT_DIR = "D:/analiz_sonuclari"

# Gerekli klasörleri otomatik oluştur
os.makedirs("D:/hf_cache", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[SİSTEM] Önbellek ve grafik çıktıları {OUTPUT_DIR} dizinine yönlendirildi.")

# ====================================================================
# SİSTEM VE SSL YAMASI (KURUMSAL AĞ ENGELLERİNİ AŞMA)
# ====================================================================
import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["GIT_SSL_NO_VERIFY"] = "true"

os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

print("[BAĞLANTI] HuggingFace Hub erişimi kontrol ediliyor...")
try:
    from huggingface_hub import HfApi
    import httpx

    client = httpx.Client(verify=False)
    HfApi().list_models(limit=1)
    print("   HuggingFace Hub: OK")
except Exception as e:
    print(f"   [UYARI] Hub'a erişilemiyor: {e}")
    print("   → SSL bypass uygulandı, indirme aşamasına geçiliyor...")

# ====================================================================
# KONFİGÜRASYON
# ====================================================================
print("[BAŞLANGIÇ] Parametreler yükleniyor...")

NUM_TRAIN = 100
NUM_TEST = 20
BATCH_SIZE = 4
N_SAMPLES = 10
MAX_TOKENS = 128
USE_4BIT = True

PRODUCTION = False
BASE_MODEL_ID = "Qwen/Qwen3-4B"

MODEL_CONFIGS = {
    "multiSFT_Model": {"proxy": "Qwen/Qwen2.5-1.5B-Instruct", "adapter": None},
    "GRPO_Model": {"proxy": "yeyeyeyeye2/Qwen2.5-1.5B-Open-R1-GRPO-gsm8k", "adapter": None},
    "SFT_GRPO_Model": {"proxy": "Essi-Narim/qwen2.5b-sft-grpo-reasoning", "adapter": None},
    "multiSFT_GRPO_Model": {"proxy": "BounharAbdelaziz/Qwen2.5-3B-GRPO-Math-GSM8K", "adapter": None},
}

PAIRS = list(itertools.combinations(range(N_SAMPLES), 2))

try:
    import flash_attn

    ATTN_IMPL = "flash_attention_2"
    print("   [INFO] Flash Attention 2 aktif")
except ImportError:
    ATTN_IMPL = "sdpa"
    print("   [INFO] Flash Attention 2 yok → SDPA")

torch.backends.cuda.matmul.allow_tf32 = True


# ====================================================================
# YARDIMCI FONKSİYONLAR
# ====================================================================

def extract_truth(text):
    if "####" in text:
        return re.sub(r'[^\d.-]', '', text.split("####")[-1].strip())
    return ""


def extract_pred(text):
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else ""


def jaccard_dist(t1, t2):
    w1 = set(re.findall(r'\w+', t1.lower()))
    w2 = set(re.findall(r'\w+', t2.lower()))
    if not w1 and not w2:
        return 0.0
    return 1.0 - len(w1 & w2) / len(w1 | w2)


def load_model_and_tokenizer(model_id, adapter=None):
    print(f"   [YÜKLEME] {model_id}", end="", flush=True)
    t0 = time.time()

    # cache_dir eklenerek modellerin zorunlu olarak D diskine inmesi garantileniyor
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=False, cache_dir="D:/hf_cache/models"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "token": False,
        "cache_dir": "D:/hf_cache/models"  # Büyük model dosyalarını D diskine yazar
    }
    if torch.cuda.is_available():
        kwargs["attn_implementation"] = ATTN_IMPL
    if USE_4BIT and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        print(" [4bit]", end="", flush=True)
    else:
        kwargs["torch_dtype"] = torch.float16

    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if adapter:
        is_local = adapter.startswith("./") or adapter.startswith("/")
        if is_local and not os.path.exists(adapter):
            print(f"\n   [UYARI] Local adaptör yok: {adapter} → adaptörsüz devam")
        else:
            mdl = PeftModel.from_pretrained(mdl, adapter, cache_dir="D:/hf_cache/models")
            print(f" + LoRA({adapter})", end="", flush=True)

    mdl.eval()
    print(f" → hazır ({time.time() - t0:.1f}s)")
    return mdl, tok


def save_plot(x_diff, y_acc, model_tag, split_name):
    avg_acc = np.mean(y_acc) * 100
    avg_diff = np.mean(x_diff)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(x_diff, y_acc, alpha=0.1, color='darkblue', edgecolors='none', s=15)
    ax.set_title(f"Model Dağılım Analizi: {model_tag} ({split_name})", fontsize=14, fontweight='bold')
    ax.set_xlabel(
        "Cevaplar Arası Farklılık (1 - Jaccard Similarity)\n<-- Benzer Çözümler (Ezber) | Farklı Çözüm Yolları (Yaratıcı) -->",
        fontsize=11)
    ax.set_ylabel("Ortalama Doğruluk\n<-- Yanlış Cevaplar (0.0) | Doğru Cevaplar (1.0) -->", fontsize=11)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.4)
    ax.text(0.75, 0.93, 'MÜKEMMEL YER\n(Sağ Üst)', fontsize=11, color='green', fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='green', boxstyle='round,pad=0.5'))
    ax.text(0.02, 0.93, f"Ort. Doğruluk: %{avg_acc:.2f}\nOrt. Farklılık: {avg_diff:.2f}", fontsize=10,
            transform=ax.transAxes, bbox=dict(facecolor='lightgray', alpha=0.5))
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # GÜNCELLEME: Grafik dosyası D:\analiz_sonuclari altına kaydedilir
    fname = os.path.join(OUTPUT_DIR, f"GRAFIK_{model_tag}_{split_name}.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fname


# ====================================================================
# ADIM 1: VERİ SETİ
# ====================================================================
print("\n[ADIM 1/3] GSM8K-TR veriseti yükleniyor...")
# cache_dir eklenerek veri setinin D diskine indirilmesi sağlandı
full_ds = load_dataset("ytu-ce-cosmos/gsm8k_tr", split="train", cache_dir="D:/hf_cache/datasets")
total = NUM_TRAIN + NUM_TEST
if len(full_ds) < total:
    raise ValueError(f"Yetersiz veri: {len(full_ds)} < {total}")

splits = {
    "Egitim_Sorulari": full_ds.select(range(NUM_TRAIN)),
    "Test_Sorulari": full_ds.select(range(NUM_TRAIN, total)),
}
print(f"   Eğitim: {len(splits['Egitim_Sorulari'])} | Test: {len(splits['Test_Sorulari'])}")

# ====================================================================
# ADIM 2+3: MODEL DÖNGÜSÜ VE ANALİZ
# ====================================================================
mod_str = "PRODUCTION (BASE + LoRA)" if PRODUCTION else "PROXY (HF'den küçük modeller)"
print(f"\n[ADIM 2/3] {len(MODEL_CONFIGS)} model | Mod: {mod_str}\n")

for m_idx, (model_tag, cfg) in enumerate(MODEL_CONFIGS.items(), 1):
    print(f"{'=' * 62}")
    print(f"[MODEL {m_idx}/{len(MODEL_CONFIGS)}] {model_tag}")
    print(f"{'=' * 62}")

    if not PRODUCTION:
        print(f"   Mod: proxy → {cfg['proxy']}")
        model, tok = load_model_and_tokenizer(cfg["proxy"])
    else:
        adapter = cfg["adapter"]
        if adapter is None:
            print(f"   Mod: base → {BASE_MODEL_ID}")
            model, tok = load_model_and_tokenizer(BASE_MODEL_ID)
        else:
            print(f"   Mod: base + LoRA → {BASE_MODEL_ID} + {adapter}")
            model, tok = load_model_and_tokenizer(BASE_MODEL_ID, adapter=adapter)

    for split_name, dataset in splits.items():
        n = len(dataset)
        print(f"\n   [ANALİZ] {split_name} | {n} soru | batch={BATCH_SIZE}")
        t_split = time.time()

        questions = [d['question'] for d in dataset]
        truths = [extract_truth(d['answer']) for d in dataset]
        x_diff, y_acc = [], []

        with tqdm(total=n, desc=f"   {model_tag}/{split_name}", unit="soru", ncols=90) as pbar:
            for b_start in range(0, n, BATCH_SIZE):
                bq = questions[b_start: b_start + BATCH_SIZE]
                bgt = truths[b_start: b_start + BATCH_SIZE]

                enc = tok(
                    [f"Soru: {q}\nCevap:" for q in bq],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(model.device)

                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=MAX_TOKENS,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        num_return_sequences=N_SAMPLES,
                        pad_token_id=tok.pad_token_id,
                        use_cache=True,
                    ).cpu()

                for i, gt in enumerate(bgt):
                    texts = []
                    for j in range(N_SAMPLES):
                        idx = i * N_SAMPLES + j
                        gen_tokens = out[idx][enc.input_ids.shape[1]:]
                        text = tok.decode(gen_tokens, skip_special_tokens=True)
                        texts.append(text)

                    accs = [1 if extract_pred(t) == gt else 0 for t in texts]
                    for pa, pb in PAIRS:
                        x_diff.append(jaccard_dist(texts[pa], texts[pb]))
                        y_acc.append((accs[pa] + accs[pb]) / 2.0)

                pbar.update(len(bq))

        elapsed = time.time() - t_split
        print(f"   [SONUÇ] {elapsed:.0f}s | "
              f"Ort.Doğruluk: %{np.mean(y_acc) * 100:.2f} | "
              f"Ort.Farklılık: {np.mean(x_diff):.3f}")

        fname = save_plot(x_diff, y_acc, model_tag, split_name)
        print(f"   [KAYDEDİLDİ] {fname}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n   [VRAM TEMİZLENDİ] {model_tag}")

print(f"\n{'=' * 62}")
print(f"[BAŞARI] Tüm analizler tamamlandı! Grafiklere '{OUTPUT_DIR}' klasöründen ulaşabilirsiniz.")
print(f"{'=' * 62}")