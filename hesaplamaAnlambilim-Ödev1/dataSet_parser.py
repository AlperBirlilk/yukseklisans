import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_IN = ROOT / "qa_dataset.json"
DEFAULT_OUT = ROOT / "qa_dataset_1000.json"
SAMPLE_SIZE = 1000
HF_DATASET = "dilanbakr/tquad"


def answer_text(answers):
    if answers is None:
        return ""
    if isinstance(answers, dict):
        t = answers.get("text", "")
    else:
        t = answers
    if isinstance(t, list):
        return t[0] if t else ""
    return t if t is not None else ""


def row_to_qa(row):
    return {
        "question": row["questions"],
        "answer": answer_text(row["answers"]),
    }


def build_split(split_ds):
    return [row_to_qa(split_ds[i]) for i in range(len(split_ds))]


def write_qa_dataset_json(path: Path) -> None:
    """qa_dataset.json yoksa veya sifirdan uretmek icin HF'den indirir."""
    ds = load_dataset(HF_DATASET)
    payload = {
        "train": build_split(ds["train"]),
        "validation": build_split(ds["validation"]),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Wrote", path.name)
    print("train:", len(payload["train"]), "validation:", len(payload["validation"]))


def load_all_pairs(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    train = data.get("train") or []
    val = data.get("validation") or []
    pool = list(train) + list(val)
    if len(pool) < SAMPLE_SIZE:
        raise ValueError(
            f"Yetersiz ornek: {len(pool)} < {SAMPLE_SIZE}."
        )
    return pool


def main():
    p = argparse.ArgumentParser(
        description="qa_dataset.json yoksa HF'den uretir; ardindan rastgele 1000 QA yazar."
    )
    p.add_argument("--input", type=Path, default=DEFAULT_IN, help="Kaynak JSON")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Cikti JSON")
    p.add_argument("--seed", type=int, default=42, help="Tekrarlanabilirlik icin tohum")
    p.add_argument(
        "--rebuild-base",
        action="store_true",
        help="qa_dataset.json var olsa bile HF'den yeniden olustur",
    )
    args = p.parse_args()

    if args.rebuild_base or not args.input.exists():
        print("Building base JSON from Hugging Face..." if not args.input.exists() else "Rebuilding base JSON from Hugging Face...")
        write_qa_dataset_json(args.input)

    pool = load_all_pairs(args.input)
    rng = random.Random(args.seed)
    sample = rng.sample(pool, SAMPLE_SIZE)

    out = {
        "train": sample,
        "validation": [],
    }
    args.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Wrote", args.output.name)
    print("train:", len(out["train"]), "validation:", len(out["validation"]))


if __name__ == "__main__":
    main()
