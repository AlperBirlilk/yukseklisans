import torch
print(torch.cuda.is_available())       # True olmalı
print(torch.cuda.get_device_name(0))   # örn: "NVIDIA GeForce RTX 3060"
print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")



import json

with open("dataset.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

train = [q for q in raw["data"] if q["split"] == "train"]
test  = [q for q in raw["data"] if q["split"] == "test"]

print(f"Toplam : {raw['total']}")
print(f"Train  : {len(train)}")
print(f"Test   : {len(test)}")