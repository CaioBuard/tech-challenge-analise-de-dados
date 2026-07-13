import yaml
import shutil
from pathlib import Path

BASE = Path("data/datasets/video")
sources = [
    ("detection/posture-final",   {"Bad": "bad", "Good": "good"}),
    ("detection/posture-duduk",   {"Unlabeled": "bad", "bad_posture": "bad", "good_posture": "good"}),
    ("detection/posture-v1",      {"Bad": "bad", "Good": "good"}),
]
target = BASE / "detection/posture-unified"

# Limpar destino
if target.exists():
    shutil.rmtree(target)

for split in ["train", "valid", "test"]:
    (target / split / "images").mkdir(parents=True, exist_ok=True)
    (target / split / "labels").mkdir(parents=True, exist_ok=True)

total = 0
for src_dir, class_map in sources:
    src = BASE / src_dir
    for split in ["train", "valid", "test"]:
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if not img_dir.exists():
            continue
        for img_file in img_dir.glob("*"):
            # Copiar imagem
            dst_img = target / split / "images" / f"{src_dir.split('/')[-1]}_{img_file.name}"
            shutil.copy2(img_file, dst_img)

            # Processar label
            lbl_file = lbl_dir / (img_file.stem + ".txt")
            if lbl_file.exists():
                new_lines = []
                for line in lbl_file.read_text().splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_class = int(parts[0])
                    # Mapear via indice
                    with open(src / "data.yaml") as f:
                        cfg = yaml.safe_load(f)
                    old_name = cfg["names"][old_class]
                    new_name = class_map.get(old_name)
                    if new_name is None:
                        continue  # pular classes nao mapeadas
                    new_class = 0 if new_name == "good" else 1
                    parts[0] = str(new_class)
                    new_lines.append(" ".join(parts))
                if new_lines:
                    dst_lbl = target / split / "labels" / (dst_img.stem + ".txt")
                    dst_lbl.write_text("\n".join(new_lines) + "\n")
            total += 1

# data.yaml unificado
data_yaml = {
    "path": str(target),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 2,
    "names": ["good", "bad"],
}
with open(target / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print(f"Dataset unificado criado: {target}")
print(f"Total imagens: {total}")
print(f"Classes: good, bad")
