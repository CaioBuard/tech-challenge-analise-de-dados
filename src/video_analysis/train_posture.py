"""
Treinamento do classificador postural com YOLOv8-cls.
Usa o dataset Sitting Posture Classification (Roboflow).
"""

from ultralytics import YOLO
from pathlib import Path
import argparse


def train_posture_classifier(
    data_dir: str = "data/datasets/video/Sitting Posture Classification.folder",
    model_size: str = "n",
    epochs: int = 50,
    imgsz: int = 224,
    batch: int = 16,
):
    """Treina YOLOv8-cls para classificacao postural."""

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {data_path}")

    # Verificar estrutura
    train_dir = data_path / "train"
    val_dir = data_path / "valid"

    if not train_dir.exists():
        raise FileNotFoundError(f"Diretorio train nao encontrado: {train_dir}")

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"[TRAIN] Dataset encontrado: {data_path}")
    print(f"[TRAIN] Classes detectadas ({len(classes)}):")
    for i, cls in enumerate(classes):
        n_imgs = len(list((train_dir / cls).glob("*")))
        print(f"  [{i}] {cls[:60]} -> {n_imgs} imagens (train)")

    # Carregar modelo pretrained
    model_name = f"yolov8{model_size}-cls.pt"
    print(f"\n[TRAIN] Carregando modelo {model_name}...")
    model = YOLO(model_name)

    # Configurar caminho de saida
    output_dir = Path("data/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    project_name = "posture_classifier"
    run_name = f"yolov8{model_size}_{epochs}e"

    # Treinar
    print(f"\n[TRAIN] Iniciando treinamento...")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(output_dir / project_name),
        name=run_name,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        patience=10,          # early stopping
        save=True,
        save_period=10,
        plots=True,
        device="mps",         # Apple Silicon GPU
        verbose=True,
    )

    # Avaliar no conjunto de validacao
    print(f"\n[TRAIN] Avaliando no conjunto de validacao...")
    metrics = model.val(data=str(data_path), split="valid")

    # Exportar modelo
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    final_model = output_dir / "yolov8_posture.pt"
    import shutil
    shutil.copy(best_model, final_model)
    print(f"\n[TRAIN] Modelo salvo em: {final_model}")
    print(f"[TRAIN] Treinamento concluido!")

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar classificador postural YOLOv8")
    parser.add_argument("--data", default="data/datasets/video/Sitting Posture Classification.folder")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    train_posture_classifier(
        data_dir=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )
