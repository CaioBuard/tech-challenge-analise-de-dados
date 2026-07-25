"""Processamento de ZIP com arquivos Excel, audio e video."""

from pathlib import Path
import shutil
import tempfile
import uuid
import zipfile

# Extensoes aceitas como planilha.
EXCEL_EXTENSIONS = {".xls", ".xlsx"}

# Extensoes tratadas como video.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Extensoes tratadas como audio.
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def _has_video_container_signature(file_path):
    """Verifica a assinatura de contêineres de vídeo suportados pelo app."""
    with Path(file_path).open("rb") as handle:
        header = handle.read(32)
    return (
        header[4:8] == b"ftyp"
        or (header[:4] == b"RIFF" and header[8:12] == b"AVI ")
        or header[:4] == b"\x1aE\xdf\xa3"
    )


def _has_excel_container_signature(file_path):
    """Verifica se o arquivo tem estrutura XLS ou XLSX antes de processa-lo."""
    with Path(file_path).open("rb") as handle:
        header = handle.read(8)
    return header.startswith(b"PK") or header == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _safe_extract(zip_file, destination):
    """Extrai um ZIP evitando escrita fora da pasta temporaria."""
    destination_root = Path(destination).resolve()

    for member in zip_file.infolist():
        if member.is_dir():
            continue

        target_path = (destination_root / member.filename).resolve()
        if destination_root not in target_path.parents and target_path != destination_root:
            raise ValueError(f"Caminho inseguro dentro do ZIP: {member.filename}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with zip_file.open(member) as source:
            with target_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def extract_zip(zip_bytes):
    """Extrai bytes de ZIP para uma pasta temporaria e retorna arquivos."""
    temp_dir = Path(tempfile.mkdtemp(prefix="clinical_zip_"))
    zip_path = temp_dir / "upload.zip"
    zip_path.write_bytes(zip_bytes)

    with zipfile.ZipFile(zip_path) as zip_file:
        _safe_extract(zip_file, temp_dir / "extracted")

    files = [path for path in (temp_dir / "extracted").rglob("*") if path.is_file()]
    return temp_dir, files


def classify_files(files):
    """Classifica arquivos extraidos por tipo."""
    classified = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix in EXCEL_EXTENSIONS:
            classified.append({"type": "xls", "path": file_path})
        elif suffix in VIDEO_EXTENSIONS:
            classified.append({"type": "video", "path": file_path})
        elif suffix in AUDIO_EXTENSIONS:
            classified.append({"type": "audio", "path": file_path})
        else:
            classified.append({"type": "ignored", "path": file_path})
    return classified


def _process_excel(file_path, output_root):
    """Processa uma planilha e executa a biblioteca de anomalias."""
    if not _has_excel_container_signature(file_path):
        return None

    from clinical_anomaly_detection import ClinicalAnomalyDetector
    from xls_converter import excel_to_patient_input

    patient_input = excel_to_patient_input(file_path)
    detector = ClinicalAnomalyDetector(
        dataset_path="dataset/mimic-iv-clinical-database-demo-2.2",
        model_dir="models",
    )
    detector.train_if_needed()

    report_dir = Path(output_root) / file_path.stem
    report = detector.generate_patient_report(patient_input, output_dir=report_dir)
    return {
        "type": "xls",
        "filename": file_path.name,
        "patient_input": patient_input,
        "alerts": report["alerts"],
        "scores": report["scores"],
        "images": report["images"],
    }


def _process_video(file_path, output_root):
    """Avalia um video e grava seus relatorios na pasta deste upload."""
    if not _has_video_container_signature(file_path):
        return None

    from video_analysis import VideoEvaluator

    report_dir = Path(output_root) / file_path.stem
    print(f"[VIDEO] Iniciando avaliacao: {file_path.name}", flush=True)
    try:
        evaluation = VideoEvaluator(report_dir=str(report_dir)).evaluate(str(file_path))
    except ValueError:
        print(f"[VIDEO] Arquivo ignorado por nao ser um video legivel: {file_path.name}", flush=True)
        return None

    print(f"[VIDEO] Avaliacao concluida: {file_path.name}", flush=True)
    return {
        "type": "video",
        "filename": file_path.name,
        "evaluation": evaluation,
    }


def process_classified_files(classified_files, output_root):
    """Processa arquivos classificados e retorna resultados para a UI."""
    results = []

    for item in classified_files:
        file_path = item["path"]

        try:
            if item["type"] == "xls":
                result = _process_excel(file_path, output_root)
                if result is not None:
                    results.append(result)
            elif item["type"] == "video":
                result = _process_video(file_path, output_root)
                if result is not None:
                    results.append(result)
            elif item["type"] == "audio":
                from interview_analisys.transcribe_and_analyze_audio import executar_transcricao_e_analise

                result = executar_transcricao_e_analise(str(file_path))
                if result is not None:
                    results.append(result)
            else:
                results.append({
                    "type": "ignored",
                    "filename": file_path.name,
                    "message": "Arquivo ignorado: tipo nao suportado.",
                })
        except Exception as error:
            print(f"[WEB] Falha ao processar {file_path.name}: {error}", flush=True)
            results.append({
                "type": "error",
                "filename": file_path.name,
                "message": "Nao foi possivel processar este arquivo. Os demais arquivos do ZIP continuaram sendo analisados.",
            })

    return results


def process_zip_bytes(zip_bytes, generated_root):
    """Processa bytes de ZIP usando o fluxo local."""
    upload_id = str(uuid.uuid4())
    output_root = Path(generated_root) / upload_id
    output_root.mkdir(parents=True, exist_ok=True)

    temp_dir, files = extract_zip(zip_bytes)
    classified = classify_files(files)
    results = process_classified_files(classified, output_root)

    return {"upload_id": upload_id, "temp_dir": str(temp_dir), "results": results}
