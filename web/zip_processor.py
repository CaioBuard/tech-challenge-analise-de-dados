"""Processamento de ZIP com arquivos Excel, audio e video."""

from pathlib import Path
import shutil
import tempfile
import uuid
import zipfile

from clinical_anomaly_detection import ClinicalAnomalyDetector

from xls_converter import excel_to_patient_input


# Extensoes aceitas como planilha.
EXCEL_EXTENSIONS = {".xls", ".xlsx"}

# Extensoes tratadas como video.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Extensoes tratadas como audio.
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def _safe_extract(zip_file, destination):
    """Extrai um ZIP evitando escrita fora da pasta temporaria."""
    # Resolve a pasta de destino para comparar caminhos absolutos.
    destination_root = Path(destination).resolve()
    # Percorre cada item do ZIP.
    for member in zip_file.infolist():
        # Ignora diretorios explicitos.
        if member.is_dir():
            continue
        # Resolve o caminho final do arquivo.
        target_path = (destination_root / member.filename).resolve()
        # Bloqueia caminhos maliciosos com ../ para fora da pasta.
        if destination_root not in target_path.parents and target_path != destination_root:
            raise ValueError(f"Caminho inseguro dentro do ZIP: {member.filename}")
        # Cria diretorios pais quando necessario.
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Abre o arquivo de origem dentro do ZIP.
        with zip_file.open(member) as source:
            # Abre o arquivo de destino no disco.
            with target_path.open("wb") as target:
                # Copia o conteudo do ZIP para o destino.
                shutil.copyfileobj(source, target)


def extract_zip(zip_bytes):
    """Extrai bytes de ZIP para uma pasta temporaria e retorna arquivos."""
    # Cria uma pasta temporaria isolada para este upload.
    temp_dir = Path(tempfile.mkdtemp(prefix="clinical_zip_"))
    # Define caminho temporario do ZIP recebido.
    zip_path = temp_dir / "upload.zip"
    # Salva os bytes recebidos em disco para abrir com zipfile.
    zip_path.write_bytes(zip_bytes)
    # Abre o ZIP para extracao.
    with zipfile.ZipFile(zip_path) as zip_file:
        # Extrai arquivos com protecao contra zip slip.
        _safe_extract(zip_file, temp_dir / "extracted")
    # Lista todos os arquivos extraidos.
    files = [path for path in (temp_dir / "extracted").rglob("*") if path.is_file()]
    # Retorna a pasta temporaria e os arquivos encontrados.
    return temp_dir, files


def classify_files(files):
    """Classifica arquivos extraidos por tipo."""
    # Lista de metadados classificados.
    classified = []
    # Percorre cada arquivo extraido.
    for file_path in files:
        # Le a extensao em minusculo.
        suffix = file_path.suffix.lower()
        # Classifica planilhas.
        if suffix in EXCEL_EXTENSIONS:
            classified.append({"type": "xls", "path": file_path})
        # Classifica videos.
        elif suffix in VIDEO_EXTENSIONS:
            classified.append({"type": "video", "path": file_path})
        # Classifica audios.
        elif suffix in AUDIO_EXTENSIONS:
            classified.append({"type": "audio", "path": file_path})
        # Registra arquivos ignorados.
        else:
            classified.append({"type": "ignored", "path": file_path})
    # Retorna classificacao completa.
    return classified


def _process_excel(file_path, output_root):
    """Processa uma planilha e executa a biblioteca de anomalias."""
    # Converte a planilha para o formato patient_input.
    patient_input = excel_to_patient_input(file_path)
    # Cria o detector usando caminhos relativos ao projeto raiz.
    detector = ClinicalAnomalyDetector(
        dataset_path="dataset/mimic-iv-clinical-database-demo-2.2",
        model_dir="models",
    )
    # Garante modelos treinados ou carregados.
    detector.train_if_needed()
    # Cria uma pasta propria para as imagens desta planilha.
    report_dir = Path(output_root) / file_path.stem
    # Gera alertas, scores e imagens.
    report = detector.generate_patient_report(patient_input, output_dir=report_dir)
    # Retorna resultado estruturado para a tela.
    return {
        "type": "xls",
        "filename": file_path.name,
        "patient_input": patient_input,
        "alerts": report["alerts"],
        "scores": report["scores"],
        "images": report["images"],
    }


def process_classified_files(classified_files, output_root):
    """Processa arquivos classificados e retorna resultados para a UI."""
    # Lista final de resultados.
    results = []
    # Percorre cada arquivo classificado.
    for item in classified_files:
        # Busca o caminho do arquivo.
        file_path = item["path"]
        # Processa planilhas.
        if item["type"] == "xls":
            results.append(_process_excel(file_path, output_root))
        # Retorna mensagem pendente para video.
        elif item["type"] == "video":
            results.append({"type": "video", "filename": file_path.name, "message": "Pendente: implementar retorno do video."})
        # Retorna mensagem pendente para audio.
        elif item["type"] == "audio":
            results.append({"type": "audio", "filename": file_path.name, "message": "Pendente: implementar retorno do audio."})
        # Registra arquivos ignorados.
        else:
            results.append({"type": "ignored", "filename": file_path.name, "message": "Arquivo ignorado: tipo nao suportado."})
    # Retorna todos os resultados.
    return results


def process_zip_bytes(zip_bytes, generated_root):
    """Processa bytes de ZIP usando o fluxo local."""
    # Cria um identificador unico para as imagens deste upload.
    upload_id = str(uuid.uuid4())
    # Cria pasta onde os PNGs gerados serao salvos.
    output_root = Path(generated_root) / upload_id
    # Garante que a pasta exista.
    output_root.mkdir(parents=True, exist_ok=True)
    # Extrai o ZIP para pasta temporaria.
    temp_dir, files = extract_zip(zip_bytes)
    # Classifica arquivos extraidos.
    classified = classify_files(files)
    # Processa os arquivos classificados.
    results = process_classified_files(classified, output_root)
    # Retorna metadados do processamento.
    return {"upload_id": upload_id, "temp_dir": str(temp_dir), "results": results}
