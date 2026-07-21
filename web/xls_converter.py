"""Conversao de XLS/XLSX para o formato esperado pela biblioteca clinica."""

from pathlib import Path

import pandas as pd


# Abas esperadas no arquivo Excel.
EXPECTED_SHEETS = {
    "vitals": "vitals",
    "prescriptions": "prescriptions",
    "transfers": "transfers",
}


def _normalize_columns(frame):
    """Normaliza nomes de colunas para facilitar leitura do Excel."""
    # Cria uma copia para preservar o dataframe original.
    result = frame.copy()
    # Remove espacos, troca por minusculo e substitui espacos internos por underline.
    result.columns = [str(column).strip().lower().replace(" ", "_") for column in result.columns]
    # Retorna o dataframe com colunas padronizadas.
    return result


def _clean_records(frame):
    """Remove linhas vazias e converte valores pandas para dicionarios simples."""
    # Remove linhas completamente vazias.
    cleaned = frame.dropna(how="all").copy()
    # Converte datas para texto, mantendo compatibilidade com a lib principal.
    for column in cleaned.columns:
        # Detecta colunas de data/hora pelo nome.
        if column in {"charttime", "starttime", "stoptime", "intime", "outtime"}:
            # Converte valores para datetime quando possivel.
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
            # Formata datas validas como string.
            cleaned[column] = cleaned[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    # Troca NaN por None para gerar dicionarios mais limpos.
    cleaned = cleaned.where(pd.notna(cleaned), None)
    # Retorna lista de registros no formato esperado por patient_input.
    return cleaned.to_dict(orient="records")


def _read_sheet_if_exists(excel_file, normalized_sheet_names, sheet_name):
    """Le uma aba se ela existir, senao retorna lista vazia."""
    # Busca o nome real da aba ignorando maiusculas e espacos.
    actual_name = normalized_sheet_names.get(sheet_name)
    # Se a aba nao existir, retorna lista vazia.
    if actual_name is None:
        return []
    # Le a aba encontrada.
    frame = pd.read_excel(excel_file, sheet_name=actual_name)
    # Normaliza as colunas da aba.
    frame = _normalize_columns(frame)
    # Retorna registros limpos.
    return _clean_records(frame)


def excel_to_patient_input(file_path):
    """Converte um arquivo Excel para o dicionario aceito pela lib."""
    # Garante que o caminho seja um Path.
    path = Path(file_path)
    # Abre o arquivo Excel com pandas.
    excel_file = pd.ExcelFile(path)
    # Cria mapa de nomes normalizados para nomes reais das abas.
    normalized_sheet_names = {name.strip().lower(): name for name in excel_file.sheet_names}
    # Monta a estrutura esperada pelo ClinicalAnomalyDetector.
    return {
        "vitals": _read_sheet_if_exists(excel_file, normalized_sheet_names, EXPECTED_SHEETS["vitals"]),
        "prescriptions": _read_sheet_if_exists(excel_file, normalized_sheet_names, EXPECTED_SHEETS["prescriptions"]),
        "transfers": _read_sheet_if_exists(excel_file, normalized_sheet_names, EXPECTED_SHEETS["transfers"]),
    }
