"""Configuracoes centrais usadas pela biblioteca."""

from pathlib import Path

# Caminho padrao do dataset dentro deste projeto.
DEFAULT_DATASET_PATH = Path("dataset") / "mimic-iv-clinical-database-demo-2.2"

# Caminho padrao para salvar e carregar modelos treinados.
DEFAULT_MODEL_DIR = Path("models")

# Nome do arquivo que guarda o modelo de sinais vitais.
VITAL_MODEL_FILE = "vital_signs_model.joblib"

# Nome do arquivo que guarda o modelo de prescricoes.
PRESCRIPTION_MODEL_FILE = "prescriptions_model.joblib"

# Nome do arquivo que guarda o modelo de movimentacao.
TRANSFER_MODEL_FILE = "transfers_model.joblib"

# Colunas numericas usadas pelo modelo de sinais vitais.
VITAL_FEATURE_COLUMNS = [
    "heart_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "heart_rate_delta",
    "spo2_delta",
    "systolic_bp_delta",
]

# Colunas numericas usadas pelo modelo de prescricoes.
PRESCRIPTION_FEATURE_COLUMNS = [
    "total_prescriptions",
    "unique_drugs",
    "iv_prescriptions",
    "avg_dose",
    "max_dose",
    "dose_change_count",
    "route_change_count",
]

# Colunas numericas usadas pelo modelo de movimentacao do paciente.
TRANSFER_FEATURE_COLUMNS = [
    "transfer_count",
    "unique_careunits",
    "total_los_hours",
    "avg_stay_hours",
    "min_stay_hours",
    "icu_transfer_count",
    "quick_transfer_count",
]

# Mapeamento dos itemids do MIMIC-IV para sinais vitais importantes.
VITAL_ITEMID_MAP = {
    220045: "heart_rate",
    220277: "spo2",
    220179: "systolic_bp",
    220050: "systolic_bp",
    220180: "diastolic_bp",
    220051: "diastolic_bp",
    220181: "mean_bp",
    220052: "mean_bp",
}
