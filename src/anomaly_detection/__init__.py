"""Módulo de Detecção de Anomalias - Sinais vitais, prescrições, movimentação."""


def analyze(input_path: str, data_type: str = "vitals"):
    print(f"[ANOMALY] Iniciando detecção de anomalias: {input_path}")
    print(f"[ANOMALY] Tipo de dados: {data_type}")

    if data_type == "vitals":
        from .vitals_detector import detect_vital_anomalies
        detect_vital_anomalies(input_path)
    elif data_type == "prescription":
        from .prescription_detector import detect_prescription_anomalies
        detect_prescription_anomalies(input_path)
    elif data_type == "movement":
        from .movement_detector import detect_movement_anomalies
        detect_movement_anomalies(input_path)
