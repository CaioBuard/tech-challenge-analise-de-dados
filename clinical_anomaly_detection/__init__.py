"""Biblioteca para deteccao de anomalias."""

# Expoe a classe principal para permitir: from clinical_anomaly_detection import ClinicalAnomalyDetector.
from .service import ClinicalAnomalyDetector

# Define quais nomes publicos este pacote entrega para outros codigos.
__all__ = ["ClinicalAnomalyDetector"]
