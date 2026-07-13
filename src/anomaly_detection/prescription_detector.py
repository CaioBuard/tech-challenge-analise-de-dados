"""
Detector de anomalias em prescricoes medicas.

Detecta:
- Alteracoes inesperadas de dosagem
- Interacoes medicamentosas perigosas
- Padroes anormais de prescricao
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.ensemble import IsolationForest
from collections import defaultdict


class PrescriptionAnomalyDetector:
    """Detector de anomalias em prescricoes."""

    # Interacoes medicamentosas conhecidas (simplificado)
    DANGEROUS_INTERACTIONS = {
        ("Morfina", "Heparina"): "Risco aumentado de sangramento",
        ("Captopril", "Losartana"): "Duplicacao de anti-hipertensivo",
        ("Insulina", "Captopril"): "Potencializacao de hipoglicemia",
        ("Ceftriaxona", "Heparina"): "Risco de precipitacao",
    }

    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False
        self.medication_stats = {}

    def fit(self, df: pd.DataFrame):
        """Treina com dados historicos de prescricao."""
        self.medication_stats = {}
        for med in df["medication"].unique():
            med_df = df[df["medication"] == med]
            self.medication_stats[med] = {
                "mean_dose": med_df["dose_mg"].mean(),
                "std_dose": med_df["dose_mg"].std(),
                "median_dose": med_df["dose_mg"].median(),
                "q25": med_df["dose_mg"].quantile(0.25),
                "q75": med_df["dose_mg"].quantile(0.75),
            }
        self.is_fitted = True

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta anomalias em prescricoes.
        """
        result = df.copy()

        if not self.is_fitted:
            self.fit(df)

        # 1. Doses anormais (fora do IQR)
        result["dose_expected"] = result.apply(
            lambda row: self.medication_stats.get(
                row["medication"], {}
            ).get("median_dose", 0),
            axis=1,
        )
        result["dose_zscore"] = result.apply(
            lambda row: abs(
                (row["dose_mg"] - self.medication_stats.get(
                    row["medication"], {}
                ).get("mean_dose", row["dose_mg"]))
                / max(self.medication_stats.get(
                    row["medication"], {}
                ).get("std_dose", 1), 0.01)
            ),
            axis=1,
        )
        result["anomaly_dose"] = result["dose_zscore"] > 3.0

        # 2. Interacoes perigosas (por paciente, mesma janela de 6h)
        interactions_found = []

        for patient_id in result["patient_id"].unique():
            patient_df = result[result["patient_id"] == patient_id].sort_values("timestamp")
            patient_meds = patient_df["medication"].tolist()

            for i in range(len(patient_meds)):
                for j in range(i + 1, min(i + 10, len(patient_meds))):
                    pair = tuple(sorted([patient_meds[i], patient_meds[j]]))
                    if pair in self.DANGEROUS_INTERACTIONS:
                        interactions_found.append(
                            result[result["patient_id"] == patient_id].index[i]
                        )

        result["anomaly_interaction"] = False
        result.loc[interactions_found, "anomaly_interaction"] = True

        # 3. Mudancas bruscas (dose > 3x a dose anterior do mesmo paciente+med)
        result["anomaly_sudden_change"] = False
        for (patient, med), group in result.groupby(["patient_id", "medication"]):
            if len(group) < 2:
                continue
            prev_dose = group["dose_mg"].shift(1)
            ratio = group["dose_mg"] / prev_dose.replace(0, np.nan)
            sudden = (ratio > 3.0) | (ratio < 0.33)
            result.loc[sudden.index[sudden], "anomaly_sudden_change"] = True

        # Combinar
        result["anomaly_combined"] = (
            result["anomaly_dose"]
            | result["anomaly_interaction"]
            | result["anomaly_sudden_change"]
        )

        # Nivel de alerta
        result["alert_level"] = "normal"
        result.loc[result["anomaly_interaction"], "alert_level"] = "critical"
        result.loc[
            result["anomaly_dose"] & (result["alert_level"] != "critical"),
            "alert_level"
        ] = "warning"
        result.loc[
            result["anomaly_sudden_change"] & (result["alert_level"] != "critical"),
            "alert_level"
        ] = "warning"

        return result

    def generate_summary(self, result_df: pd.DataFrame) -> Dict:
        """Sumario dos resultados."""
        total = len(result_df)
        anomalies = result_df["anomaly_combined"].sum()
        criticals = (result_df["alert_level"] == "critical").sum()
        interactions = result_df["anomaly_interaction"].sum()

        return {
            "total_prescriptions": total,
            "anomalies_detected": int(anomalies),
            "anomaly_rate": round(anomalies / max(total, 1), 4),
            "dose_anomalies": int(result_df["anomaly_dose"].sum()),
            "dangerous_interactions": int(interactions),
            "sudden_changes": int(result_df["anomaly_sudden_change"].sum()),
            "critical_alerts": int(criticals),
            "status": "critical" if interactions > 0
                      else "warning" if anomalies > total * 0.05
                      else "normal",
        }


def detect_prescription_anomalies(input_path: str) -> Dict:
    """Funcao CLI."""
    df = pd.read_csv(input_path)
    detector = PrescriptionAnomalyDetector()
    results = detector.detect(df)
    summary = detector.generate_summary(results)
    return {"results": results, "summary": summary}
