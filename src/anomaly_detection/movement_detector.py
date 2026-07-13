"""
Detector de anomalias em dados de movimento do paciente.

Detecta:
- Quedas (spikes de aceleracao)
- Imobilidade prolongada
- Padroes anormais de atividade
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.ensemble import IsolationForest


class MovementAnomalyDetector:
    """Detector de anomalias em dados de acelerometro/movimento."""

    # Limiares para deteccao de quedas (baseado em literatura)
    FALL_THRESHOLD_G = 2.5      # aceleracao > 2.5g indica queda
    IMMOBILITY_HOURS = 2.0       # periodo sem movimento significativo

    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """Treina nos dados de movimento."""
        features = df[["acc_magnitude"]].copy()
        features["acc_magnitude_rolling"] = (
            features["acc_magnitude"].rolling(10, center=True).mean()
        ).fillna(features["acc_magnitude"].mean())
        self.model.fit(features)
        self.is_fitted = True

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta anomalias de movimento.
        """
        result = df.copy()

        # 1. Deteccao de quedas (spikes de aceleracao)
        result["anomaly_fall"] = result["acc_magnitude"] > self.FALL_THRESHOLD_G

        # Agrupar spikes consecutivos como uma unica queda
        fall_groups = []
        in_fall = False
        fall_start = 0

        for idx in result.index:
            if result.at[idx, "anomaly_fall"] and not in_fall:
                in_fall = True
                fall_start = idx
            elif not result.at[idx, "anomaly_fall"] and in_fall:
                in_fall = False
                duration = idx - fall_start
                if duration >= 3:  # minimo 3 amostras para considerar queda
                    fall_groups.append({
                        "start_idx": fall_start,
                        "end_idx": idx,
                        "duration_samples": duration,
                        "max_acc": result.loc[fall_start:idx, "acc_magnitude"].max(),
                    })

        # 2. Deteccao de imobilidade prolongada
        window_size = int(self.IMMOBILITY_HOURS * 3600)  # amostras em 2h
        window_size = min(window_size, len(result) // 2)
        result["anomaly_immobility"] = False

        if len(result) > window_size:
            rolling_std = (
                result["acc_magnitude"]
                .rolling(window_size, center=False)
                .std()
                .fillna(0)
            )
            result["anomaly_immobility"] = rolling_std < 0.05

        # 3. Padroes anormais (ML)
        if not self.is_fitted:
            self.fit(df)

        features = result[["acc_magnitude"]].copy()
        features["acc_magnitude_rolling"] = (
            features["acc_magnitude"].rolling(10, center=True).mean()
        ).fillna(features["acc_magnitude"].mean())

        ml_preds = self.model.predict(features)
        result["anomaly_ml"] = ml_preds == -1

        # Combinar
        result["anomaly_combined"] = (
            result["anomaly_fall"]
            | result["anomaly_immobility"]
            | result["anomaly_ml"]
        )

        # Nivel de alerta
        result["alert_level"] = "normal"
        result.loc[result["anomaly_fall"], "alert_level"] = "critical"
        result.loc[
            result["anomaly_immobility"] & (result["alert_level"] == "normal"),
            "alert_level"
        ] = "warning"
        result.loc[
            result["anomaly_ml"] & (result["alert_level"] == "normal"),
            "alert_level"
        ] = "warning"

        return result

    def generate_summary(self, result_df: pd.DataFrame) -> Dict:
        """Sumario dos resultados."""
        total = len(result_df)
        anomalies = result_df["anomaly_combined"].sum()
        falls = result_df["anomaly_fall"].sum()
        immobility = result_df["anomaly_immobility"].sum()
        criticals = (result_df["alert_level"] == "critical").sum()

        # Periodos de imobilidade
        immobility_periods = []
        in_immobility = False
        start_ts = None
        for idx, row in result_df.iterrows():
            if row["anomaly_immobility"] and not in_immobility:
                in_immobility = True
                start_ts = row["timestamp"]
            elif not row["anomaly_immobility"] and in_immobility:
                in_immobility = False
                immobility_periods.append({
                    "start": str(start_ts),
                    "end": str(row["timestamp"]),
                })

        return {
            "total_samples": total,
            "anomalies_detected": int(anomalies),
            "anomaly_rate": round(anomalies / max(total, 1), 4),
            "fall_events": int(falls),
            "immobility_periods": len(immobility_periods),
            "immobility_details": immobility_periods[:5],
            "critical_alerts": int(criticals),
            "average_acceleration": round(result_df["acc_magnitude"].mean(), 3),
            "status": "critical" if falls > 0
                      else "warning" if immobility > total * 0.1
                      else "normal",
        }


def detect_movement_anomalies(input_path: str) -> Dict:
    """Funcao CLI."""
    df = pd.read_csv(input_path)
    detector = MovementAnomalyDetector()
    results = detector.detect(df)
    summary = detector.generate_summary(results)
    return {"results": results, "summary": summary}
