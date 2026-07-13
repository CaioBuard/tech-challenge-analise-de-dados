"""
Detector de anomalias em sinais vitais.

Usa IsolationForest + regras clinicas para detectar:
- Taquicardia / Bradicardia
- Hipertensao / Hipotensao
- Hipoxia (baixa saturacao de O2)
- Febre / Hipotermia
- Taquipneia
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class VitalSignsAnomalyDetector:
    """Detector de anomalias em sinais vitais."""

    # Limiares clinicos (referencia: OMS / protocolos hospitalares)
    CLINICAL_THRESHOLDS = {
        "heart_rate": {"low": 50, "high": 100, "critical_low": 40, "critical_high": 130},
        "blood_pressure_sys": {"low": 90, "high": 140, "critical_low": 70, "critical_high": 180},
        "blood_pressure_dia": {"low": 60, "high": 90, "critical_low": 40, "critical_high": 110},
        "spo2": {"low": 92, "high": 100, "critical_low": 85, "critical_high": 100},
        "respiratory_rate": {"low": 10, "high": 20, "critical_low": 6, "critical_high": 30},
        "temperature": {"low": 35.5, "high": 37.5, "critical_low": 34, "critical_high": 39},
    }

    def __init__(self, contamination: float = 0.05):
        """
        Args:
            contamination: Proporcao esperada de anomalias (0.0 a 0.5)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples="auto",
        )
        self.is_fitted = False
        self.feature_columns = []

    def fit(self, df: pd.DataFrame):
        """Treina o detector em dados normais."""
        feature_cols = [c for c in df.columns
                        if c in self.CLINICAL_THRESHOLDS]
        self.feature_columns = feature_cols

        data = df[feature_cols].copy()
        data_scaled = self.scaler.fit_transform(data)
        self.model.fit(data_scaled)
        self.is_fitted = True
        print(f"[ANOMALY-VITALS] Modelo treinado em {len(df)} registros, "
              f"{len(feature_cols)} features")

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta anomalias combinando:
        1. Regras clinicas (limiares fisiologicos)
        2. IsolationForest (padroes estatisticos)

        Returns:
            DataFrame com colunas adicionais: anomaly_score, anomaly_clinical,
            anomaly_ml, anomaly_combined, alert_level, alert_message
        """
        result = df.copy()

        # --- Metodo 1: Regras clinicas ---
        clinical_alerts = []
        for idx, row in result.iterrows():
            row_alerts = []
            for col, thresholds in self.CLINICAL_THRESHOLDS.items():
                if col not in result.columns:
                    continue
                value = row[col]
                if value <= thresholds["critical_low"]:
                    row_alerts.append({
                        "vital": col,
                        "value": value,
                        "severity": "critical",
                        "direction": "low",
                    })
                elif value < thresholds["low"]:
                    row_alerts.append({
                        "vital": col,
                        "value": value,
                        "severity": "warning",
                        "direction": "low",
                    })
                elif value >= thresholds["critical_high"]:
                    row_alerts.append({
                        "vital": col,
                        "value": value,
                        "severity": "critical",
                        "direction": "high",
                    })
                elif value > thresholds["high"]:
                    row_alerts.append({
                        "vital": col,
                        "value": value,
                        "severity": "warning",
                        "direction": "high",
                    })

            clinical_alerts.append(row_alerts)

        result["alerts_clinical"] = clinical_alerts
        result["anomaly_clinical"] = [len(a) > 0 for a in clinical_alerts]

        # --- Metodo 2: IsolationForest ---
        if not self.is_fitted:
            self.fit(df)

        data = result[self.feature_columns].copy()
        data_scaled = self.scaler.transform(data)
        ml_scores = self.model.decision_function(data_scaled)
        ml_predictions = self.model.predict(data_scaled)

        result["anomaly_score_ml"] = ml_scores
        result["anomaly_ml"] = ml_predictions == -1  # -1 = anomalia

        # --- Combinar ---
        result["anomaly_combined"] = (
            result["anomaly_clinical"] | result["anomaly_ml"]
        )

        # Determinar nivel de alerta
        result["alert_level"] = "normal"
        result["alert_message"] = ""

        for idx in result.index:
            alerts = result.at[idx, "alerts_clinical"]
            is_ml = result.at[idx, "anomaly_ml"]

            if is_ml and len(alerts) > 0:
                level = "critical"
            elif len(alerts) > 0 and any(a["severity"] == "critical" for a in alerts):
                level = "critical"
            elif is_ml or len(alerts) > 0:
                level = "warning"
            else:
                level = "normal"

            result.at[idx, "alert_level"] = level

            # Mensagem
            if level != "normal":
                msgs = [f"{a['vital']}={a['value']}" for a in alerts]
                result.at[idx, "alert_message"] = "; ".join(msgs)

        return result

    def generate_alert_summary(self, result_df: pd.DataFrame) -> Dict:
        """Gera sumario de alertas a partir dos resultados."""
        total = len(result_df)
        anomalies = result_df["anomaly_combined"].sum()
        criticals = (result_df["alert_level"] == "critical").sum()
        warnings = (result_df["alert_level"] == "warning").sum()

        # Alertas por tipo de sinal vital
        alert_counts = {}
        for alerts in result_df["alerts_clinical"]:
            for alert in alerts:
                vital = alert["vital"]
                sev = alert["severity"]
                key = f"{vital}_{sev}"
                alert_counts[key] = alert_counts.get(key, 0) + 1

        # Evolucao temporal dos sinais
        vital_cols = [c for c in result_df.columns
                      if c in self.CLINICAL_THRESHOLDS]
        trends = {}
        for col in vital_cols:
            if len(result_df) > 1:
                slope = np.polyfit(range(len(result_df)), result_df[col], 1)[0]
                trends[col] = {
                    "slope": round(slope, 3),
                    "trend": "rising" if slope > 0.01 else "falling" if slope < -0.01 else "stable",
                }

        return {
            "total_records": total,
            "anomalies_detected": int(anomalies),
            "anomaly_rate": round(anomalies / max(total, 1), 4),
            "critical_alerts": int(criticals),
            "warning_alerts": int(warnings),
            "alerts_by_vital": alert_counts,
            "trends": trends,
            "status": "critical" if criticals > 0
                      else "warning" if warnings > total * 0.05
                      else "normal",
        }


def detect_vital_anomalies(input_path: str) -> Dict:
    """Funcao de entrada para o CLI."""
    df = pd.read_csv(input_path)
    detector = VitalSignsAnomalyDetector(contamination=0.05)
    results = detector.detect(df)
    summary = detector.generate_alert_summary(results)
    return {"results": results, "summary": summary}
