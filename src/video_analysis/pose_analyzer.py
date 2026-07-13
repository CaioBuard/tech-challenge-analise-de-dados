"""
Analisador postural usando YOLOv8-pose.

Extrai keypoints do corpo e detecta desvios posturais em:
- Sessoes de fisioterapia
- Monitoramento de pacientes acamados
- Analise de marcha
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import math


# Indices dos keypoints COCO (usados pelo YOLOv8-pose)
KP = {
    "nariz": 0, "olho_esq": 1, "olho_dir": 2,
    "ouvido_esq": 3, "ouvido_dir": 4,
    "ombro_esq": 5, "ombro_dir": 6,
    "cotovelo_esq": 7, "cotovelo_dir": 8,
    "punho_esq": 9, "punho_dir": 10,
    "quadril_esq": 11, "quadril_dir": 12,
    "joelho_esq": 13, "joelho_dir": 14,
    "tornozelo_esq": 15, "tornozelo_dir": 16,
}


class PoseAnalyzer:
    """Analisador de postura corporal."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            print("[POSE] Usando YOLOv8n-pose pretrained...")
            self.model = YOLO("yolov8n-pose.pt")

    def extract_keypoints(self, frame: np.ndarray,
                          conf: float = 0.5) -> List[Dict]:
        """Extrai keypoints de todas as pessoas no frame."""
        results = self.model(frame, conf=conf, verbose=False)
        people = []

        for r in results:
            if r.keypoints is None:
                continue

            for i, kpts in enumerate(r.keypoints.data):
                keypoints = {}
                for name, idx in KP.items():
                    x, y, c = kpts[idx].tolist()
                    keypoints[name] = {"x": x, "y": y, "confidence": c}

                bbox = None
                if r.boxes and i < len(r.boxes):
                    box = r.boxes[i]
                    bbox = box.xyxy[0].tolist()

                people.append({
                    "id": i,
                    "keypoints": keypoints,
                    "bbox": bbox,
                })

        return people

    def compute_angle(self, a: Tuple[float, float],
                      b: Tuple[float, float],
                      c: Tuple[float, float]) -> float:
        """Calcula angulo ABC (em graus)."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def analyze_posture(self, keypoints: Dict[str, Dict]) -> Dict:
        """
        Analisa postura de uma pessoa e retorna metricas + desvios.

        Returns:
            Dict com angulos, desvios, e alertas
        """
        metrics = {}
        alerts = []

        # Verificar confianca minima dos keypoints principais
        critical_kps = ["ombro_esq", "ombro_dir", "quadril_esq", "quadril_dir"]
        low_conf = [k for k in critical_kps
                    if keypoints.get(k, {}).get("confidence", 0) < 0.3]
        if len(low_conf) > 2:
            return {"error": "Keypoints insuficientes", "alerts": []}

        def kp_xy(name):
            k = keypoints.get(name, {})
            return (k.get("x", 0), k.get("y", 0))

        # Angulo do tronco (ombro-quadril)
        shoulder_mid = np.mean([kp_xy("ombro_esq"), kp_xy("ombro_dir")], axis=0)
        hip_mid = np.mean([kp_xy("quadril_esq"), kp_xy("quadril_dir")], axis=0)
        trunk_angle = abs(math.degrees(math.atan2(
            shoulder_mid[0] - hip_mid[0],
            shoulder_mid[1] - hip_mid[1]
        )))
        metrics["trunk_angle"] = round(trunk_angle, 1)

        if trunk_angle > 15:
            alerts.append({
                "type": "trunk_tilt",
                "severity": "medium" if trunk_angle < 30 else "high",
                "value": round(trunk_angle, 1),
                "message": f"Inclinacao do tronco: {trunk_angle:.0f} graus",
            })

        # Angulo do pescoco (ombro medio - nariz)
        nose = kp_xy("nariz")
        neck_angle = abs(math.degrees(math.atan2(
            nose[0] - shoulder_mid[0],
            nose[1] - shoulder_mid[1]
        )))
        metrics["neck_angle"] = round(neck_angle, 1)

        if neck_angle > 20:
            alerts.append({
                "type": "neck_tilt",
                "severity": "medium",
                "value": round(neck_angle, 1),
                "message": f"Inclinacao do pescoco: {neck_angle:.0f} graus",
            })

        # Angulo do joelho esquerdo (quadril-joelho-tornozelo)
        left_knee = self.compute_angle(
            kp_xy("quadril_esq"), kp_xy("joelho_esq"), kp_xy("tornozelo_esq")
        )
        metrics["left_knee_angle"] = round(left_knee, 1)

        # Angulo do joelho direito
        right_knee = self.compute_angle(
            kp_xy("quadril_dir"), kp_xy("joelho_dir"), kp_xy("tornozelo_dir")
        )
        metrics["right_knee_angle"] = round(right_knee, 1)

        # Simetria dos ombros
        shoulder_diff = abs(
            keypoints.get("ombro_esq", {}).get("y", 0) -
            keypoints.get("ombro_dir", {}).get("y", 0)
        )
        metrics["shoulder_symmetry"] = round(shoulder_diff, 1)

        if shoulder_diff > 20:
            alerts.append({
                "type": "shoulder_asymmetry",
                "severity": "medium",
                "value": round(shoulder_diff, 1),
                "message": f"Assimetria de ombros: {shoulder_diff:.0f}px",
            })

        return {
            "metrics": metrics,
            "alerts": alerts,
            "has_alerts": len(alerts) > 0,
        }

    def analyze_video_posture(self, video_path: str,
                              sample_rate: int = 10) -> Dict:
        """
        Analisa postura ao longo de um video.

        Returns:
            Sumario com metricas medias, alertas e evolucao temporal
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Nao foi possivel abrir video: {video_path}")

        frame_results = []
        all_metrics = []
        all_alerts = []
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                people = self.extract_keypoints(frame)

                for person in people:
                    analysis = self.analyze_posture(person["keypoints"])
                    if "error" not in analysis:
                        analysis["frame"] = frame_idx
                        analysis["person_id"] = person["id"]
                        frame_results.append(analysis)
                        all_metrics.append(analysis["metrics"])
                        all_alerts.extend(analysis.get("alerts", []))

            frame_idx += 1

        cap.release()

        # Agregar metricas
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    avg_metrics[f"avg_{key}"] = round(np.mean(values), 1)
                    avg_metrics[f"std_{key}"] = round(np.std(values), 1)

        # Agrupar alertas por tipo
        alert_summary = {}
        for alert in all_alerts:
            atype = alert["type"]
            if atype not in alert_summary:
                alert_summary[atype] = {"count": 0, "max_severity": "low"}
            alert_summary[atype]["count"] += 1
            if alert["severity"] == "high":
                alert_summary[atype]["max_severity"] = "high"
            elif alert["severity"] == "medium" and alert_summary[atype]["max_severity"] != "high":
                alert_summary[atype]["max_severity"] = "medium"

        return {
            "video_file": video_path,
            "total_frames": total_frames,
            "frames_analyzed": len(frame_results),
            "people_detected": len(set(r["person_id"] for r in frame_results)),
            "average_metrics": avg_metrics,
            "alert_summary": alert_summary,
            "total_alerts": len(all_alerts),
            "detailed_alerts": all_alerts[:20],  # limitar para nao ficar gigante
        }

    def compare_to_reference(self, patient_keypoints: Dict,
                             reference_keypoints: Dict,
                             tolerance: float = 0.15) -> Dict:
        """
        Compara postura do paciente com referencia (ex: exercicio correto).

        Returns:
            Score de similaridade e desvios por articulacao
        """
        deviations = {}
        total_deviation = 0
        count = 0

        for joint_name in KP.keys():
            pat = patient_keypoints.get(joint_name, {})
            ref = reference_keypoints.get(joint_name, {})

            if pat.get("confidence", 0) < 0.3 or ref.get("confidence", 0) < 0.3:
                continue

            dx = pat["x"] - ref["x"]
            dy = pat["y"] - ref["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            deviations[joint_name] = round(dist, 1)
            total_deviation += dist
            count += 1

        avg_deviation = total_deviation / max(count, 1)
        similarity = max(0, 1.0 - avg_deviation / 100)  # normalizado

        return {
            "similarity_score": round(similarity, 3),
            "average_deviation_px": round(avg_deviation, 1),
            "joint_deviations": deviations,
            "within_tolerance": avg_deviation < (100 * tolerance),
        }
