"""
Detector de objetos e areas criticas com YOLOv8.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - depende do ambiente
    YOLO = None


class ObjectDetector:
    """Detector de objetos baseado em YOLOv8."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Caminho para modelo customizado. Se None, usa YOLOv8n.
        """
        if YOLO is None:
            raise ImportError(
                "ultralytics nao instalado. Instale com: pip install ultralytics"
            )
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            print("[DETECTOR] Usando YOLOv8n pretrained...")
            self.model = YOLO("yolov8n.pt")
        self.class_names = self.model.names

    def detect_frame(self, frame: np.ndarray, conf: float = 0.25) -> List[Dict]:
        """
        Detecta objetos em um frame.

        Returns:
            Lista de deteccoes: [{class_id, class_name, confidence, bbox, center}]
        """
        results = self.model(frame, conf=conf, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])

                detections.append({
                    "class_id": cls_id,
                    "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                    "confidence": conf_val,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "area": (x2 - x1) * (y2 - y1),
                })

        return detections

    def detect_video(self, video_path: str, sample_rate: int = 5,
                     conf: float = 0.25) -> Dict[str, List]:
        """
        Analisa video inteiro, amostrando frames.

        Args:
            video_path: Caminho do video
            sample_rate: Analisar 1 frame a cada N
            conf: Threshold de confianca

        Returns:
            Dict com deteccoes por frame e estatisticas
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Nao foi possivel abrir video: {video_path}")

        frame_detections = {}
        all_objects = {}
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                detections = self.detect_frame(frame, conf=conf)
                frame_detections[frame_idx] = detections

                for det in detections:
                    name = det["class_name"]
                    if name not in all_objects:
                        all_objects[name] = 0
                    all_objects[name] += 1

            frame_idx += 1

        cap.release()

        # Estatisticas
        total_detections = sum(len(v) for v in frame_detections.values())
        stats = {
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": total_frames / fps if fps > 0 else 0,
            "frames_analyzed": len(frame_detections),
            "total_detections": total_detections,
            "objects_detected": all_objects,
            "detections_per_class": dict(sorted(
                all_objects.items(), key=lambda x: x[1], reverse=True
            )),
        }

        return {
            "frame_detections": frame_detections,
            "statistics": stats,
        }

    def detect_anomalous_objects(self, video_path: str,
                                 expected_classes: List[str],
                                 unexpected_classes: List[str],
                                 conf: float = 0.25,
                                 sample_rate: int = 5,
                                 min_confidence: float = 0.45) -> Dict:
        """
        Detecta objetos anomalos (inesperados) em video clinico.

        Args:
            video_path: Caminho do video
            expected_classes: Classes esperadas no video (ex: instrumentos cirurgicos)
            unexpected_classes: Classes que indicam anomalia (ex: pessoa, celular)
            conf: Threshold de confianca

        Returns:
            Relatorio de anomalias por frame
        """
        results = self.detect_video(video_path, sample_rate=sample_rate, conf=conf)
        anomalies = []
        frames_with_anomalies = set()

        for frame_idx, detections in results["frame_detections"].items():
            frame_anomalies = []
            for det in detections:
                if (
                    det["class_name"] in unexpected_classes
                    and det["confidence"] >= min_confidence
                ):
                    frame_anomalies.append({
                        "frame": frame_idx,
                        "object": det["class_name"],
                        "confidence": det["confidence"],
                        "position": det["center"],
                        "severity": "high" if det["confidence"] > 0.7 else "medium",
                    })
            if frame_anomalies:
                frames_with_anomalies.add(frame_idx)
                anomalies.extend(frame_anomalies)

        return {
            "video_file": video_path,
            "anomalies": anomalies,
            "total_anomalies": len(anomalies),
            "frames_analyzed": results["statistics"]["frames_analyzed"],
            "frames_with_anomalies": len(frames_with_anomalies),
            "anomaly_rate": (
                len(frames_with_anomalies)
                / max(results["statistics"]["frames_analyzed"], 1)
            ),
        }
