"""Detector de objetos com YOLOv8."""

from pathlib import Path
from typing import Dict, List, Optional
import importlib

import numpy as np

class ObjectDetector:
    """Detector de objetos baseado em YOLOv8."""

    def __init__(self, model_path: Optional[str] = None):
        try:
            self.cv2 = importlib.import_module("cv2")
        except ImportError as error:  # pragma: no cover - depende do sistema operacional
            raise ImportError(
                "OpenCV nao carregou. Reinstale opencv-python no ambiente da aplicacao."
            ) from error
        self.model_path = model_path
        self.model = None
        self.class_names = {}

    def _load_model(self):
        """Carrega YOLO apenas depois de confirmar que o video pode ser aberto."""
        if self.model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - depende do ambiente
            raise ImportError(
                "ultralytics nao instalado. Instale as dependencias de web/requirements.txt"
            ) from error
        if self.model_path and Path(self.model_path).exists():
            self.model = YOLO(self.model_path)
        else:
            self.model = YOLO("yolov8n.pt")
        self.class_names = self.model.names

    def detect_frame(self, frame: np.ndarray, conf: float = 0.25) -> List[Dict]:
        """Retorna os objetos detectados em um frame."""
        self._load_model()
        detections = []
        for result in self.model(frame, conf=conf, verbose=False):
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                detections.append({
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                    "confidence": float(box.conf[0]),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                })
        return detections

    def detect_video(self, video_path: str, sample_rate: int = 15, conf: float = 0.25) -> Dict:
        """Analisa um a cada ``sample_rate`` frames do video."""
        capture = self.cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Nao foi possivel abrir video: {video_path}")

        frame_detections, objects_detected = {}, {}
        frame_index = 0
        total_frames = int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(self.cv2.CAP_PROP_FPS)
        while True:
            read, frame = capture.read()
            if not read:
                break
            if frame_index % sample_rate == 0:
                detections = self.detect_frame(frame, conf)
                frame_detections[frame_index] = detections
                for detection in detections:
                    name = detection["class_name"]
                    objects_detected[name] = objects_detected.get(name, 0) + 1
            frame_index += 1
        capture.release()

        return {
            "frame_detections": frame_detections,
            "statistics": {
                "total_frames": total_frames,
                "fps": fps,
                "duration_seconds": total_frames / fps if fps > 0 else 0,
                "frames_analyzed": len(frame_detections),
                "total_detections": sum(len(value) for value in frame_detections.values()),
                "objects_detected": objects_detected,
                "detections_per_class": dict(sorted(objects_detected.items(), key=lambda item: item[1], reverse=True)),
            },
        }
