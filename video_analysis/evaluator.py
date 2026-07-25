"""Regras de validacao de videos clinicos."""

from pathlib import Path
from typing import Dict, List, Optional

from utils import setup_logging
from .detector import ObjectDetector
from .report_generator import VideoReportGenerator


UNEXPECTED_OBJECTS = {"cell phone", "bottle", "scissors", "knife"}


class VideoEvaluator:
    """Avalia enquadramento, pessoas e objetos inadequados em um video."""

    def __init__(self, sample_rate: int = 15, conf: float = 0.25, report_dir: Optional[str] = None):
        if sample_rate <= 0:
            raise ValueError("sample_rate deve ser maior que zero")
        self.sample_rate = sample_rate
        self.conf = conf
        self.logger = setup_logging("video_evaluator")
        self.detector = ObjectDetector()
        self.reporter = VideoReportGenerator(report_dir) if report_dir else None

    def evaluate(self, video_path: str) -> Dict:
        path = Path(video_path)
        self.logger.info("[VIDEO] %s", path.name)
        detections = self.detector.detect_video(str(path), self.sample_rate, self.conf)
        statistics = detections["statistics"]
        frame_detections = detections["frame_detections"]
        anomalies = self._find_anomalies(frame_detections)
        summary = self._build_procedure_summary(frame_detections, anomalies)
        result = {
            "video_name": path.name,
            "frames_total": statistics["total_frames"],
            "frames_analyzed": statistics["frames_analyzed"],
            "fps": statistics["fps"],
            "duration_seconds": round(statistics["duration_seconds"], 2),
            "total_detections": statistics["total_detections"],
            "objects_detected": statistics["detections_per_class"],
            "unique_objects": sorted(statistics["objects_detected"]),
            "unexpected_objects": sorted({item["object"] for item in anomalies["items"]}),
            "frames_with_unexpected_objects": anomalies["frames_with_anomalies"],
            "unexpected_object_events": len(anomalies["items"]),
            "anomaly_rate": round(anomalies["rate"], 4),
            "procedure_summary": summary,
            "procedure_findings": summary["findings"],
            "status": self._status_from(summary["findings"]),
        }
        if self.reporter:
            result["report_paths"] = self.reporter.generate_video_report(result)
        return result

    def _find_anomalies(self, frame_detections: Dict[int, List[Dict]]) -> Dict:
        items, frames = [], set()
        for frame, detections in frame_detections.items():
            for detection in detections:
                if detection["class_name"] in UNEXPECTED_OBJECTS and detection["confidence"] >= max(self.conf, 0.45):
                    frames.add(frame)
                    items.append({"frame": frame, "object": detection["class_name"]})
        analyzed = max(len(frame_detections), 1)
        return {"items": items, "frames_with_anomalies": len(frames), "rate": len(frames) / analyzed}

    @staticmethod
    def _status_from(findings: List[Dict]) -> str:
        if any(item["severity"] == "critical" for item in findings):
            return "critical"
        if any(item["severity"] == "warning" for item in findings):
            return "warning"
        return "normal"

    @staticmethod
    def _build_procedure_summary(frame_detections: Dict[int, List[Dict]], anomalies: Dict) -> Dict:
        analyzed = max(len(frame_detections), 1)
        patient_frames = sum(any(item["class_name"] == "person" for item in detections) for detections in frame_detections.values())
        crowded_frames = sum(sum(item["class_name"] == "person" for item in detections) >= 3 for detections in frame_detections.values())
        patient_rate, crowd_rate = patient_frames / analyzed, crowded_frames / analyzed
        findings = []
        if patient_rate < 0.7:
            findings.append({"type": "low_patient_visibility", "severity": "critical" if patient_rate < 0.5 else "warning", "evidence": f"Paciente visivel em {patient_rate:.0%} dos frames analisados.", "impact": "A sessao perde confiabilidade para avaliacao automatica do procedimento.", "recommendation": "Reposicionar a camera para manter o paciente enquadrado durante toda a atividade."})
        if crowd_rate >= 0.2:
            findings.append({"type": "crowded_scene_interference", "severity": "warning", "evidence": f"Tres ou mais pessoas presentes em {crowd_rate:.0%} dos frames analisados.", "impact": "Ha excesso de pessoas no enquadramento.", "recommendation": "Priorizar gravacao com paciente e terapeuta apenas."})
        if anomalies["frames_with_anomalies"] >= 3 or anomalies["rate"] >= 0.01:
            labels = ", ".join(sorted({item["object"] for item in anomalies["items"]}))
            findings.append({"type": "unexpected_objects", "severity": "critical" if anomalies["rate"] >= 0.2 else "warning", "evidence": f"Objetos inadequados detectados em {anomalies['frames_with_anomalies']} de {analyzed} frames ({labels}).", "impact": "Objetos estranhos ao contexto clinico podem indicar falha de preparo do ambiente.", "recommendation": "Remover objetos nao essenciais do campo de visao antes da gravacao."})
        if not findings:
            findings.append({"type": "procedure_ok", "severity": "normal", "evidence": "Nao foram observados desvios relevantes no enquadramento ou no ambiente.", "impact": "O video atende ao basico para avaliacao automatica do procedimento.", "recommendation": "Manter o mesmo padrao de gravacao nas proximas sessoes."})
        return {"frames_analyzed": analyzed, "frames_with_patient": patient_frames, "patient_presence_rate": round(patient_rate, 4), "frames_with_three_or_more_people": crowded_frames, "crowd_rate": round(crowd_rate, 4), "frames_with_unexpected_objects": anomalies["frames_with_anomalies"], "unexpected_rate": round(anomalies["rate"], 4), "findings": findings}
