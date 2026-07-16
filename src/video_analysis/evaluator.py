"""
Avaliacao de videos clinicos baseada em deteccao por frames.
"""

from pathlib import Path
from typing import Dict, Iterable, List

from src.utils import setup_logging
from src.video_analysis.detector import ObjectDetector
from src.video_analysis.report_generator import VideoReportGenerator


EXPECTED_OBJECTS = ["person", "bed", "chair", "bench"]
UNEXPECTED_OBJECTS = ["cell phone", "bottle", "scissors", "knife"]


class VideoEvaluator:
    """Avalia um ou mais videos e produz relatorios consolidados."""

    def __init__(self, sample_rate: int = 15, conf: float = 0.25):
        self.sample_rate = sample_rate
        self.conf = conf
        self.logger = setup_logging("video_evaluator")
        self.detector = ObjectDetector()
        self.reporter = VideoReportGenerator()

    def evaluate(self, video_path: str) -> Dict:
        """Avalia um video individual."""
        path = Path(video_path)
        self.logger.info("[VIDEO] %s", path)

        detections = self.detector.detect_video(
            str(path),
            sample_rate=self.sample_rate,
            conf=self.conf,
        )
        anomalies = self.detector.detect_anomalous_objects(
            str(path),
            expected_classes=EXPECTED_OBJECTS,
            unexpected_classes=UNEXPECTED_OBJECTS,
            conf=self.conf,
            sample_rate=self.sample_rate,
            min_confidence=max(self.conf, 0.45),
        )

        statistics = detections["statistics"]
        frame_detections = detections["frame_detections"]
        unique_objects = sorted(statistics["objects_detected"].keys())
        unexpected_objects = sorted({item["object"] for item in anomalies["anomalies"]})
        procedure_summary = self._build_procedure_summary(frame_detections, anomalies)

        result = {
            "video_file": str(path),
            "video_name": path.name,
            "frames_total": statistics["total_frames"],
            "frames_analyzed": statistics["frames_analyzed"],
            "fps": statistics["fps"],
            "duration_seconds": round(statistics["duration_seconds"], 2),
            "total_detections": statistics["total_detections"],
            "objects_detected": statistics["detections_per_class"],
            "unique_objects": unique_objects,
            "unexpected_objects": unexpected_objects,
            "frames_with_unexpected_objects": anomalies["frames_with_anomalies"],
            "unexpected_object_events": anomalies["total_anomalies"],
            "anomaly_rate": round(anomalies["anomaly_rate"], 4),
            "procedure_summary": procedure_summary,
            "procedure_findings": procedure_summary["findings"],
            "status": self._status_from(procedure_summary["findings"]),
        }

        report_path = self.reporter.generate_video_report(result)
        result["report_path"] = report_path
        return result

    def evaluate_many(self, video_paths: Iterable[str]) -> List[Dict]:
        """Avalia varios videos em sequencia."""
        return [self.evaluate(video_path) for video_path in video_paths]

    @staticmethod
    def _status_from(findings: List[Dict]) -> str:
        if any(item["severity"] == "critical" for item in findings):
            return "critical"
        if any(item["severity"] == "warning" for item in findings):
            return "warning"
        return "normal"

    @staticmethod
    def _build_procedure_summary(frame_detections: Dict[int, List[Dict]], anomalies: Dict) -> Dict:
        frames_analyzed = max(len(frame_detections), 1)
        frames_with_patient = 0
        frames_with_multiple_people = 0

        for detections in frame_detections.values():
            people = [det for det in detections if det["class_name"] == "person"]
            if people:
                frames_with_patient += 1
            if len(people) > 1:
                frames_with_multiple_people += 1

        patient_presence_rate = frames_with_patient / frames_analyzed
        multi_person_rate = frames_with_multiple_people / frames_analyzed
        unexpected_rate = anomalies["frames_with_anomalies"] / frames_analyzed

        findings = []

        if patient_presence_rate < 0.7:
            findings.append({
                "type": "low_patient_visibility",
                "severity": "critical" if patient_presence_rate < 0.5 else "warning",
                "evidence": f"Paciente visivel em {patient_presence_rate:.0%} dos frames analisados.",
                "impact": "A sessao perde confiabilidade para avaliacao automatica do procedimento.",
                "recommendation": "Reposicionar a camera para manter o paciente enquadrado durante toda a atividade.",
            })

        if multi_person_rate >= 0.2:
            findings.append({
                "type": "multiple_people_interference",
                "severity": "warning",
                "evidence": f"Mais de uma pessoa presente em {multi_person_rate:.0%} dos frames analisados.",
                "impact": "Ha interferencia visual no acompanhamento do procedimento.",
                "recommendation": "Gravar a sessao com apenas paciente e terapeuta estritamente necessarios no enquadramento.",
            })

        if anomalies["frames_with_anomalies"] > 0:
            labels = sorted({item["object"] for item in anomalies["anomalies"]})
            unexpected_rate_text = VideoEvaluator._format_rate(
                anomalies["frames_with_anomalies"],
                frames_analyzed,
            )
            findings.append({
                "type": "unexpected_objects",
                "severity": "critical" if unexpected_rate >= 0.2 else "warning",
                "evidence": (
                    f"Objetos inadequados detectados em "
                    f"{anomalies['frames_with_anomalies']} de {frames_analyzed} frames "
                    f"({unexpected_rate_text}; {', '.join(labels)})."
                ),
                "impact": "Objetos estranhos ao contexto clinico podem indicar falha de preparo do ambiente.",
                "recommendation": "Remover objetos nao essenciais do campo de visao antes de iniciar a gravacao.",
            })

        if not findings:
            findings.append({
                "type": "procedure_ok",
                "severity": "normal",
                "evidence": "Nao foram observados desvios relevantes no enquadramento ou no ambiente.",
                "impact": "O video atende ao basico para avaliacao automatica do procedimento.",
                "recommendation": "Manter o mesmo padrao de gravacao nas proximas sessoes.",
            })
        return {
            "frames_analyzed": frames_analyzed,
            "frames_with_patient": frames_with_patient,
            "patient_presence_rate": round(patient_presence_rate, 4),
            "frames_with_multiple_people": frames_with_multiple_people,
            "multi_person_rate": round(multi_person_rate, 4),
            "frames_with_unexpected_objects": anomalies["frames_with_anomalies"],
            "unexpected_rate": round(unexpected_rate, 4),
            "findings": findings,
        }

    @staticmethod
    def _format_rate(part: int, total: int) -> str:
        """Formata percentual sem mascarar ocorrencias pequenas."""
        if total <= 0 or part <= 0:
            return "0%"
        rate = part / total
        if rate < 0.01:
            return "<1%"
        return f"{rate:.0%}"
