"""
Geracao de relatorios de avaliacao de video.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.utils import generate_report_filename, get_output_dir


class VideoReportGenerator:
    """Gera relatorios em JSON e TXT para cada video avaliado."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else get_output_dir("reports")

    def generate_video_report(self, evaluation: Dict) -> str:
        """Gera relatorio completo de um video."""
        video_stem = Path(evaluation["video_name"]).stem
        filename = generate_report_filename(f"video_evaluation_{video_stem}")
        json_path = self.output_dir / f"{filename}.json"
        txt_path = self.output_dir / f"{filename}.txt"

        report = {
            "type": "video_evaluation",
            "timestamp": datetime.now().isoformat(),
            "summary": evaluation,
        }

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write("=" * 60 + "\n")
            handle.write("RELATORIO DE AVALIACAO DE VIDEO\n")
            handle.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            handle.write(f"Video: {evaluation['video_name']}\n")
            handle.write(f"Status: {evaluation['status'].upper()}\n")
            handle.write("=" * 60 + "\n\n")
            handle.write(f"Frames totais: {evaluation['frames_total']}\n")
            handle.write(f"Frames analisados: {evaluation['frames_analyzed']}\n")
            handle.write(f"Duracao (s): {evaluation['duration_seconds']}\n")
            handle.write(f"Deteccoes totais: {evaluation['total_detections']}\n")
            handle.write(
                f"Frames com objetos inesperados: "
                f"{evaluation['frames_with_unexpected_objects']}\n"
            )
            handle.write(
                f"Paciente visivel em: "
                f"{evaluation['procedure_summary']['patient_presence_rate']:.0%} dos frames\n"
            )
            handle.write(
                f"Frames com mais de uma pessoa: "
                f"{evaluation['procedure_summary']['frames_with_multiple_people']}\n"
            )
            handle.write(f"Eventos inesperados: {evaluation['unexpected_object_events']}\n")
            handle.write(f"Taxa de anomalia: {evaluation['anomaly_rate']:.2%}\n\n")

            handle.write("OBJETOS DETECTADOS\n")
            if evaluation["objects_detected"]:
                for label, count in evaluation["objects_detected"].items():
                    handle.write(f"  {label}: {count}\n")
            else:
                handle.write("  Nenhum objeto detectado.\n")

            handle.write("\nOBJETOS INESPERADOS\n")
            if evaluation["unexpected_objects"]:
                for label in evaluation["unexpected_objects"]:
                    handle.write(f"  {label}\n")
            else:
                handle.write("  Nenhum objeto inesperado.\n")

            handle.write("\nDESVIOS OU FALHAS NO PROCEDIMENTO\n")
            for finding in evaluation["procedure_findings"]:
                handle.write(f"  [{finding['severity'].upper()}] {finding['type']}\n")
                handle.write(f"    Evidencia: {finding['evidence']}\n")
                handle.write(f"    Impacto: {finding['impact']}\n")
                handle.write(f"    Recomendacao: {finding['recommendation']}\n")

        return str(json_path)
