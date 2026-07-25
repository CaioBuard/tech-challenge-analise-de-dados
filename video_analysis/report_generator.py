"""Geracao de relatorios da avaliacao de video."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class VideoReportGenerator:
    """Gera os relatorios JSON e TXT dentro da pasta da solicitacao web."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_video_report(self, evaluation: Dict) -> Dict[str, str]:
        video_stem = Path(evaluation["video_name"]).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"video_evaluation_{video_stem}_{timestamp}"
        json_path = self.output_dir / f"{base_name}.json"
        txt_path = self.output_dir / f"{base_name}.txt"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump({"type": "video_evaluation", "timestamp": datetime.now().isoformat(), "summary": evaluation}, handle, indent=2, ensure_ascii=False)
        with txt_path.open("w", encoding="utf-8") as handle:
            handle.write(f"RELATORIO DE AVALIACAO DE VIDEO\nVideo: {evaluation['video_name']}\nStatus: {evaluation['status'].upper()}\n\n")
            handle.write(f"Frames analisados: {evaluation['frames_analyzed']}\n")
            handle.write(f"Paciente visivel em: {evaluation['procedure_summary']['patient_presence_rate']:.0%} dos frames\n")
            handle.write(f"Taxa de anomalia: {evaluation['anomaly_rate']:.2%}\n\n")
            for finding in evaluation["procedure_findings"]:
                handle.write(f"[{finding['severity'].upper()}] {finding['type']}\n")
                handle.write(f"Evidencia: {finding['evidence']}\nRecomendacao: {finding['recommendation']}\n\n")
        return {"json": str(json_path), "txt": str(txt_path)}
