"""
Gerador de relatorios automaticos para analise de video.

Gera relatorios indicando desvios ou falhas em procedimentos clinicos.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from src.utils import get_output_dir, generate_report_filename


class VideoReportGenerator:
    """Gera relatorios de analise de video clinico."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else get_output_dir("video")

    def generate_posture_report(self, posture_results: Dict,
                                video_name: str = "") -> str:
        """Gera relatorio de analise postural."""
        filename = generate_report_filename(f"posture_{video_name}")
        report_path = self.output_dir / f"{filename}.json"
        txt_path = self.output_dir / f"{filename}.txt"

        report = {
            "type": "posture_analysis",
            "timestamp": datetime.now().isoformat(),
            "video": posture_results.get("video_file", video_name),
            "summary": {
                "frames_analyzed": posture_results.get("frames_analyzed", 0),
                "people_detected": posture_results.get("people_detected", 0),
                "total_alerts": posture_results.get("total_alerts", 0),
            },
            "average_metrics": posture_results.get("average_metrics", {}),
            "alert_summary": posture_results.get("alert_summary", {}),
            "recommendations": [],
        }

        # Gerar recomendacoes baseadas nos alertas
        alert_summary = posture_results.get("alert_summary", {})
        recs = report["recommendations"]

        if "trunk_tilt" in alert_summary:
            count = alert_summary["trunk_tilt"]["count"]
            sev = alert_summary["trunk_tilt"]["max_severity"]
            recs.append({
                "issue": "Inclinacao anormal do tronco",
                "occurrences": count,
                "severity": sev,
                "recommendation": "Avaliar ergonomia da cadeira e encaminhar para "
                                  "fisioterapia postural.",
            })

        if "neck_tilt" in alert_summary:
            count = alert_summary["neck_tilt"]["count"]
            recs.append({
                "issue": "Inclinacao anormal do pescoco",
                "occurrences": count,
                "severity": alert_summary["neck_tilt"]["max_severity"],
                "recommendation": "Verificar altura do monitor e postura cervical.",
            })

        if "shoulder_asymmetry" in alert_summary:
            count = alert_summary["shoulder_asymmetry"]["count"]
            sev = alert_summary["shoulder_asymmetry"]["max_severity"]
            recs.append({
                "issue": "Assimetria de ombros detectada",
                "occurrences": count,
                "severity": sev,
                "recommendation": "Possivel escoliose ou desvio postural. "
                                  "Recomendada avaliacao ortopedica.",
            })

        # Salvar JSON
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Salvar TXT legivel
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("RELATORIO DE ANALISE POSTURAL\n")
            f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"Video: {video_name}\n")
            f.write("=" * 60 + "\n\n")

            f.write("RESUMO\n")
            f.write(f"  Frames analisados: {report['summary']['frames_analyzed']}\n")
            f.write(f"  Pessoas detectadas: {report['summary']['people_detected']}\n")
            f.write(f"  Alertas gerados: {report['summary']['total_alerts']}\n\n")

            if report["average_metrics"]:
                f.write("METRICAS MEDIAS\n")
                for k, v in report["average_metrics"].items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

            if recs:
                f.write("RECOMENDACOES\n")
                for i, rec in enumerate(recs, 1):
                    f.write(f"  {i}. [{rec['severity'].upper()}] {rec['issue']}\n")
                    f.write(f"     Ocorrencias: {rec['occurrences']}\n")
                    f.write(f"     Acao: {rec['recommendation']}\n\n")

        return str(report_path)

    def generate_anomaly_report(self, anomaly_results: Dict,
                                video_name: str = "") -> str:
        """Gera relatorio de anomalias detectadas em video."""
        filename = generate_report_filename(f"anomaly_{video_name}")
        report_path = self.output_dir / f"{filename}.json"

        report = {
            "type": "anomaly_detection",
            "timestamp": datetime.now().isoformat(),
            "video": anomaly_results.get("video_file", video_name),
            "total_anomalies": anomaly_results.get("total_anomalies", 0),
            "frames_analyzed": anomaly_results.get("frames_analyzed", 0),
            "anomaly_rate": anomaly_results.get("anomaly_rate", 0),
            "anomalies": anomaly_results.get("anomalies", []),
            "status": "critical" if anomaly_results.get("anomaly_rate", 0) > 0.1
                      else "warning" if anomaly_results.get("total_anomalies", 0) > 0
                      else "normal",
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # TXT
        txt_path = self.output_dir / f"{filename}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("RELATORIO DE ANOMALIAS EM VIDEO\n")
            f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"Video: {video_name}\n")
            f.write(f"Status: {report['status'].upper()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Anomalias detectadas: {report['total_anomalies']}\n")
            f.write(f"Taxa de anomalia: {report['anomaly_rate']:.2%}\n\n")

            if report["anomalies"]:
                f.write("ANOMALIAS ENCONTRADAS\n")
                for a in report["anomalies"][:10]:
                    f.write(f"  Frame {a.get('frame', '?')}: "
                            f"{a.get('object', '?')} "
                            f"[{a.get('severity', '?')}]\n")

        return str(report_path)

    def generate_combined_report(self, posture_results: Dict,
                                 anomaly_results: Dict,
                                 video_name: str = "") -> str:
        """Gera relatorio combinado (postura + anomalias)."""
        filename = generate_report_filename(f"combined_{video_name}")
        report_path = self.output_dir / f"{filename}.json"

        report = {
            "type": "combined_video_analysis",
            "timestamp": datetime.now().isoformat(),
            "video": video_name,
            "posture": {
                "alerts": posture_results.get("total_alerts", 0),
                "summary": posture_results.get("alert_summary", {}),
            },
            "anomalies": {
                "total": anomaly_results.get("total_anomalies", 0),
                "rate": anomaly_results.get("anomaly_rate", 0),
            },
            "overall_status": self._determine_status(posture_results, anomaly_results),
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(report_path)

    def _determine_status(self, posture: Dict, anomaly: Dict) -> str:
        """Determina status geral baseado nos resultados."""
        total_alerts = posture.get("total_alerts", 0)
        total_anomalies = anomaly.get("total_anomalies", 0)

        if total_anomalies > 10 or total_alerts > 50:
            return "critical"
        elif total_anomalies > 0 or total_alerts > 10:
            return "warning"
        else:
            return "normal"
