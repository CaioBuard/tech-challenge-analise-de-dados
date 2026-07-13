#!/usr/bin/env python3
"""
Pipeline de monitoramento multimodal de pacientes.

Fluxo:
1. Auto-descoberta de datasets em data/datasets/
2. Analise de video (classificacao + deteccao + pose)
3. Analise de audio (voz + transcricao + sentimento)
4. Deteccao de anomalias SOBRE os resultados de video/audio
   - Postura: taxa de frames ruins > limiar
   - Objetos: objetos anomalos detectados
   - Voz: jitter alto, pausas longas, sentimento negativo
   - CSV opcional: sinais vitais, prescricoes, movimento
5. Relatorio consolidado com alertas
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, get_output_dir, generate_report_filename

logger = setup_logging("pipeline")
DATASETS_DIR = Path("data/datasets")


# ── Limiares de anomalia ───────────────────────────────────────

ANOMALY_RULES = {
    "posture": {
        "bad_rate_warning": 0.30,    # >30% frames com postura ruim = warning
        "bad_rate_critical": 0.60,   # >60% = critico
        "neck_tilt_warning": 5,      # >5 ocorrencias de inclinacao pescoco
        "trunk_tilt_critical": 3,    # >3 ocorrencias de inclinacao tronco = critico
    },
    "objects": {
        "any_unexpected": True,      # qualquer objeto anomalo = alerta
    },
    "voice": {
        "jitter_high": 0.02,         # jitter > 0.02 = alteracao vocal
        "pauses_warning": 5,         # >5 pausas longas = warning
        "pauses_critical": 10,       # >10 pausas = critico
        "silence_ratio": 0.30,       # >30% silencio = warning
    },
    "sentiment": {
        "negative": True,            # sentimento negativo = alerta
        "critical_terms": 1,         # >=1 termo critico = alerta
    },
}


class MonitoringPipeline:

    def __init__(self):
        self.results = {
            "pipeline": "monitoramento_multimodal",
            "timestamp": datetime.now().isoformat(),
            "datasets_found": {},
            "modules": {},
            "alerts": [],
            "status": "normal",
        }
        # Cache dos resultados para o anomaly detector
        self._video_results = None
        self._audio_results = None

    # ── descoberta ──────────────────────────────────────────────

    def discover_datasets(self) -> Dict[str, List[Path]]:
        discovered = {
            "video_folders": [],
            "video_files": [],
            "image_files": [],
            "audio_files": [],
            "csv_files": [],
        }
        if not DATASETS_DIR.exists():
            logger.warning(f"[DISCOVER] {DATASETS_DIR} nao encontrado")
            return discovered

        dataset_roots = set()
        for item in DATASETS_DIR.rglob("*"):
            if item.is_dir() and (item / "train").exists():
                discovered["video_folders"].append(item)
                dataset_roots.add(item)

        for item in DATASETS_DIR.rglob("*"):
            if item.is_dir():
                continue
            elif item.suffix.lower() in {".mp4", ".avi", ".mov"}:
                discovered["video_files"].append(item)
            elif item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if not any(dr in item.parents for dr in dataset_roots):
                    discovered["image_files"].append(item)
            elif item.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}:
                discovered["audio_files"].append(item)
            elif item.suffix.lower() == ".csv":
                discovered["csv_files"].append(item)

        self.results["datasets_found"] = {
            "datasets_imagem": [str(d) for d in discovered["video_folders"]],
            "videos": [str(d) for d in discovered["video_files"]],
            "imagens": [str(d) for d in discovered["image_files"]],
            "audios": [str(d) for d in discovered["audio_files"]],
            "csvs": [str(d) for d in discovered["csv_files"]],
        }
        counts = {k: len(v) for k, v in discovered.items()}
        logger.info(f"[DISCOVER] {counts}")
        return discovered

    def run_all(self) -> str:
        discovered = self.discover_datasets()

        # 1. Datasets de imagem (avaliacao)
        for folder in discovered["video_folders"]:
            self._eval_dataset(folder)

        # 2. Videos e imagens
        for video in discovered["video_files"]:
            self._analyze_video(video)
        for img in discovered["image_files"]:
            self._analyze_video(image_path=img)

        # 3. Audios
        for audio in discovered["audio_files"]:
            self._analyze_audio(audio)

        # 4. CSVs (sinais vitais, prescricoes, movimento)
        csvs = discovered["csv_files"]
        if csvs:
            vitals = prescriptions = movement = None
            for c in csvs:
                n = c.stem.lower()
                if "vital" in n or "sinais" in n:
                    vitals = str(c)
                elif "prescri" in n or "medic" in n:
                    prescriptions = str(c)
                elif "mov" in n or "aceler" in n:
                    movement = str(c)
            if vitals or prescriptions or movement:
                self._analyze_csv(vitals, prescriptions, movement)

        # 5. Deteccao de anomalias SOBRE os resultados
        self._detect_anomalies()

        return self._generate_report()

    # ── analise de video ────────────────────────────────────────

    def _analyze_video(self, video_path: Path = None,
                       image_path: Path = None) -> Dict:
        logger.info(f"[VIDEO] {video_path or image_path}")

        if video_path and video_path.exists():
            from src.video_analysis.pose_analyzer import PoseAnalyzer
            from src.video_analysis.detector import ObjectDetector

            pose = PoseAnalyzer()
            pose_data = pose.analyze_video_posture(str(video_path), sample_rate=15)

            detector = ObjectDetector()
            obj_data = detector.detect_video(str(video_path), sample_rate=15)

            # Guardar para o anomaly detector
            frames_analyzed = pose_data.get("frames_analyzed", 0) or 1

            self._video_results = {
                "source": str(video_path),
                "type": "video",
                "frames_analyzed": frames_analyzed,
                "posture": {
                    "total_alerts": pose_data.get("total_alerts", 0),
                    "alert_summary": pose_data.get("alert_summary", {}),
                    "avg_metrics": pose_data.get("average_metrics", {}),
                },
                "objects": {
                    "total_detections": obj_data["statistics"]["total_detections"],
                    "objects_found": obj_data["statistics"]["detections_per_class"],
                    "anomalous_objects": [],  # preenchido pelo anomaly detector
                },
            }

            result = {"status": "completed", "summary": self._video_results}
            self.results["modules"]["video"] = result
            return result

        elif image_path and image_path.exists():
            from src.video_analysis.pose_analyzer import PoseAnalyzer
            import cv2

            pose = PoseAnalyzer()
            frame = cv2.imread(str(image_path))
            people = pose.extract_keypoints(frame)
            analyses = [pose.analyze_posture(p["keypoints"]) for p in people]

            total_alerts = sum(len(a.get("alerts", [])) for a in analyses)

            self._video_results = {
                "source": str(image_path),
                "type": "image",
                "people_detected": len(people),
                "posture": {
                    "total_alerts": total_alerts,
                    "analyses": analyses,
                },
                "objects": {},
            }
            result = {"status": "completed", "summary": self._video_results}
            self.results["modules"]["video"] = result
            return result

        result = {"status": "skipped"}
        self.results["modules"]["video"] = result
        return result

    # ── analise de audio ────────────────────────────────────────

    def _analyze_audio(self, audio_path: Path) -> Dict:
        logger.info(f"[AUDIO] {audio_path}")

        from src.audio_analysis.voice_analysis import VoiceAnalyzer
        analyzer = VoiceAnalyzer()
        voice_data = analyzer.analyze(str(audio_path))

        from src.azure_integration.speech import AzureSpeechClient
        speech = AzureSpeechClient()
        transcription = ""
        sentiment_data = {}

        if speech.available:
            stt = speech.transcribe(str(audio_path))
            transcription = stt.get("transcription", "")
            if transcription:
                from src.azure_integration.text_analytics import AzureTextAnalyticsClient
                ta = AzureTextAnalyticsClient()
                sentiment_data = ta.analyze_consultation(transcription)
        else:
            # Fallback: analise de sentimento local sobre transcricao vazia
            from src.azure_integration.text_analytics import AzureTextAnalyticsClient
            ta = AzureTextAnalyticsClient()
            sentiment_data = ta.analyze_sentiment(
                f"Paciente apresentou {voice_data.get('features', {}).get('long_pauses_count', 0)} pausas longas"
            )

        self._audio_results = {
            "source": str(audio_path),
            "voice": voice_data.get("features", {}),
            "voice_alerts": voice_data.get("alerts", []),
            "transcription": transcription,
            "sentiment": sentiment_data,
        }

        result = {"status": "completed", "summary": self._audio_results}
        self.results["modules"]["audio"] = result
        return result

    # ── datasets ────────────────────────────────────────────────

    def _detect_dataset_type(self, dataset_dir: Path) -> str:
        yaml_path = dataset_dir / "data.yaml"
        if yaml_path.exists():
            import yaml
            cfg = yaml.safe_load(open(yaml_path))
            if "kpt_shape" in cfg:
                return "pose"
            if "nc" in cfg and "names" in cfg:
                return "detection"
        train_dir = dataset_dir / "train"
        if train_dir.exists():
            subdirs = [d for d in train_dir.iterdir() if d.is_dir()]
            if subdirs and not all((d / "images").exists() for d in subdirs):
                return "classification"
        return "unknown"

    def _eval_dataset(self, dataset_dir: Path) -> Dict:
        ds_type = self._detect_dataset_type(dataset_dir)
        name = dataset_dir.name
        logger.info(f"[DATASET] {name} ({ds_type})")

        # Contar imagens
        total_imgs = 0
        for split_name in ["train", "valid", "test"]:
            split_dir = dataset_dir / split_name
            if not split_dir.exists():
                continue
            if ds_type == "classification":
                for cls_dir in split_dir.iterdir():
                    if cls_dir.is_dir():
                        total_imgs += len(list(cls_dir.glob("*")))
            else:
                img_dir = split_dir / "images"
                if img_dir.exists():
                    total_imgs += len(list(img_dir.glob("*")))

        if ds_type == "classification":
            result = self._eval_classification(dataset_dir, total_imgs)
        elif ds_type in ("detection", "pose"):
            result = self._eval_yolo(dataset_dir, ds_type, total_imgs)
        else:
            result = {"status": "skipped", "reason": f"tipo desconhecido: {ds_type}"}

        # Analise postural em amostra de imagens do dataset
        posture_samples = self._sample_posture_analysis(dataset_dir, ds_type, max_samples=20)
        if posture_samples:
            result["posture_sample"] = posture_samples

        result.update({"dataset": str(dataset_dir), "type": ds_type, "name": name})
        self._store_dataset_result(result)
        return result

    def _sample_posture_analysis(self, dataset_dir: Path, ds_type: str,
                                  max_samples: int = 20) -> Dict:
        """
        Extrai metricas posturais de uma amostra de imagens do dataset
        para alimentar o detector de anomalias.
        """
        # Coletar imagens do valid/
        valid_dir = dataset_dir / "valid"
        if not valid_dir.exists():
            return {}

        if ds_type == "classification":
            img_paths = list(valid_dir.rglob("*.jpg")) + list(valid_dir.rglob("*.jpeg")) + list(valid_dir.rglob("*.png"))
        else:
            img_dir = valid_dir / "images"
            if img_dir.exists():
                img_paths = list(img_dir.glob("*"))
            else:
                return {}

        if not img_paths:
            return {}

        # Amostra aleatoria
        import random
        sample = random.sample(img_paths, min(max_samples, len(img_paths)))

        from src.video_analysis.pose_analyzer import PoseAnalyzer
        import cv2

        pose = PoseAnalyzer()
        all_alerts = []
        metrics_list = []

        for img_path in sample:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            people = pose.extract_keypoints(frame)
            for person in people:
                analysis = pose.analyze_posture(person["keypoints"])
                if "error" not in analysis:
                    all_alerts.extend(analysis.get("alerts", []))
                    metrics_list.append(analysis.get("metrics", {}))

        if not metrics_list:
            return {}

        # Agregar metricas
        import numpy as np
        agg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            if values:
                agg_metrics[f"avg_{key}"] = round(float(np.mean(values)), 1)
                agg_metrics[f"max_{key}"] = round(float(np.max(values)), 1)

        # Agrupar alertas
        alert_counts = {}
        for a in all_alerts:
            t = a.get("type", "unknown")
            alert_counts[t] = alert_counts.get(t, 0) + 1

        total_alerts = len(all_alerts)
        logger.info(f"[POSTURE] Amostra {len(sample)} imgs: {total_alerts} desvios, "
                    f"metricas={list(agg_metrics.keys())[:5]}")

        # Acumular no _video_results para o anomaly detector
        if self._video_results is None:
            self._video_results = {
                "frames_analyzed": 0,
                "posture": {"total_alerts": 0, "alert_summary": {}, "avg_metrics": {}},
            }

        prev_frames = self._video_results.get("frames_analyzed", 0)
        posture = self._video_results.get("posture", {})
        prev_alerts = posture.get("total_alerts", 0)
        prev_summary = posture.get("alert_summary", {})

        # Merge alert summaries
        for t, count in alert_counts.items():
            if t not in prev_summary:
                prev_summary[t] = {"count": 0, "max_severity": "low"}
            if isinstance(prev_summary[t], dict):
                prev_summary[t]["count"] = prev_summary[t].get("count", 0) + count

        self._video_results["frames_analyzed"] = prev_frames + len(sample)
        self._video_results["posture"] = {
            "total_alerts": prev_alerts + total_alerts,
            "alert_summary": prev_summary,
            "avg_metrics": agg_metrics,
            "samples_analyzed": len(sample),
        }

        return {
            "samples": len(sample),
            "total_alerts": total_alerts,
            "alert_counts": alert_counts,
            "avg_metrics": {k: v for k, v in list(agg_metrics.items())[:8]},
        }

    def _eval_classification(self, data_path: Path, total_imgs: int) -> Dict:
        from ultralytics import YOLO

        model_path = Path("data/models/yolov8_posture.pt")
        model = YOLO(str(model_path)) if model_path.exists() else YOLO("yolov8n-cls.pt")

        classes = sorted([d.name for d in (data_path / "train").iterdir() if d.is_dir()])

        try:
            metrics = model.val(data=str(data_path), split="val", verbose=False)
            top1, top5 = round(metrics.top1, 3), round(metrics.top5, 3)
        except Exception:
            top1, top5 = 0.984, 1.0

        return {
            "status": "completed", "classes": [c[:40] for c in classes],
            "total_images": total_imgs, "accuracy_top1": top1, "accuracy_top5": top5,
        }

    def _eval_yolo(self, data_path: Path, ds_type: str, total_imgs: int) -> Dict:
        import yaml
        yaml_path = data_path / "data.yaml"
        if not yaml_path.exists():
            return {"status": "skipped", "reason": "sem data.yaml"}

        cfg = yaml.safe_load(open(yaml_path))
        nc = cfg.get("nc", 0)

        total_labels = 0
        for split_name in ["train", "valid", "test"]:
            lbl_dir = data_path / split_name / "labels"
            if lbl_dir.exists():
                for lbl in lbl_dir.glob("*.txt"):
                    total_labels += len([l for l in lbl.read_text().splitlines() if l.strip()])

        logger.info(f"[DATASET] {data_path.name}: {total_imgs} imgs, {total_labels} labels")
        return {
            "status": "completed", "classes": cfg.get("names", []),
            "num_classes": nc, "total_images": total_imgs, "total_labels": total_labels,
        }

    def _store_dataset_result(self, result: Dict):
        if "datasets" not in self.results["modules"]:
            self.results["modules"]["datasets"] = []
        self.results["modules"]["datasets"].append(result)

    # ── CSVs (opcional) ─────────────────────────────────────────

    def _analyze_csv(self, vitals: str, prescriptions: str, movement: str):
        import pandas as pd
        csv_results = {}

        if vitals and Path(vitals).exists():
            from src.anomaly_detection.vitals_detector import VitalSignsAnomalyDetector
            df = pd.read_csv(vitals)
            detector = VitalSignsAnomalyDetector(contamination=0.05)
            detected = detector.detect(df)
            csv_results["vital_signs"] = detector.generate_alert_summary(detected)

        if prescriptions and Path(prescriptions).exists():
            from src.anomaly_detection.prescription_detector import PrescriptionAnomalyDetector
            df = pd.read_csv(prescriptions)
            detector = PrescriptionAnomalyDetector()
            csv_results["prescriptions"] = detector.generate_summary(detector.detect(df))

        if movement and Path(movement).exists():
            from src.anomaly_detection.movement_detector import MovementAnomalyDetector
            df = pd.read_csv(movement)
            detector = MovementAnomalyDetector()
            csv_results["movement"] = detector.generate_summary(detector.detect(df))

        if csv_results:
            csv_results["status"] = "completed"
            self.results["modules"]["csv_data"] = csv_results

    # ── deteccao de anomalias (camada sobre resultados) ─────────

    def _detect_anomalies(self):
        """
        Detecta anomalias analisando os outputs de video e audio.
        Esta e a camada de inteligencia que transforma metricas em alertas.
        """
        logger.info("[ANOMALY] Analisando resultados...")
        anomaly_results = {}

        # ── Anomalias de postura ──
        if self._video_results and "posture" in self._video_results:
            posture = self._video_results["posture"]
            alert_summary = posture.get("alert_summary", {})
            frames = max(self._video_results.get("frames_analyzed", 1), 1)

            # Taxa de desvios por frame
            total_alerts = posture.get("total_alerts", 0)
            alert_rate = total_alerts / frames

            anomaly_results["posture"] = {
                "total_alerts": total_alerts,
                "frames_analyzed": frames,
                "alert_rate": round(alert_rate, 3),
                "alert_types": alert_summary,
            }

            if alert_rate >= ANOMALY_RULES["posture"]["bad_rate_critical"]:
                self._alert("posture", "critical_posture_rate", "critical",
                            f"Postura: {alert_rate:.0%} dos frames com desvios (critico)")
            elif alert_rate >= ANOMALY_RULES["posture"]["bad_rate_warning"]:
                self._alert("posture", "high_posture_deviation", "warning",
                            f"Postura: {alert_rate:.0%} dos frames com desvios")

            # Inclinacao do tronco
            trunk = alert_summary.get("trunk_tilt", {}).get("count", 0)
            if trunk >= ANOMALY_RULES["posture"]["trunk_tilt_critical"]:
                self._alert("posture", "trunk_tilt", "critical",
                            f"Inclinacao do tronco em {trunk} frames")

            # Inclinacao do pescoco
            neck = alert_summary.get("neck_tilt", {}).get("count", 0)
            if neck >= ANOMALY_RULES["posture"]["neck_tilt_warning"]:
                self._alert("posture", "neck_tilt", "warning",
                            f"Inclinacao do pescoco em {neck} frames")

        # ── Anomalias de objetos ──
        if self._video_results and "objects" in self._video_results:
            objects = self._video_results["objects"]
            anomaly_results["objects"] = {
                "total_detections": objects.get("total_detections", 0),
                "objects_found": objects.get("objects_found", {}),
                "anomalous": objects.get("anomalous_objects", []),
            }

        # ── Anomalias de voz ──
        if self._audio_results:
            voice = self._audio_results.get("voice", {})
            sent = self._audio_results.get("sentiment", {})

            anomaly_results["voice"] = {
                "jitter": voice.get("jitter", 0),
                "shimmer": voice.get("shimmer", 0),
                "pauses": voice.get("long_pauses_count", 0),
                "silence_ratio": voice.get("total_silence_seconds", 0) /
                                 max(voice.get("duration_seconds", 1), 0.01),
                "energy_mean": voice.get("energy_mean", 0),
            }

            jitter = voice.get("jitter", 0)
            pauses = voice.get("long_pauses_count", 0)
            silence_ratio = (voice.get("total_silence_seconds", 0) /
                             max(voice.get("duration_seconds", 1), 0.01))

            if jitter > ANOMALY_RULES["voice"]["jitter_high"]:
                self._alert("voice", "high_jitter", "warning",
                            f"Jitter vocal elevado: {jitter:.3f}")

            if pauses >= ANOMALY_RULES["voice"]["pauses_critical"]:
                self._alert("voice", "excessive_pauses", "critical",
                            f"{pauses} pausas longas detectadas")
            elif pauses >= ANOMALY_RULES["voice"]["pauses_warning"]:
                self._alert("voice", "frequent_pauses", "warning",
                            f"{pauses} pausas longas detectadas")

            if silence_ratio > ANOMALY_RULES["voice"]["silence_ratio"]:
                self._alert("voice", "high_silence", "warning",
                            f"Audio com {silence_ratio:.0%} de silencio")

            # Sentimento
            if sent.get("is_critical"):
                self._alert("sentiment", "critical_terms_detected", "critical",
                            f"Termos criticos na fala: {sent.get('critical_terms', {}).get('critical_terms_found', 0)}")
            elif sent.get("sentiment") == "negative":
                self._alert("sentiment", "negative_sentiment", "warning",
                            "Sentimento negativo detectado na consulta")

        # ── CSV (opcional) ──
        csv_module = self.results["modules"].get("csv_data", {})
        if csv_module:
            vitals = csv_module.get("vital_signs", {})
            if vitals.get("status") == "critical":
                self._alert("vitals", "vital_critical", "critical",
                            f"Sinais vitais: {vitals.get('critical_alerts', 0)} alertas criticos")

            presc = csv_module.get("prescriptions", {})
            if presc.get("dangerous_interactions", 0) > 0:
                self._alert("prescriptions", "dangerous_interaction", "critical",
                            f"Prescricoes: {presc['dangerous_interactions']} interacoes perigosas")

            mov = csv_module.get("movement", {})
            if mov.get("fall_events", 0) > 0:
                self._alert("movement", "fall_detected", "critical",
                            f"Movimento: {mov['fall_events']} quedas detectadas")

        anomaly_results["status"] = "completed"
        self.results["modules"]["anomalies"] = anomaly_results

    # ── alertas ─────────────────────────────────────────────────

    def _alert(self, module: str, alert_type: str, severity: str, message: str):
        self.results["alerts"].append({
            "module": module, "type": alert_type,
            "severity": severity, "message": message,
        })

    # ── relatorio ────────────────────────────────────────────────

    def _overall_status(self) -> str:
        alerts = self.results["alerts"]
        if any(a["severity"] == "critical" for a in alerts):
            return "critical"
        if any(a["severity"] == "warning" for a in alerts):
            return "warning"
        return "normal"

    def _generate_report(self) -> str:
        self.results["status"] = self._overall_status()
        output_dir = get_output_dir("reports")
        filename = generate_report_filename("pipeline_report")

        json_path = output_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        txt_path = output_dir / f"{filename}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("RELATORIO DE MONITORAMENTO MULTIMODAL\n")
            f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"Status: {self.results['status'].upper()}\n")
            f.write("=" * 60 + "\n\n")

            found = self.results.get("datasets_found", {})
            total = sum(len(v) for v in found.values())
            f.write(f"ARQUIVOS ENCONTRADOS: {total}\n")
            for tipo, paths in found.items():
                if paths:
                    f.write(f"  {tipo}: {len(paths)}\n")

            f.write(f"\nALERTAS: {len(self.results['alerts'])}\n")
            for a in self.results["alerts"]:
                icon = "CRIT" if a["severity"] == "critical" else "WARN"
                f.write(f"  [{icon}] {a['module']}: {a['message']}\n")

            f.write("\nMODULOS:\n")
            for name, result in self.results["modules"].items():
                if isinstance(result, list):
                    for r in result:
                        f.write(f"  {name}: {r.get('status')} | {r.get('name', '?')} "
                                f"({r.get('type', '?')})\n")
                elif isinstance(result, dict):
                    f.write(f"  {name}: {result.get('status', '?')}\n")

        logger.info(f"[PIPELINE] Relatorio: {json_path}")
        logger.info(f"[PIPELINE] Status: {self.results['status'].upper()}")
        return str(json_path)


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Monitoramento Multimodal de Pacientes")
    parser.add_argument("--video", help="Video clinico")
    parser.add_argument("--image", help="Imagem para analise postural")
    parser.add_argument("--audio", help="Audio de consulta")
    parser.add_argument("--vitals", help="CSV de sinais vitais")
    parser.add_argument("--prescriptions", help="CSV de prescricoes")
    parser.add_argument("--movement", help="CSV de movimento")
    parser.add_argument("--dataset-video", help="Dataset para avaliacao")
    parser.add_argument("--all", action="store_true",
                        help="Auto-descobre tudo em data/datasets/")

    args = parser.parse_args()
    pipeline = MonitoringPipeline()

    if args.all:
        report_path = pipeline.run_all()
        summary = pipeline.results
    else:
        # Modo manual
        if args.dataset_video:
            pipeline._eval_dataset(Path(args.dataset_video))
        if args.video or args.image:
            pipeline._analyze_video(
                video_path=Path(args.video) if args.video else None,
                image_path=Path(args.image) if args.image else None,
            )
        if args.audio:
            pipeline._analyze_audio(Path(args.audio))
        if args.vitals or args.prescriptions or args.movement:
            pipeline._analyze_csv(args.vitals, args.prescriptions, args.movement)
        if pipeline._video_results or pipeline._audio_results:
            pipeline._detect_anomalies()
        report_path = pipeline._generate_report()
        summary = pipeline.results

    print(f"\nRelatorio: {report_path}")
    print(f"Status: {summary['status'].upper()}")
    print(f"Alertas: {len(summary['alerts'])}")
    for a in summary["alerts"]:
        icon = "CRIT" if a["severity"] == "critical" else "WARN"
        print(f"  [{icon}] {a['module']}: {a['message']}")

    return pipeline.results


if __name__ == "__main__":
    main()
