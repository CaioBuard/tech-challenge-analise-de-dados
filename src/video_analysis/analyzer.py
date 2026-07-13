"""
Módulo de Análise de Vídeo

Processa vídeos clínicos para detectar:
- Postura (classificação com YOLOv8-cls)
- Movimentos/eventos anômalos (YOLOv8 + OpenPose)
- Áreas críticas e instrumentos cirúrgicos (YOLOv8-detect)
"""

from pathlib import Path


def analyze(input_path: str, task: str = "posture"):
    """
    Analisa um vídeo ou imagem clínica.

    Args:
        input_path: Caminho do vídeo ou imagem
        task: Tipo de análise ('posture', 'surgery', 'physio')
    """
    print(f"[VIDEO] Iniciando análise de vídeo: {input_path}")
    print(f"[VIDEO] Tarefa: {task}")

    if task == "posture":
        analyze_posture(input_path)
    elif task == "surgery":
        analyze_surgery(input_path)
    elif task == "physio":
        analyze_physiotherapy(input_path)


def analyze_posture(input_path: str):
    """Classificação postural usando YOLOv8-cls treinado."""
    from ultralytics import YOLO

    model_path = Path(__file__).parent.parent.parent / "data" / "models" / "yolov8_posture.pt"

    if model_path.exists():
        model = YOLO(str(model_path))
        print("[VIDEO] Usando modelo treinado yolov8_posture.pt")
    else:
        print("[VIDEO] Modelo treinado nao encontrado, usando YOLOv8n-cls pretrained...")
        model = YOLO("yolov8n-cls.pt")

    results = model(input_path)
    for r in results:
        top1_class = r.names[r.probs.top1]
        top1_conf = r.probs.top1conf
        print(f"  Classe: {top1_class} - Confianca: {top1_conf:.2f}")

        # Top 3 classes
        top3_idx = r.probs.top3
        for i, idx in enumerate(top3_idx):
            print(f"    #{i+1}: {r.names[idx]} ({r.probs.data[idx]:.2f})")


def analyze_surgery(input_path: str):
    """Deteccao de instrumentos e fases cirurgicas com YOLOv8."""
    from .detector import ObjectDetector

    detector = ObjectDetector()

    # Classes esperadas em ambiente cirurgico
    expected = ["scalpel", "scissors", "forceps", "syringe"]
    unexpected = ["cell phone", "person", "cup", "bottle", "book"]

    results = detector.detect_anomalous_objects(
        input_path,
        expected_classes=expected,
        unexpected_classes=unexpected,
        conf=0.25,
    )

    print(f"\n[VIDEO-SURGERY] Analise concluida:")
    print(f"  Anomalias detectadas: {results['total_anomalies']}")
    print(f"  Taxa de anomalia: {results['anomaly_rate']:.2%}")

    for anomaly in results["anomalies"][:5]:
        print(f"  Frame {anomaly['frame']}: {anomaly['object']} "
              f"[{anomaly['severity']}]")


def analyze_physiotherapy(input_path: str):
    """Analise de sessoes de fisioterapia com pose estimation."""
    from .pose_analyzer import PoseAnalyzer
    from .report_generator import VideoReportGenerator

    pose = PoseAnalyzer()
    reporter = VideoReportGenerator()

    results = pose.analyze_video_posture(input_path, sample_rate=15)

    print(f"\n[VIDEO-PHYSIO] Analise concluida:")
    print(f"  Frames analisados: {results['frames_analyzed']}")
    print(f"  Pessoas detectadas: {results['people_detected']}")
    print(f"  Alertas: {results['total_alerts']}")

    if results.get("average_metrics"):
        print("\n  Metricas medias:")
        for k, v in results["average_metrics"].items():
            print(f"    {k}: {v}")

    if results.get("alert_summary"):
        print("\n  Alertas por tipo:")
        for alert_type, info in results["alert_summary"].items():
            print(f"    {alert_type}: {info['count']}x [{info['max_severity']}]")

    report_path = reporter.generate_posture_report(
        results, Path(input_path).stem
    )
    print(f"\n  Relatorio: {report_path}")
