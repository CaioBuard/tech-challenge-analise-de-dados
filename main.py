"""
Tech Challenge Fase 4 - Monitoramento Multimodal de Pacientes com IA e Azure

Módulos:
- video_analysis: Análise de vídeos clínicos (postura, cirurgia, fisioterapia)
- audio_analysis: Processamento de áudio de consultas (Azure Speech to Text)
- anomaly_detection: Detecção de anomalias em sinais vitais e prescrições
- azure_integration: Integração com Azure Cognitive Services
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Monitoramento Multimodal de Pacientes - Tech Challenge Fase 4"
    )
    subparsers = parser.add_subparsers(dest="module", help="Módulo a executar")

    # Video
    video_parser = subparsers.add_parser("video", help="Análise de vídeo")
    video_parser.add_argument("--input", required=True, help="Caminho do vídeo")
    video_parser.add_argument("--task", choices=["posture", "surgery", "physio"],
                              default="posture", help="Tipo de análise")

    # Audio
    audio_parser = subparsers.add_parser("audio", help="Análise de áudio")
    audio_parser.add_argument("--input", required=True, help="Caminho do áudio")
    audio_parser.add_argument("--task", choices=["transcribe", "sentiment", "voice"],
                              default="transcribe", help="Tipo de análise")

    # Anomaly
    anomaly_parser = subparsers.add_parser("anomaly", help="Detecção de anomalias")
    anomaly_parser.add_argument("--input", required=True, help="Caminho do CSV")
    anomaly_parser.add_argument("--type", choices=["vitals", "prescription", "movement"],
                                default="vitals", help="Tipo de dados")

    args = parser.parse_args()

    if args.module == "video":
        from src.video_analysis import analyze
        analyze(args.input, args.task)
    elif args.module == "audio":
        from src.audio_analysis import analyze
        analyze(args.input, args.task)
    elif args.module == "anomaly":
        from src.anomaly_detection import analyze
        analyze(args.input, args.type)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
