"""
Tech Challenge Fase 4 - Avaliacao de Videos Clinicos
"""

import argparse
import sys
from pathlib import Path

from src.video_analysis import VideoEvaluator


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def discover_videos(directory: Path) -> list[str]:
    """Descobre videos em um diretorio."""
    return [
        str(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Avaliacao de Videos Clinicos - Tech Challenge Fase 4"
    )
    parser.add_argument("--input", help="Caminho de um video")
    parser.add_argument("--dir", help="Diretorio com videos")
    parser.add_argument("--sample-rate", type=int, default=15, help="Amostragem de frames")
    parser.add_argument("--conf", type=float, default=0.25, help="Confianca minima")

    args = parser.parse_args()

    video_paths: list[str] = []
    if args.input:
        video_paths = [args.input]
    elif args.dir:
        video_paths = discover_videos(Path(args.dir))
    if not video_paths:
        print("Nenhum video encontrado para avaliacao.")
        parser.print_help()
        sys.exit(1)

    evaluator = VideoEvaluator(sample_rate=args.sample_rate, conf=args.conf)

    results = evaluator.evaluate_many(video_paths)
    print(f"\nVideos avaliados: {len(results)}")
    for result in results:
        print(f"  {result['video_name']}: {result['status'].upper()}")
        print(
            f"    Frames analisados: {result['frames_analyzed']} | "
            f"Objetos detectados: {result['total_detections']}"
        )
        print(
            f"    Frames com objetos inesperados: "
            f"{result['frames_with_unexpected_objects']} | "
            f"Taxa: {result['anomaly_rate']:.2%}"
        )
        print(
            f"    Presenca do paciente: "
            f"{result['procedure_summary']['patient_presence_rate']:.0%} | "
            f"Multiplas pessoas: {result['procedure_summary']['frames_with_multiple_people']}"
        )
        if result["unexpected_objects"]:
            print(
                f"    Objetos inesperados: "
                + ", ".join(result["unexpected_objects"])
            )
        else:
            print("    Objetos inesperados: nenhum")
        print("    Resumo procedimental:")
        for finding in result["procedure_findings"]:
            print(f"      - [{finding['severity'].upper()}] {finding['evidence']}")
        print(f"    Relatorio: {result['report_path']}")


if __name__ == "__main__":
    main()
