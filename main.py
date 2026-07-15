import argparse
import json
import os
from datetime import datetime

from transcriber import transcribe_medical_audio
from analyzer import analyze_consultation

OUTPUT_DIR = "output"

def salvar_resultados(nome_base: str, transcript: dict, analysis: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    transcript_path = os.path.join(OUTPUT_DIR, f"{nome_base}_{timestamp}_transcricao.json")
    analysis_path = os.path.join(OUTPUT_DIR, f"{nome_base}_{timestamp}_analise.json")

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return transcript_path, analysis_path


def exibir_resumo(transcript: dict, analysis: dict) -> None:
    print("\n" + "=" * 72)
    print("TRANSCRICAO (por falante)")
    print("=" * 72)
    for utterance in transcript.get("utterances") or []:
        falante = utterance.get("speaker", "?")
        print(f"[{falante}] {utterance.get('text', '')}")

    print("\n" + "=" * 72)
    print("ANALISE CLINICA PRELIMINAR - apoio a decisao, requer revisao medica")
    print("=" * 72)
    print(f"Queixa principal: {analysis.get('chief_complaint')}")

    sintomas = analysis.get("symptoms_reported") or []
    print(f"Sintomas relatados: {', '.join(sintomas) if sintomas else '-'}")

    print("\nCondicoes possiveis a considerar (isto NAO e um diagnostico):")
    condicoes = analysis.get("possible_conditions") or []
    if not condicoes:
        print("  Nenhuma hipotese especifica identificada com base na transcricao.")
    for condicao in condicoes:
        print(f"  - {condicao.get('condition')} (confianca: {condicao.get('confidence')})")
        print(f"    Justificativa: {condicao.get('rationale')}")

    red_flags = analysis.get("red_flags") or []
    if red_flags:
        print("\n[ALERTA] Sinais de urgencia mencionados na consulta:")
        for sinal in red_flags:
            print(f"  - {sinal}")

    print(f"\nRecomendacao sugerida: {analysis.get('recommendation')}")

    if analysis.get("notes"):
        print(f"Observacoes: {analysis['notes']}")

    print(f"\n{analysis.get('disclaimer')}")
    print("=" * 72)


def preparar_transcricao_para_analise(transcript: dict) -> str:
    utterances = transcript.get("utterances") or []
    linhas = []

    for utterance in utterances:
        texto = (utterance.get("text") or "").strip()
        if not texto:
            continue

        falante = utterance.get("speaker") or "Speaker"
        linhas.append(f"{falante}: {texto}")

    if linhas:
        return "\n".join(linhas)

    return transcript.get("text") or ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcreve uma consulta medica e gera uma analise clinica preliminar."
    )
    parser.add_argument("audio_path", help="Caminho para o arquivo de audio da consulta")
    args = parser.parse_args()

    if not os.path.isfile(args.audio_path):
        raise SystemExit(f"Arquivo nao encontrado: {args.audio_path}")

    nome_base = os.path.splitext(os.path.basename(args.audio_path))[0]

    try:
        transcript = transcribe_medical_audio(args.audio_path)
        print("Transcricao concluida. Gerando analise clinica preliminar...")
        analysis = analyze_consultation(preparar_transcricao_para_analise(transcript))
    except Exception as exc:
        raise SystemExit(f"Erro durante o processamento: {exc}") from exc

    exibir_resumo(transcript, analysis)
    salvar_resultados(nome_base, transcript, analysis)

if __name__ == "__main__":
    main()
