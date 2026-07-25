import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from interview_analisys.transcriber import transcribe_medical_audio
from interview_analisys.analyzer import analyze_consultation

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


def construir_texto_para_exibicao(transcript: dict, analysis: dict) -> str:
    linhas = ["TRANSCRICAO"]
    utterances = transcript.get("utterances") or []

    if utterances:
        for utterance in utterances:
            texto = (utterance.get("text") or "").strip()
            if not texto:
                continue
            falante = utterance.get("speaker") or "Speaker"
            linhas.append(f"[{falante}] {texto}")
    else:
        linhas.append(transcript.get("text") or "Nenhuma transcricao disponivel.")

    linhas.append("")
    linhas.append("ANALISE CLINICA PRELIMINAR")
    linhas.append(f"Queixa principal: {analysis.get('chief_complaint') or '-'}")

    sintomas = analysis.get("symptoms_reported") or []
    linhas.append(f"Sintomas relatados: {', '.join(sintomas) if sintomas else '-'}")
    linhas.append("")
    linhas.append("Condicoes possiveis a considerar:")

    condicoes = analysis.get("possible_conditions") or []
    if not condicoes:
        linhas.append("- Nenhuma hipotese especifica identificada com base na transcricao.")
    for condicao in condicoes:
        linhas.append(
            f"- {condicao.get('condition')} (confianca: {condicao.get('confidence')})"
        )
        linhas.append(f"  Justificativa: {condicao.get('rationale')}")

    red_flags = analysis.get("red_flags") or []
    if red_flags:
        linhas.append("")
        linhas.append("Sinais de urgencia mencionados:")
        for sinal in red_flags:
            linhas.append(f"- {sinal}")

    linhas.append("")
    linhas.append(f"Recomendacao sugerida: {analysis.get('recommendation') or '-'}")

    if analysis.get("notes"):
        linhas.append(f"Observacoes: {analysis['notes']}")

    disclaimer = analysis.get("disclaimer")
    if disclaimer:
        linhas.append("")
        linhas.append(disclaimer)

    return "\n".join(linhas)


def executar_transcricao_e_analise(audio_path: str) -> dict[str, Any]:
    if not os.path.isfile(audio_path):
        raise SystemExit(f"Arquivo nao encontrado: {audio_path}")

    nome_base = os.path.splitext(os.path.basename(audio_path))[0]

    try:
        transcript = transcribe_medical_audio(audio_path)
        print("Transcricao concluida. Gerando analise clinica preliminar...")
        analysis = analyze_consultation(preparar_transcricao_para_analise(transcript))
    except Exception as exc:
        raise SystemExit(f"Erro durante o processamento: {exc}") from exc

    exibir_resumo(transcript, analysis)
    salvar_resultados(nome_base, transcript, analysis)

    return {
        "type": "text",
        "filename": os.path.basename(audio_path),
        "path": audio_path,
        "message": "Transcricao e analise concluídas.",
        "text": construir_texto_para_exibicao(transcript, analysis),
    }

if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "CAR0001.mp3"
    executar_transcricao_e_analise(audio_path)
