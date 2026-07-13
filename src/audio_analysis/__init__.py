"""Módulo de Análise de Áudio - Azure Speech to Text + Text Analytics."""


def analyze(input_path: str, task: str = "transcribe"):
    print(f"[AUDIO] Iniciando análise de áudio: {input_path}")
    print(f"[AUDIO] Tarefa: {task}")

    if task == "transcribe":
        from .transcriber import transcribe
        transcribe(input_path)
    elif task == "sentiment":
        from .sentiment import analyze_sentiment
        analyze_sentiment(input_path)
    elif task == "voice":
        from .voice_analysis import analyze_voice
        analyze_voice(input_path)
