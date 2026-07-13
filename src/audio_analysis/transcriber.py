"""
Transcricao de audio usando Azure Speech to Text.

Suporte a:
- Azure Cognitive Services (se configurado)
- Fallback local via SpeechRecognition
"""

from pathlib import Path
from typing import Dict, Optional


def transcribe(audio_path: str, language: str = "pt-BR") -> Dict:
    """
    Transcreve audio de consulta medica.

    Prioridade: Azure Speech to Text > Whisper local > SpeechRecognition

    Returns:
        Dict com transcricao, confianca e metadata
    """
    from src.azure_integration.speech import AzureSpeechClient

    client = AzureSpeechClient()

    if client.available:
        print("[TRANSCRIBER] Usando Azure Speech to Text...")
        result = client.transcribe(audio_path, language=language)
    else:
        print("[TRANSCRIBER] Azure nao configurado, usando fallback local...")
        result = _transcribe_local(audio_path)

    if "transcription" in result and result["transcription"]:
        print(f"[TRANSCRIBER] Transcricao: {result['transcription'][:200]}...")
    else:
        print(f"[TRANSCRIBER] Falha na transcricao: {result.get('error', 'desconhecido')}")

    return result


def _transcribe_local(audio_path: str) -> Dict:
    """Fallback local usando SpeechRecognition."""
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio, language="pt-BR")
            return {
                "success": True,
                "transcription": text,
                "method": "google_speech_recognition",
            }
        except sr.UnknownValueError:
            return {
                "success": False,
                "transcription": "",
                "error": "Google Speech Recognition nao entendeu o audio",
            }
        except sr.RequestError:
            return {
                "success": False,
                "transcription": "",
                "error": "Erro no servico Google Speech Recognition",
            }

    except ImportError:
        # Tentar whisper se disponivel
        try:
            import whisper
            model = whisper.load_model("tiny")
            result = model.transcribe(audio_path, language="pt")
            return {
                "success": True,
                "transcription": result["text"],
                "method": "whisper_tiny",
                "segments": result.get("segments", []),
            }
        except ImportError:
            return {
                "success": False,
                "transcription": "",
                "error": "Nenhum motor de transcricao disponivel. "
                         "Instale: pip install SpeechRecognition ou whisper",
            }
        except Exception as e:
            return {
                "success": False,
                "transcription": "",
                "error": f"Erro no whisper: {str(e)}",
            }
    except Exception as e:
        return {
            "success": False,
            "transcription": "",
            "error": f"Erro na transcricao local: {str(e)}",
        }
