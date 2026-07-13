"""
Cliente Azure Speech to Text para transcricao de consultas medicas.
"""

import os
from pathlib import Path
from typing import Dict, Optional


class AzureSpeechClient:
    """Cliente para Azure Speech Services."""

    def __init__(self, subscription_key: Optional[str] = None,
                 region: Optional[str] = None):
        """
        Args:
            subscription_key: Azure subscription key.
                              Se None, usa env AZURE_SPEECH_KEY.
            region: Azure region. Se None, usa env AZURE_SPEECH_REGION.
        """
        self.subscription_key = subscription_key or os.getenv("AZURE_SPEECH_KEY")
        self.region = region or os.getenv("AZURE_SPEECH_REGION", "eastus")
        self._available = bool(self.subscription_key)

    @property
    def available(self) -> bool:
        return self._available

    def transcribe(self, audio_path: str, language: str = "pt-BR") -> Dict:
        """
        Transcreve audio usando Azure Speech to Text.

        Returns:
            Dict com transcricao, confianca e metadata
        """
        if not self.available:
            return {
                "error": "Azure Speech nao configurado. "
                         "Defina AZURE_SPEECH_KEY e AZURE_SPEECH_REGION.",
                "transcription": "",
            }

        try:
            import azure.cognitiveservices.speech as speechsdk

            speech_config = speechsdk.SpeechConfig(
                subscription=self.subscription_key,
                region=self.region,
            )
            speech_config.speech_recognition_language = language

            audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )

            result = recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return {
                    "success": True,
                    "transcription": result.text,
                    "confidence": 1.0,  # Azure SDK nao expoe confianca por utterance
                    "language": language,
                    "duration_seconds": getattr(result, "duration", 0),
                }
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return {
                    "success": False,
                    "transcription": "",
                    "error": f"NoMatch: {result.no_match_details}",
                }
            else:
                return {
                    "success": False,
                    "transcription": "",
                    "error": f"Recognition failed: {result.reason}",
                }

        except ImportError:
            return {
                "error": "azure-cognitiveservices-speech nao instalado. "
                         "Instale com: pip install azure-cognitiveservices-speech",
                "transcription": "",
            }
        except Exception as e:
            return {
                "error": str(e),
                "transcription": "",
            }

    def transcribe_continuous(self, audio_path: str,
                              language: str = "pt-BR") -> Dict:
        """Transcreve audio longo (ate 60 min)"""
        if not self.available:
            return {"error": "Azure Speech nao configurado.", "segments": []}

        try:
            import azure.cognitiveservices.speech as speechsdk

            speech_config = speechsdk.SpeechConfig(
                subscription=self.subscription_key, region=self.region
            )
            speech_config.speech_recognition_language = language
            audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )

            segments = []

            def recognized_cb(evt):
                segments.append({
                    "text": evt.result.text,
                    "offset_ms": evt.result.offset / 10000,
                    "duration_ms": evt.result.duration / 10000,
                })

            recognizer.recognized.connect(recognized_cb)
            recognizer.start_continuous_recognition()

            import time
            time.sleep(getattr(audio_config, "duration", 10))

            recognizer.stop_continuous_recognition()

            full_text = " ".join(s["text"] for s in segments)

            return {
                "success": True,
                "full_transcription": full_text,
                "segments": segments,
                "segment_count": len(segments),
                "language": language,
            }

        except Exception as e:
            return {"error": str(e), "segments": []}

    def transcribe_file(self, audio_path: str) -> str:
        """Interface simples: retorna texto transcrito."""
        result = self.transcribe(audio_path)
        return result.get("transcription", "")
