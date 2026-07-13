"""
Cliente Azure Text Analytics para analise de sentimento e extracao de termos.

Analisa transcricoes de consultas medicas para:
- Sentimento (positivo, negativo, neutro)
- Termos medicos criticos
- Entidades relevantes
"""

import os
from typing import Dict, List, Optional


# Termos medicos criticos para monitorar (fallback sem Azure)
CRITICAL_TERMS = {
    "pt": [
        "dor", "forte", "insuportavel", "sangramento", "falta de ar",
        "desmaio", "convulsao", "infarto", "avc", "parada", "emergencia",
        "urgente", "alergia", "reacao", "infeccao", "febre alta",
        "nao consigo respirar", "pressao baixa", "pressao alta",
        "nao estou bem", "piorou", "grave", "critico",
    ],
    "en": [
        "pain", "severe", "unbearable", "bleeding", "shortness of breath",
        "fainting", "seizure", "heart attack", "stroke", "emergency",
        "urgent", "allergy", "reaction", "infection", "high fever",
        "cannot breathe", "low blood pressure", "high blood pressure",
        "not feeling well", "worsened", "severe", "critical",
    ],
}


class AzureTextAnalyticsClient:
    """Cliente para Azure Text Analytics."""

    def __init__(self, endpoint: Optional[str] = None,
                 key: Optional[str] = None):
        self.endpoint = endpoint or os.getenv("AZURE_TEXT_ENDPOINT")
        self.key = key or os.getenv("AZURE_TEXT_KEY")
        self._available = bool(self.endpoint and self.key)

    @property
    def available(self) -> bool:
        return self._available

    def analyze_sentiment(self, text: str, language: str = "pt-BR") -> Dict:
        """
        Analisa sentimento do texto.

        Returns:
            Dict com sentiment, confidence scores, e sentences
        """
        if not self.available:
            return self._fallback_sentiment(text, language)

        try:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential

            client = TextAnalyticsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key),
            )

            documents = [{"id": "1", "text": text, "language": language}]
            response = client.analyze_sentiment(documents)

            result = response[0]
            if result.is_error:
                return {"error": f"Azure error: {result.error}"}

            return {
                "success": True,
                "sentiment": result.sentiment,
                "positive_score": round(result.confidence_scores.positive, 3),
                "neutral_score": round(result.confidence_scores.neutral, 3),
                "negative_score": round(result.confidence_scores.negative, 3),
                "sentences": [
                    {
                        "text": s.text,
                        "sentiment": s.sentiment,
                        "positive": round(s.confidence_scores.positive, 3),
                        "negative": round(s.confidence_scores.negative, 3),
                    }
                    for s in result.sentences
                ],
            }

        except ImportError:
            return self._fallback_sentiment(text, language)
        except Exception as e:
            return {"error": str(e)}

    def extract_critical_terms(self, text: str,
                               language: str = "pt") -> Dict:
        """
        Extrai termos medicos criticos do texto.

        Usa Azure Entity Recognition se disponivel, senao fallback local.
        """
        if self.available:
            return self._extract_entities_azure(text, language)
        else:
            return self._extract_terms_local(text, language)

    def _extract_entities_azure(self, text: str, language: str) -> Dict:
        """Extrai entidades via Azure Text Analytics."""
        try:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential

            client = TextAnalyticsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key),
            )

            documents = [{"id": "1", "text": text, "language": language}]
            response = client.recognize_entities(documents)
            result = response[0]

            if result.is_error:
                return {"error": str(result.error)}

            medical_entities = [
                {
                    "text": e.text,
                    "category": e.category,
                    "confidence": round(e.confidence_score, 3),
                }
                for e in result.entities
                if e.category in [
                    "MedicationName", "Dosage", "MedicalEvent",
                    "Symptom", "Diagnosis", "TreatmentName",
                ]
            ]

            return {
                "success": True,
                "critical_terms_found": len(medical_entities),
                "entities": medical_entities,
                "is_critical": len(medical_entities) > 0,
            }

        except Exception as e:
            return {"error": str(e)}

    def _extract_terms_local(self, text: str, language: str) -> Dict:
        """Fallback local para extracao de termos criticos."""
        terms_list = CRITICAL_TERMS.get(language[:2], CRITICAL_TERMS["pt"])
        text_lower = text.lower()

        found = []
        for term in terms_list:
            if term in text_lower:
                found.append(term)

        return {
            "success": True,
            "method": "local_fallback",
            "critical_terms_found": len(found),
            "terms": found,
            "is_critical": len(found) > 0,
        }

    def _fallback_sentiment(self, text: str, language: str) -> Dict:
        """Fallback local para analise de sentimento."""
        terms_list = CRITICAL_TERMS.get(language[:2], CRITICAL_TERMS["pt"])
        text_lower = text.lower()

        negative_count = sum(1 for t in terms_list if t in text_lower)

        if negative_count > 3:
            sentiment = "negative"
            neg_score = min(0.9, 0.5 + negative_count * 0.1)
        elif negative_count > 0:
            sentiment = "mixed"
            neg_score = 0.3 + negative_count * 0.1
        else:
            sentiment = "neutral"
            neg_score = 0.1

        return {
            "success": True,
            "method": "local_fallback",
            "sentiment": sentiment,
            "negative_score": round(neg_score, 3),
            "neutral_score": round(1 - neg_score - 0.1, 3),
            "positive_score": 0.1 if negative_count > 0 else 0.6,
            "sentences": [],
        }

    def analyze_consultation(self, transcription: str,
                             language: str = "pt-BR") -> Dict:
        """
        Analise completa de uma consulta transcrita.

        Returns:
            Sentimento + termos criticos + recomendacao
        """
        sentiment = self.analyze_sentiment(transcription, language)
        terms = self.extract_critical_terms(transcription, language)

        is_critical = (
            terms.get("is_critical", False)
            or sentiment.get("sentiment") == "negative"
        )

        recommendation = "normal"
        if is_critical:
            if sentiment.get("sentiment") == "negative" and terms.get("is_critical"):
                recommendation = "immediate_attention"
            elif terms.get("is_critical"):
                recommendation = "review_required"
            else:
                recommendation = "monitor"

        return {
            "transcription": transcription[:200] + "..." if len(transcription) > 200 else transcription,
            "sentiment": sentiment,
            "critical_terms": terms,
            "is_critical": is_critical,
            "recommendation": recommendation,
            "recommendation_text": {
                "normal": "Consulta sem alertas detectados.",
                "monitor": "Monitorar paciente - sentimento negativo detectado.",
                "review_required": "Revisar - termos criticos detectados na fala.",
                "immediate_attention": "ATENCAO IMEDIATA - Termos criticos "
                                       "e sentimento negativo detectados.",
            }.get(recommendation, ""),
        }
