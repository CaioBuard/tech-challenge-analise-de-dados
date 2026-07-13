"""
Analise de sentimento e termos criticos em transcricoes de consultas medicas.

Usa Azure Text Analytics (se configurado) ou fallback local com dicionario.
"""

from typing import Dict, Optional


def analyze_sentiment(text: str, language: str = "pt-BR") -> Dict:
    """
    Analisa sentimento de texto transcrito de consulta medica.

    Args:
        text: Texto transcrito da consulta
        language: Idioma (pt-BR, en-US)

    Returns:
        Dict com sentimento, scores, termos criticos e recomendacao
    """
    from src.azure_integration.text_analytics import AzureTextAnalyticsClient

    client = AzureTextAnalyticsClient()

    if client.available:
        print("[SENTIMENT] Usando Azure Text Analytics...")
    else:
        print("[SENTIMENT] Azure nao configurado, usando fallback local...")

    # Analise de sentimento
    sentiment = client.analyze_sentiment(text, language)

    # Extracao de termos criticos
    critical = client.extract_critical_terms(text, language)

    # Recomendacao baseada na analise
    is_critical = (
        critical.get("is_critical", False)
        or sentiment.get("sentiment") == "negative"
    )

    recommendation = "normal"
    if is_critical:
        if sentiment.get("sentiment") == "negative" and critical.get("is_critical"):
            recommendation = "immediate_attention"
        elif critical.get("is_critical"):
            recommendation = "review_required"
        else:
            recommendation = "monitor"

    RECOMMENDATION_TEXT = {
        "normal": "Consulta sem alertas detectados.",
        "monitor": "Monitorar paciente - sentimento negativo detectado.",
        "review_required": "Revisar consulta - termos medicos criticos identificados.",
        "immediate_attention": "ATENCAO IMEDIATA - Sentimento negativo e termos criticos detectados.",
    }

    result = {
        "text_preview": text[:300] + "..." if len(text) > 300 else text,
        "sentiment": sentiment,
        "critical_terms": critical,
        "is_critical": is_critical,
        "recommendation": recommendation,
        "recommendation_text": RECOMMENDATION_TEXT.get(recommendation, ""),
    }

    print(f"[SENTIMENT] Resultado: {recommendation}")
    print(f"[SENTIMENT] Sentimento: {sentiment.get('sentiment', '?')}")
    print(f"[SENTIMENT] Termos criticos: {critical.get('critical_terms_found', 0)}")

    return result
