"""Integração com Azure Cognitive Services."""

from .speech import AzureSpeechClient
from .text_analytics import AzureTextAnalyticsClient

__all__ = ["AzureSpeechClient", "AzureTextAnalyticsClient"]
