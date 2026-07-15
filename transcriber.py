"""
Transcricao de consultas medicas usando a API da AssemblyAI.

Funcionalidades ativadas:
- Medical Mode (domain="medical-v1"): maior acuracia em termos clinicos
  (nomes de medicamentos, procedimentos, condicoes etc.).
- Speaker Diarization + Speaker Identification por papel (Doctor / Patient),
  para deixar claro no transcript quem disse o que.
- Redacao de PII: remove dados que identificam o paciente (nome, telefone,
  e-mail, data de nascimento etc.) SEM remover o conteudo clinico (sintomas,
  condicoes, medicamentos), que e necessario para a etapa de analise.

Referencia: https://www.assemblyai.com/docs/speech-to-text
"""

import time
import requests

from config import (
    ASSEMBLYAI_API_KEY,
    ASSEMBLYAI_BASE_URL,
    MEDICAL_DOMAIN,
    LANGUAGE_CODE,
)

HEADERS = {"authorization": ASSEMBLYAI_API_KEY}

PII_POLICIES = [
    "person_name",
    "phone_number",
    "email_address",
    "date_of_birth",
    "us_social_security_number",
    "location",
    "banking_information",
]


def upload_audio(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        response = requests.post(
            f"{ASSEMBLYAI_BASE_URL}/v2/upload", headers=HEADERS, data=audio_file
        )
    response.raise_for_status()
    return response.json()["upload_url"]


def request_transcription(audio_url: str, redact_pii: bool = True) -> str:
    payload = {
        "audio_url": audio_url,
        "speech_models": ["universal-3-5-pro", "universal-2"],
        "domain": MEDICAL_DOMAIN,
        "language_code": LANGUAGE_CODE,
        "speaker_labels": True,
        "speech_understanding": {
            "request": {
                "speaker_identification": {
                    "speaker_type": "role",
                    "known_values": ["Doctor", "Patient"],
                }
            }
        },
    }

    if redact_pii:
        payload.update(
            {
                "redact_pii": True,
                "redact_pii_policies": PII_POLICIES,
                "redact_pii_sub": "hash",
            }
        )

    response = requests.post(
        f"{ASSEMBLYAI_BASE_URL}/v2/transcript", headers=HEADERS, json=payload
    )
    if not response.ok:
        print(response.status_code)
        print(response.text)
        response.raise_for_status() 

    return response.json()["id"]


def poll_transcription(transcript_id: str, poll_interval: float = 3.0) -> dict:
    endpoint = f"{ASSEMBLYAI_BASE_URL}/v2/transcript/{transcript_id}"

    while True:
        response = requests.get(endpoint, headers=HEADERS)
        response.raise_for_status()
        transcript = response.json()

        status = transcript["status"]
        if status == "completed":
            return transcript
        if status == "error":
            raise RuntimeError(f"Falha na transcricao: {transcript.get('error')}")

        time.sleep(poll_interval)


def transcribe_medical_audio(file_path: str, redact_pii: bool = True) -> dict:
    print(f"[1/3] Enviando audio para a AssemblyAI: {file_path}")
    audio_url = upload_audio(file_path)

    print("[2/3] Transcrevendo (Medical Mode + diarizacao de falantes)...")
    transcript_id = request_transcription(audio_url, redact_pii=redact_pii)

    print("[3/3] Aguardando conclusao do processamento...")
    transcript = poll_transcription(transcript_id)

    return transcript
