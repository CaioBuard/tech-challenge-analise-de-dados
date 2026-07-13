"""
Analisador de voz/audio para deteccao de alteracoes vocais.

Usa librosa para extrair features acusticas e detectar:
- Fadiga vocal (jitter, shimmer reduzidos, energia baixa)
- Disartria (articulacao alterada)
- Dificuldades respiratorias (pausas longas, pitch alterado)

Este e o fallback local quando Azure nao esta disponivel.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional


class VoiceAnalyzer:
    """Analisador de caracteristicas vocais usando librosa."""

    def __init__(self):
        self.sr = None
        self.y = None

    def load_audio(self, audio_path: str) -> bool:
        """Carrega arquivo de audio."""
        try:
            import librosa
            self.y, self.sr = librosa.load(audio_path, sr=16000)
            return True
        except Exception as e:
            print(f"[VOICE] Erro ao carregar audio: {e}")
            return False

    def extract_features(self) -> Dict:
        """Extrai features acusticas do audio."""
        if self.y is None:
            return {"error": "Audio nao carregado"}

        import librosa

        features = {}

        # Energia (RMS)
        rms = librosa.feature.rms(y=self.y)[0]
        features["energy_mean"] = float(np.mean(rms))
        features["energy_std"] = float(np.std(rms))
        features["energy_min"] = float(np.min(rms))

        # Pitch (F0)
        f0, voiced_flag, _ = librosa.pyin(
            self.y,
            fmin=librosa.note_to_hz("C2"),    # ~65 Hz
            fmax=librosa.note_to_hz("C7"),    # ~2093 Hz
            sr=self.sr,
        )
        f0_voiced = f0[voiced_flag] if np.any(voiced_flag) else np.array([0])

        features["pitch_mean_hz"] = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0
        features["pitch_std_hz"] = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0
        features["pitch_min_hz"] = float(np.min(f0_voiced)) if len(f0_voiced) > 0 else 0
        features["pitch_max_hz"] = float(np.max(f0_voiced)) if len(f0_voiced) > 0 else 0
        features["voiced_fraction"] = float(np.mean(voiced_flag))

        # Jitter (variacao de pitch a curto prazo)
        if len(f0_voiced) > 2:
            jitter = np.mean(np.abs(np.diff(f0_voiced))) / (np.mean(f0_voiced) + 1e-8)
            features["jitter"] = float(jitter)
        else:
            features["jitter"] = 0

        # Shimmer (variacao de amplitude)
        if len(rms) > 2:
            shimmer = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-8)
            features["shimmer"] = float(shimmer)
        else:
            features["shimmer"] = 0

        # Zero-crossing rate (articulacao)
        zcr = librosa.feature.zero_crossing_rate(self.y)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        # Spectral centroid (brilho do som)
        spec_cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        features["spectral_centroid_mean"] = float(np.mean(spec_cent))
        features["spectral_centroid_std"] = float(np.std(spec_cent))

        # MFCCs (coeficientes mel-cepstrais)
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        for i in range(13):
            features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))

        # Duracao
        features["duration_seconds"] = float(len(self.y) / self.sr)

        # Detectar pausas longas (silencios > 0.5s)
        silence_threshold = np.mean(rms) * 0.3
        is_silence = rms < silence_threshold
        silence_durations = []
        current_silence = 0
        hop_length = 512
        for s in is_silence:
            if s:
                current_silence += hop_length / self.sr
            else:
                if current_silence > 0.5:
                    silence_durations.append(current_silence)
                current_silence = 0
        features["long_pauses_count"] = len(silence_durations)
        features["total_silence_seconds"] = float(sum(silence_durations))

        return features

    def analyze(self, audio_path: str) -> Dict:
        """
        Analisa audio e detecta alteracoes vocais indicativas
        de condicoes medicas.

        Returns:
            Dict com features, metricas e alertas
        """
        if not self.load_audio(audio_path):
            return {"error": f"Nao foi possivel carregar: {audio_path}"}

        features = self.extract_features()
        if "error" in features:
            return features

        alerts = []
        analysis = {}

        # --- Fadiga vocal ---
        # Baixa energia + jitter/shimmer alterado
        if features.get("energy_mean", 1.0) < 0.02:
            alerts.append({
                "type": "vocal_fatigue",
                "severity": "warning",
                "message": "Baixa energia vocal detectada - possivel fadiga",
            })

        if features.get("jitter", 0) > 0.02:
            alerts.append({
                "type": "high_jitter",
                "severity": "warning",
                "message": f"Jitter elevado: {features['jitter']:.3f}",
            })

        # --- Dificuldade respiratoria ---
        # Pausas longas frequentes + pitch baixo
        if features.get("long_pauses_count", 0) > 5:
            alerts.append({
                "type": "respiratory_difficulty",
                "severity": "medium",
                "message": f"Pausas longas detectadas: "
                           f"{features['long_pauses_count']} pausas",
            })

        if features.get("total_silence_seconds", 0) > features.get("duration_seconds", 60) * 0.3:
            alerts.append({
                "type": "excessive_silence",
                "severity": "medium",
                "message": "Mais de 30% do audio em silencio",
            })

        # --- Disartria (fala arrastada) ---
        # ZCR baixo + spectral centroid baixo
        if features.get("zcr_mean", 1.0) < 0.05 and features.get("voiced_fraction", 0) > 0.3:
            alerts.append({
                "type": "possible_dysarthria",
                "severity": "warning",
                "message": "Padrao de fala pode indicar disartria",
            })

        analysis = {
            "audio_file": audio_path,
            "duration_seconds": features.get("duration_seconds", 0),
            "features": features,
            "alerts": alerts,
            "total_alerts": len(alerts),
            "status": "normal" if len(alerts) == 0
                      else "warning" if all(a["severity"] == "warning" for a in alerts)
                      else "attention_required",
        }

        return analysis


def analyze_voice(input_path: str) -> Dict:
    """Funcao CLI."""
    analyzer = VoiceAnalyzer()
    return analyzer.analyze(input_path)
