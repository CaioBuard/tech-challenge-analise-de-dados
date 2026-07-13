# Plano de Implementacao - Tech Challenge Fase 4

> **Goal:** Implementar os 3 modulos de IA: analise de video, deteccao de anomalias e analise de audio.

**Arquitetura:** Pipeline modular com YOLOv8 para video, PyOD/IsolationForest para anomalias, e Azure Cognitive Services para audio.

**Tech Stack:** Python 3.10+, Ultralytics YOLOv8, OpenCV, PyOD, scikit-learn, Azure SDK

---

## Fase 1: Ambiente e Infraestrutura

### Task 1: Criar venv e instalar dependencias
- Criar virtualenv em `tech-challenge-fase4/.venv/`
- Instalar requirements.txt

### Task 2: Baixar modelo YOLOv8 base
- Download do YOLOv8n-cls.pt (classificacao, ~6MB)
- Download do YOLOv8n.pt (deteccao, ~6MB)
- Download do YOLOv8n-pose.pt (pose estimation, ~9MB)

---

## Fase 2: Analise de Video

### Task 3: Treinar classificador postural com dataset Sitting Posture
- Script `src/video_analysis/train_posture.py`
- Treinar YOLOv8-cls no dataset de 2.347 imagens
- Salvar modelo em `data/models/yolov8_posture.pt`

### Task 4: Implementar analise de video completa
- `src/video_analysis/detector.py` - deteccao de objetos com YOLOv8
- `src/video_analysis/pose_analyzer.py` - analise postural com YOLOv8-pose
- `src/video_analysis/report_generator.py` - geracao de relatorios
- Integrar no analyzer.py existente

### Task 5: Testar com video de exemplo
- Criar script de teste que processa um video e gera relatorio

---

## Fase 3: Deteccao de Anomalias

### Task 6: Implementar detector de anomalias em sinais vitais
- `src/anomaly_detection/vitals_detector.py`
- Usar IsolationForest + DBSCAN para detectar outliers
- Gerar alertas com severidade

### Task 7: Implementar detector de anomalias em prescricoes
- `src/anomaly_detection/prescription_detector.py`
- Detectar mudancas bruscas em tratamento

### Task 8: Implementar detector de anomalias em movimento
- `src/anomaly_detection/movement_detector.py`
- Analisar series temporais de acelerometro/pose

### Task 9: Criar gerador de dados sinteticos para teste
- `src/anomaly_detection/synthetic_data.py`
- Gerar sinais vitais realistas com anomalias injetadas

---

## Fase 4: Analise de Audio

### Task 10: Implementar cliente Azure Speech to Text
- `src/azure_integration/speech.py`
- Transcrever audios de consultas

### Task 11: Implementar cliente Azure Text Analytics
- `src/azure_integration/text_analytics.py`
- Analise de sentimentos e extracao de termos criticos

### Task 12: Implementar analise vocal local (fallback sem Azure)
- `src/audio_analysis/voice_analysis.py`
- Extrair features de audio com librosa (pitch, energy, jitter)
- Detectar fadiga vocal, disartria

---

## Fase 5: Pipeline e Relatorios

### Task 13: Criar pipeline integrado
- Script `run_pipeline.py` que orquestra video + audio + anomalias

### Task 14: Gerar relatorio tecnico
- Template em `reports/`
- Resultados e exemplos de anomalias detectadas

---

## Ordem de Execucao

```
[FASE 1] Ambiente -> [FASE 2] Video -> [FASE 3] Anomalias -> [FASE 4] Audio -> [FASE 5] Pipeline
```
