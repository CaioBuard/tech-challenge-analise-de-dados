# Tech Challenge - Fase 4

**Monitoramento Multimodal de Pacientes com IA e Azure**

Projeto da Fase 4 do Tech Challenge - Pós-Graduação.

## Objetivo

Sistema de monitoramento contínuo de pacientes por meio de dados multimodais
(áudio, vídeo e texto) para identificar sinais precoces de risco, integrado com
serviços gerenciados em nuvem (Azure Cognitive Services).

## Módulos

### 1. Análise de Vídeo (`src/video_analysis/`)
- Processamento de vídeos clínicos (cirurgias, fisioterapia)
- Detecção postural com OpenPose
- Detecção de objetos e áreas críticas com YOLOv8
- Geração de relatórios automáticos de desvios

### 2. Análise de Áudio (`src/audio_analysis/`)
- Processamento de áudios de consultas médicas
- Detecção de alterações vocais (fadiga, disartria)
- Transcrição com Azure Speech to Text
- Análise de sentimentos e termos críticos com Azure Text Analytics

### 3. Detecção de Anomalias (`src/anomaly_detection/`)
- Análise de séries temporais de sinais vitais
- Detecção de alterações em prescrições
- Padrões de movimentação do paciente
- Alertas automáticos para equipe médica

### 4. Integração Azure (`src/azure_integration/`)
- Azure Speech to Text
- Azure Text Analytics
- Azure Cognitive Services

## Estrutura do Projeto

```
tech-challenge-fase4/
├── data/
│   ├── datasets/          # Datasets (áudio, vídeo, sinais vitais)
│   ├── models/            # Modelos treinados
│   └── outputs/           # Resultados e relatórios gerados
├── src/
│   ├── video_analysis/    # Módulo de análise de vídeo
│   ├── audio_analysis/    # Módulo de análise de áudio
│   ├── anomaly_detection/ # Módulo de detecção de anomalias
│   ├── azure_integration/ # Integração com Azure
│   └── utils/             # Utilitários compartilhados
├── notebooks/             # Jupyter notebooks exploratórios
├── docs/                  # Documentação
├── reports/               # Relatórios técnicos
└── requirements.txt       # Dependências
```

## Tecnologias

- Python 3.10+
- Ultralytics YOLOv8
- OpenPose (análise postural)
- Azure Cognitive Services (Speech to Text, Text Analytics)
- Scikit-learn / PyOD (detecção de anomalias)
- OpenCV, Librosa (processamento audiovisual)

## Datasets Utilizados

### Análise de Vídeo
- [Sitting Posture Classification](https://universe.roboflow.com/leonardo-sabino/sitting-posture-classification-ccvao-31o25) - 2.347 imagens para classificação postural

### Fontes Sugeridas
- [PhysioNet](https://physionet.org/) - Sinais vitais e dados clínicos
- [Google AudioSet](https://research.google.com/audioset/) - Dataset de áudio
