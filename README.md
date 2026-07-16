# Tech Challenge - Fase 4

**Avaliacao de Videos Clinicos com YOLOv8**

Este projeto foi migrado para um escopo unico: **avaliar videos**.

## Objetivo

Receber videos clinicos ou videos de demonstracao de fisioterapia e produzir:

- leitura e amostragem do video;
- deteccao de objetos por frame com `YOLOv8`;
- identificacao de objetos inesperados;
- relatorio final por video.

## O que o projeto faz hoje

- avalia um video individual;
- avalia uma pasta com varios videos;
- gera relatorios em `data/outputs/reports/`;
- resume objetos detectados, eventos inesperados e taxa de anomalia.

## O que saiu do projeto

- pipeline multimodal;
- analise de audio;
- Azure;
- sinais vitais;
- pose estimation;
- treinamento e avaliacao de datasets.

## Como executar

### Avaliar um video

```bash
python3 main.py --input data/datasets/video/raw/kimore/fisioterapia_01.mp4
```

### Avaliar uma pasta de videos

```bash
python3 main.py --dir data/datasets/video/raw/kimore
```

## Estrutura recomendada

```text
data/datasets/video/raw/
└── kimore/
    ├── fisioterapia_01.mp4
    └── fisioterapia_02.mp4
```

## Saida

Para cada video, o projeto gera:

- um `.json` com o resumo da avaliacao;
- um `.txt` legivel para apresentacao.
