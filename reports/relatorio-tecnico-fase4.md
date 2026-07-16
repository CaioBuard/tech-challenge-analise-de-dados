# Relatorio Tecnico - Avaliacao de Videos

## Visao Geral

O projeto foi reduzido para um unico objetivo: avaliacao de videos clinicos ou
videos de demonstracao assistida com apoio de deteccao visual por `YOLOv8`.

## Fluxo

1. Receber um video ou um diretorio com videos.
2. Amostrar frames em taxa configuravel.
3. Detectar objetos por frame.
4. Consolidar contagens e objetos inesperados.
5. Gerar relatorio por video.

## Saidas

Cada video produz:

- total de frames analisados;
- duracao estimada;
- objetos detectados;
- objetos inesperados;
- eventos inesperados;
- taxa de anomalia.

## Entregavel

O foco da entrega agora e demonstrar avaliacao automatizada de videos com
relatorios objetivos e reprodutiveis.
