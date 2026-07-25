# Plataforma multimodal de análise clínica

Projeto acadêmico que reúne quatro componentes:

- `clinical_anomaly_detection`: detecção de padrões incomuns em sinais vitais,
  prescrições e movimentações hospitalares;
- `video_analysis`: avaliação do enquadramento e do ambiente de vídeos com
  YOLOv8;
- `interview_analisys`: transcrição de entrevistas médicas e geração de uma
  análise clínica preliminar;
- `web`: interface Flask que recebe um ZIP e encaminha cada arquivo ao
  componente adequado.

> **Aviso clínico:** este projeto é uma prova de conceito para apoio à análise.
> Seus alertas, classificações e hipóteses não são diagnósticos e não devem
> orientar decisões clínicas sem revisão de um profissional habilitado. O uso
> com dados reais exige validação técnica e clínica, controles de segurança e
> avaliação de conformidade com a LGPD e demais normas aplicáveis.

## Visão geral

A interface web é o ponto de entrada integrado. O usuário envia um arquivo ZIP
contendo uma ou mais planilhas, gravações de áudio e vídeos. Um fluxo LangGraph
extrai o conteúdo, classifica os arquivos por extensão e executa o processador
correspondente.

```text
ZIP enviado pelo usuário
        |
        v
extração segura -> classificação -> processamento
                                      |
                 +--------------------+--------------------+
                 |                    |                    |
                 v                    v                    v
          Excel clínico            áudio                vídeo
          Isolation Forest    AssemblyAI + Gemini       YOLOv8
                 |                    |                    |
                 +--------------------+--------------------+
                                      |
                                      v
                         resultados na interface web
```

Os componentes também podem ser usados diretamente em Python, sem iniciar a
aplicação web.

## Tecnologias principais

- Python, pandas e NumPy para tratamento dos dados;
- scikit-learn para os modelos Isolation Forest;
- matplotlib para os gráficos clínicos;
- AssemblyAI para transcrição, diarização e redação de PII;
- Google Gemini para análise estruturada da transcrição;
- OpenCV e Ultralytics YOLOv8 para análise dos vídeos;
- Flask e Jinja2 para a interface;
- LangGraph para orquestração do upload.

## Estrutura do repositório

```text
.
|-- clinical_anomaly_detection/   # Detecção de anomalias clínicas
|-- interview_analisys/           # Transcrição e análise da entrevista
|-- video_analysis/               # Detecção e avaliação de vídeo
|-- web/                          # Aplicação Flask e fluxo do ZIP
|   |-- example/                  # Planilhas e ZIPs de demonstração
|   |-- static/
|   `-- templates/
|-- dataset/                      # MIMIC-IV Clinical Database Demo 2.2
|-- models/                       # Modelos clínicos persistidos em joblib
|-- reports/                      # Gráficos de testes clínicos
|-- config.py                     # Configuração das APIs externas
|-- requirements.txt
|-- test_anomaly_detection.py     # Demonstração normal x anômala
`-- test_web_app.py               # Testes auxiliares da apresentação web
```

O nome do pacote de entrevistas está grafado no código como
`interview_analisys`. Use essa grafia nos imports e comandos. O pacote de vídeo
usa o nome `video_analysis`.

## Pré-requisitos e instalação

- Python 3.9 ou superior;
- acesso à internet para as APIs de áudio e para o download inicial do modelo
  YOLOv8;
- chaves da AssemblyAI e do Google AI para processar áudios.

No Windows PowerShell:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No Linux ou macOS:

```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Variáveis de ambiente

Crie um arquivo `.env` na raiz:

```dotenv
ASSEMBLYAI_API_KEY=sua_chave_da_assemblyai
GOOGLE_API_KEY=sua_chave_do_google_ai
```

As configurações utilizadas estão em `config.py`:

- idioma da transcrição: inglês (`en`);
- domínio de transcrição: `medical-v1`;
- modelo de análise: `gemini-3.1-flash-lite`.

As chaves só são necessárias para a análise de áudio. A detecção clínica e a
análise de vídeo não utilizam essas APIs.

## Execução da aplicação web

Na raiz do projeto, execute:

```powershell
python web\app.py
```

Abra `http://localhost:8000`, selecione um arquivo `.zip` e aguarde o
processamento. Há arquivos prontos para demonstração em `web/example/`.

O ZIP pode conter múltiplos arquivos e subpastas. Formatos reconhecidos:

- planilhas: `.xls` e `.xlsx`;
- vídeos: `.mp4`, `.avi`, `.mov`, `.mkv` e `.webm`;
- áudios: `.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg` e `.flac`.

Outros arquivos são marcados como não suportados. A falha no processamento de
um item não interrompe os demais arquivos do mesmo ZIP.

O fluxo executado por `web/flow.py` possui três nós:

```text
extract -> classify -> process
```

Os artefatos da interface são organizados por upload em
`web/generated/<identificador>/`. A extração valida os caminhos internos do ZIP
para impedir que um arquivo seja escrito fora da pasta temporária.

## 1. Detecção de anomalias clínicas

O pacote `clinical_anomaly_detection` utiliza uma abordagem híbrida:

1. regras explícitas destacam limites e eventos previamente definidos;
2. três modelos Isolation Forest procuram padrões diferentes dos observados no
   MIMIC-IV Demo.

O propósito é priorizar dados para revisão. Uma anomalia estatística não
significa necessariamente uma doença ou um erro clínico.

### Dados de referência

O treinamento lê três tabelas do MIMIC-IV Clinical Database Demo 2.2:

- `icu/chartevents.csv.gz`: sinais vitais;
- `hosp/prescriptions.csv.gz`: prescrições;
- `hosp/transfers.csv.gz`: movimentações entre unidades.

De `chartevents`, o projeto seleciona os `itemid` correspondentes a frequência
cardíaca, SpO2 e pressões sistólica, diastólica e média. A coluna `warning` não
é usada como target: o detector atual é não supervisionado e procura padrões
estatisticamente isolados, em vez de tentar reproduzir avisos documentados.

### Features

São treinados modelos independentes porque cada domínio possui uma distribuição
e uma unidade de análise diferente.

**Sinais vitais**

- frequência cardíaca;
- saturação de oxigênio;
- pressões sistólica, diastólica e média;
- variações temporais da frequência cardíaca, da SpO2 e da pressão sistólica.

**Prescrições**

- total de prescrições e medicamentos únicos;
- quantidade de prescrições intravenosas;
- dose média e máxima;
- quantidade de mudanças de dose e de via.

**Transferências**

- total de movimentações e unidades únicas;
- duração total, média e mínima das permanências;
- passagens por UTI;
- transferências com permanência inferior a seis horas.

### Pipeline de aprendizado

Cada domínio utiliza o mesmo pipeline:

```text
SimpleImputer(mediana)
        -> StandardScaler
        -> IsolationForest
```

O imputador trata valores ausentes, o scaler padroniza as features e o
Isolation Forest aprende o padrão predominante sem precisar de rótulos. O
modelo usa 200 árvores, `random_state=42` e `contamination=0.08`.

Na predição:

- `1` representa uma observação classificada como normal;
- `-1` representa uma observação classificada como anômala;
- scores menores indicam observações mais isoladas;
- score negativo fica do lado anômalo da fronteira do modelo.

`contamination=0.08` é uma hipótese técnica usada para definir a fronteira de
decisão. Não significa que 8% dos pacientes estejam doentes.

Os artefatos são gravados em `models/` com a versão do scikit-learn. Se forem
incompatíveis com o ambiente atual, o serviço tenta treinar novamente.

### Regras complementares

As regras atuais geram alertas para:

- frequência cardíaca acima de 120 ou abaixo de 45 bpm;
- SpO2 abaixo de 90%;
- pressão sistólica abaixo de 90 ou acima de 180 mmHg;
- pressão diastólica abaixo de 50 ou acima de 120 mmHg;
- oito ou mais prescrições;
- presença de prescrições IV;
- quatro ou mais movimentações;
- passagem por UTI ou unidade intensiva.

Esses limites são simplificações do protótipo e precisam de validação clínica.
Alertas das regras e dos modelos podem coexistir.

### Uso direto

```python
from clinical_anomaly_detection import ClinicalAnomalyDetector

detector = ClinicalAnomalyDetector(
    dataset_path="dataset/mimic-iv-clinical-database-demo-2.2",
    model_dir="models",
)

status = detector.train_if_needed()  # "loaded" ou "trained"

patient = {
    "vitals": [
        {
            "charttime": "2150-01-01 08:00:00",
            "heart_rate": 82,
            "spo2": 96,
            "systolic_bp": 120,
            "diastolic_bp": 75,
            "mean_bp": 90,
        }
    ],
    "prescriptions": [],
    "transfers": [],
}

alerts = detector.predict_patient(patient)
report = detector.generate_patient_report(patient, output_dir="reports/patient")
```

`predict_patient()` retorna alertas padronizados com categoria, severidade,
título, mensagem e score opcional. `generate_patient_report()` retorna:

- `alerts`: alertas das regras e dos modelos;
- `scores`: score médio de cada categoria;
- `images`: caminhos dos gráficos produzidos.

Os gráficos são:

- `vital_timeseries.png`: evolução temporal dos sinais vitais;
- `feature_comparison.png`: boxplots da referência e médias do paciente;
- `anomaly_scores.png`: score médio de cada categoria.

O gráfico de comparação usa a média do paciente e pode suavizar uma medição
extrema. A conclusão deve considerar os alertas, os scores, a série temporal e
o contexto clínico, não apenas esse boxplot.

Para executar a demonstração com um paciente estável e outro propositalmente
extremo:

```powershell
python test_anomaly_detection.py
```

## 2. Formato das planilhas clínicas

A aplicação converte uma planilha em um `patient_input`. O arquivo pode ter até
três abas; abas ausentes viram listas vazias. Os nomes das abas não diferenciam
maiúsculas de minúsculas.

### Aba `vitals`

| Coluna | Descrição |
|---|---|
| `charttime` | Data e hora da medição |
| `heart_rate` | Frequência cardíaca em bpm |
| `spo2` | Saturação de oxigênio em percentual |
| `systolic_bp` | Pressão sistólica em mmHg |
| `diastolic_bp` | Pressão diastólica em mmHg |
| `mean_bp` | Pressão arterial média em mmHg |

### Aba `prescriptions`

| Coluna | Descrição |
|---|---|
| `starttime` | Início da prescrição |
| `stoptime` | Fim ou suspensão |
| `drug` | Medicamento |
| `dose_val_rx` | Valor da dose |
| `dose_unit_rx` | Unidade, como `mg`, `g` ou `mL` |
| `route` | Via, como `PO`, `IV` ou `IM` |

### Aba `transfers`

| Coluna | Descrição |
|---|---|
| `eventtype` | Evento, como `admit`, `transfer` ou `discharge` |
| `careunit` | Unidade de cuidado |
| `intime` | Entrada na unidade |
| `outtime` | Saída da unidade |

Os exemplos `web/example/paciente_sem_anomalia.xlsx` e
`web/example/paciente_com_anomalia.xlsx` podem ser usados como referência.

## 3. Análise de vídeo

O pacote `video_analysis` avalia requisitos básicos do enquadramento de um
procedimento gravado. Ele não reconhece procedimentos, não avalia a técnica do
profissional e não interpreta a condição clínica do paciente.

### Funcionamento

1. O OpenCV abre o vídeo e coleta metadados.
2. Um frame é analisado a cada 15 frames por padrão.
3. O `yolov8n.pt` detecta objetos com confiança mínima padrão de 0,25.
4. As detecções são consolidadas em indicadores e achados.
5. Opcionalmente são gerados relatórios JSON e TXT.

Na primeira execução, o Ultralytics pode baixar automaticamente o arquivo
`yolov8n.pt`.

### Critérios atuais

- **Baixa visibilidade:** a classe `person` aparece em menos de 70% dos frames
  analisados; abaixo de 50%, a severidade é crítica.
- **Cena cheia:** três ou mais pessoas aparecem em pelo menos 20% dos frames.
- **Objetos inesperados:** `cell phone`, `bottle`, `scissors` ou `knife` com
  confiança mínima de 0,45; gera achado se ocorrer em pelo menos três frames ou
  em 1% dos frames. A partir de 20%, a severidade é crítica.

O status final é `normal`, `warning` ou `critical`, conforme o achado de maior
severidade.

### Uso direto

```python
from video_analysis import VideoEvaluator

evaluator = VideoEvaluator(
    sample_rate=15,
    conf=0.25,
    report_dir="reports/video",
)

result = evaluator.evaluate("caminho/para/video.mp4")
print(result["status"])
print(result["procedure_findings"])
```

O resultado inclui duração, FPS, frames analisados, objetos encontrados,
presença de pessoas, taxa de cena cheia, objetos inesperados, achados e caminhos
dos relatórios.

### Limitações do vídeo

- A classe `person` é usada como aproximação da presença do paciente; o modelo
  não distingue paciente, profissional e acompanhante.
- Resultado depende de iluminação, oclusão, posição da câmera e qualidade.
- A amostragem pode deixar de observar eventos entre os frames processados.
- Os objetos e limiares são regras específicas deste protótipo.

## 4. Transcrição e análise da entrevista

O pacote `interview_analisys` executa duas etapas externas.

### Transcrição com AssemblyAI

O áudio é enviado à AssemblyAI e processado de forma assíncrona com:

- domínio médico `medical-v1`;
- idioma inglês;
- modelos de fala `universal-3-5-pro` e `universal-2`;
- diarização;
- identificação dos papéis `Doctor` e `Patient`;
- redação de PII habilitada.

As políticas de PII incluem nome, telefone, e-mail, nascimento, localização,
informações bancárias e Social Security Number dos Estados Unidos. A redação
substitui os identificadores por hashes, mas o conteúdo clínico continua sendo
processado.

### Análise estruturada com Gemini

A transcrição identificada por falante é enviada ao Gemini com temperatura
zero e um schema JSON. A resposta contém:

- queixa principal;
- sintomas relatados;
- histórico relevante;
- possíveis condições, justificativa e confiança;
- sinais de alerta;
- recomendação;
- observações;
- aviso de que o resultado não é diagnóstico.

O prompt determina que o modelo use somente o conteúdo transcrito e não forneça
diagnóstico definitivo. O parser normaliza a resposta e preserva o texto bruto
em `raw_response` se não conseguir recuperar um JSON válido.

### Execução direta

```powershell
python interview_analisys\transcribe_and_analyze_audio.py caminho\consulta.mp3
```

Sem argumento, o script tenta processar `CAR0001.mp3`. Ele mostra um resumo no
terminal e salva dois arquivos na pasta `output/`:

```text
<audio>_<timestamp>_transcricao.json
<audio>_<timestamp>_analise.json
```

No fluxo web, o conteúdo formatado também é exibido na página de resultados.

### Privacidade e limitações do áudio

- O arquivo e a transcrição são enviados a serviços externos.
- O código atual não solicita a exclusão posterior da transcrição da
  AssemblyAI.
- A redação de PII reduz exposição, mas não torna automaticamente o fluxo
  compatível com LGPD ou HIPAA.
- A qualidade depende do áudio, idioma, ruído, sotaque e sobreposição de falas.
- O Gemini recebe apenas o que foi transcrito: não possui exame físico,
  prontuário completo nem resultados adicionais.
- Toda hipótese e recomendação precisa de revisão profissional.

## Testes

O teste demonstrativo da detecção clínica pode ser executado diretamente:

```powershell
python test_anomaly_detection.py
```

Para executar os testes automatizados disponíveis, instale o pytest caso ele
não esteja presente e rode:

```powershell
python -m pip install pytest
python -m pytest -q
```

Alguns fluxos dependem de APIs, rede, modelo YOLO e arquivos multimídia; por
isso, testes completos de integração exigem esses recursos.

## Saídas geradas

| Pasta | Conteúdo |
|---|---|
| `models/` | Pipelines clínicos em formato joblib |
| `reports/` | Gráficos e relatórios de execuções diretas |
| `output/` | JSONs da transcrição e da análise de áudio |
| `web/generated/` | Gráficos clínicos e relatórios de vídeo por upload |

Essas pastas podem conter dados sensíveis. Em uma implantação real, aplique
autenticação, autorização, criptografia, política de retenção, auditoria e
remoção segura.

## Limitações gerais

- O MIMIC-IV Demo é pequeno e não representa todas as populações ou contextos.
- Os modelos clínicos não foram validados prospectivamente.
- Um score negativo indica anomalia estatística, não gravidade clínica.
- As regras clínicas e de vídeo são heurísticas do protótipo.
- A aplicação Flask usa servidor de desenvolvimento e não está preparada para
  exposição pública.
- Não há autenticação, banco de usuários ou isolamento permanente dos uploads.
- Modelos, APIs e formatos aceitos pelos provedores podem mudar.

Antes de qualquer uso fora de demonstração acadêmica, são necessários testes de
desempenho, avaliação de vieses, validação por especialistas, monitoramento,
gestão de consentimento e uma arquitetura de produção adequada.
