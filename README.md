# Assistente de Transcrição e Análise de Consultas Médicas

Transcreve consultas médicas em inglês usando a API da AssemblyAI e gera uma
**análise clínica preliminar** (hipóteses possíveis, não um diagnóstico) para
apoiar — nunca substituir — a avaliação de um profissional de saúde.

> ⚠️ **Aviso importante**
> Este projeto **não diagnostica pacientes**. Ele organiza o que foi dito na
> consulta e sugere possíveis condições a investigar, sempre com nível de
> confiança e justificativa, para que **um médico licenciado** revise e decida.
> Não use a saída deste script como decisão clínica final, e não o implante em
> produção sem validação clínica, avaliação de responsabilidade legal e,
> se for o caso, um BAA (Business Associate Addendum) com a AssemblyAI para
> tratamento de PHI conforme a HIPAA (ou equivalente da LGPD, se aplicável).

## O que o projeto faz

1. **Transcreve** o áudio da consulta com `speaker_labels` (diarização) e
   identificação de papéis, rotulando cada fala como `Doctor` ou `Patient`.
2. Ativa o **Medical Mode** da AssemblyAI (`domain="medical-v1"`), que reduz
   significativamente erros de reconhecimento em termos médicos (nomes de
   medicamentos, condições, procedimentos) em comparação ao modelo padrão.
3. Aplica **redação de PII** (nome, telefone, e-mail, data de nascimento etc.)
   para proteger a identidade do paciente, mantendo intacto o conteúdo clínico
   necessário para a análise.
4. Envia a transcrição para um LLM (via **LLM Gateway** da AssemblyAI, que dá
   acesso a modelos como Claude usando a mesma API key) com um prompt que pede
   uma **análise estruturada em JSON**: queixa principal, sintomas, hipóteses
   possíveis com nível de confiança, sinais de alerta e recomendação de
   próximo passo.
5. Salva transcrição e análise em `output/` e, por padrão, **apaga a
   transcrição dos servidores da AssemblyAI** ao final (minimização de dados).

## Por que "hipóteses" e não "diagnóstico"

Um modelo de linguagem analisando um texto de consulta não tem acesso a exame
físico, exames laboratoriais, histórico completo do prontuário ou capacidade
de exame clínico — só ao que foi verbalizado. Por isso a saída é modelada como
**apoio à decisão clínica** (na linha de ferramentas como Nabla, Suki ou Abridge),
com hipóteses, justificativa e confiança, e não como veredito. Essa é também a
abordagem recomendada pela própria AssemblyAI em seus tutoriais de "AI medical
scribe" (que geram notas SOAP para revisão do médico, não diagnósticos finais).

## Estrutura do projeto

```
medical_consult_assistant/
├── config.py          # Configurações e constantes (modelo, endpoints, etc.)
├── transcriber.py      # Upload + transcrição (Medical Mode, diarização, PII)
├── analyzer.py          # Análise clínica preliminar via LLM Gateway
├── main.py                # Script principal (CLI)
├── requirements.txt
├── .env.example
└── output/                # Transcrições e análises geradas (criado em runtime)
```

## Como usar

### 1. Pré-requisitos

- Python 3.9+
- Uma conta na AssemblyAI com créditos/cartão cadastrado (o Medical Mode e o
  LLM Gateway são recursos pagos, além da transcrição em si) — crie em
  https://www.assemblyai.com/dashboard/signup

### 2. Instalação

```bash
pip install -r requirements.txt
```

### 3. Configuração

```bash
cp .env.example .env
# edite o .env e cole sua chave:
# ASSEMBLYAI_API_KEY=sua_chave_aqui
```

### 4. Execução

```bash
python main.py caminho/para/consulta.mp3
```

Formatos aceitos: qualquer formato suportado pela AssemblyAI (mp3, wav, m4a,
mp4 etc.). Para manter a transcrição salva na plataforma da AssemblyAI (em vez
de apagá-la ao final), use `--manter-transcricao`.

### Exemplo de saída (`output/*_analise.json`)

```json
{
  "chief_complaint": "Persistent headaches for two weeks",
  "symptoms_reported": ["headache", "sensitivity to light", "mild nausea"],
  "relevant_history": ["mother has history of migraines"],
  "possible_conditions": [
    {
      "condition": "Migraine",
      "rationale": "Unilateral headache with photophobia and nausea, plus family history",
      "confidence": "medium"
    }
  ],
  "red_flags": [],
  "recommendation": "Refer to neurology if symptoms persist or worsen; consider headache diary",
  "notes": "No fever or neurological deficits reported.",
  "disclaimer": "This output was generated automatically by a language model..."
}
```

## Privacidade e conformidade

- **PII vs. conteúdo clínico**: `transcriber.py` redige apenas identificadores
  pessoais (nome, telefone, e-mail, data de nascimento, endereço, dados
  bancários). Termos clínicos (sintomas, condições, medicamentos) **não** são
  redigidos, pois são necessários para a análise — avalie se isso é adequado
  ao seu caso de uso e à legislação aplicável.
- **HIPAA / BAA**: a AssemblyAI atua como "business associate" sob a HIPAA e
  oferece BAA para contas que processam PHI em produção — solicite com o time
  de vendas da AssemblyAI antes de usar dados reais de pacientes.
- **Minimização de dados**: por padrão, o script apaga a transcrição dos
  servidores da AssemblyAI ao final da execução (`delete_transcript`).
- **LGPD**: se for usar com pacientes no Brasil, avalie as bases legais e
  requisitos da LGPD para dados de saúde (dado sensível), incluindo
  consentimento explícito e finalidade específica.

## Limitações e pontos de atenção

- **Modelos e preços mudam com frequência.** O identificador do modelo em
  `config.py` (`LLM_MODEL`) e os preços do Medical Mode citados nos comentários
  refletem a documentação da AssemblyAI em julho/2026. Confira a lista atual
  de modelos em https://www.assemblyai.com/docs/llm-gateway/overview antes de
  usar em produção.
- **Qualidade depende do áudio**: ruído, sobreposição de falas e sotaques
  fortes reduzem a acurácia da transcrição e, por consequência, da análise.
- **O parser de JSON tem fallback**: se o modelo eventualmente responder algo
  que não seja um JSON válido, `analyzer.py` retorna a resposta bruta no campo
  `raw_response` em vez de quebrar a execução — vale revisar esses casos.
- Este projeto foi desenhado para **áudio pré-gravado** (processamento
  assíncrono). Para transcrição em tempo real durante a consulta, a AssemblyAI
  também oferece uma API de streaming (Universal-Streaming), que exigiria uma
  arquitetura diferente (WebSocket).

## Possíveis extensões

- Gerar nota clínica em formato SOAP (Subjective, Objective, Assessment, Plan)
  além das hipóteses diagnósticas.
- Integrar com um prontuário eletrônico (EHR).
- Adicionar streaming em tempo real para transcrição ao vivo durante a consulta.
- Adicionar autenticação/multiusuário se for virar um serviço com várias contas.
