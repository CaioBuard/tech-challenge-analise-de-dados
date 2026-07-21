# Web Upload - Clinical Anomaly Detection

Subprojeto Flask + Jinja2 para subir um arquivo ZIP e processar arquivos internos usando LangGraph.

## Como executar

Instale as dependencias a partir da raiz do projeto:

```powershell
.\env\Scripts\python.exe -m pip install -r web\requirements.txt
```

Execute o servidor:

```powershell
.\env\Scripts\python.exe web\app.py
```

Acesse:

```txt
http://localhost:8000
```

## Formato esperado do Excel

O arquivo `.xlsx` ou `.xls` deve ter ate tres abas:

- `vitals`
- `prescriptions`
- `transfers`

### Aba vitals

```txt
charttime, heart_rate, spo2, systolic_bp, diastolic_bp, mean_bp
```

Significado das colunas:

- `charttime`: data e hora em que os sinais vitais foram medidos.
- `heart_rate`: frequencia cardiaca do paciente, em batimentos por minuto.
- `spo2`: saturacao de oxigenio no sangue, em percentual.
- `systolic_bp`: pressao arterial sistolica, em mmHg.
- `diastolic_bp`: pressao arterial diastolica, em mmHg.
- `mean_bp`: pressao arterial media, em mmHg.

### Aba prescriptions

```txt
starttime, stoptime, drug, dose_val_rx, dose_unit_rx, route
```

Significado das colunas:

- `starttime`: data e hora de inicio da prescricao.
- `stoptime`: data e hora de fim ou suspensao da prescricao.
- `drug`: nome do medicamento prescrito.
- `dose_val_rx`: valor da dose prescrita.
- `dose_unit_rx`: unidade da dose prescrita, como `mg`, `g`, `mL` ou `mcg/min`.
- `route`: via de administracao do medicamento, como `PO` oral, `IV` intravenosa ou `IM` intramuscular.

### Aba transfers

```txt
eventtype, careunit, intime, outtime
```

Significado das colunas:

- `eventtype`: tipo de evento de movimentacao, como `admit`, `transfer` ou `discharge`.
- `careunit`: unidade/setor onde o paciente ficou, como `Medicine`, `Emergency Department` ou `Medical Intensive Care Unit`.
- `intime`: data e hora de entrada do paciente na unidade.
- `outtime`: data e hora de saida do paciente da unidade.

Arquivos de audio e video ainda retornam apenas mensagens de pendencia.

## Fluxo

O processamento e orquestrado com LangGraph:

```txt
extract -> classify -> process
```

## Estrutura

```txt
web/
    app.py
    flow.py
    zip_processor.py
    xls_converter.py
    templates/
        base.html
        index.html
        result.html
        error.html
    static/
        style.css
```
