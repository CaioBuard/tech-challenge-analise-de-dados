"""Geracao de alertas automaticos para equipe medica."""


def _alert(category, severity, title, message, score=None):
    """Cria um dicionario padronizado de alerta."""
    # Retorna a estrutura acordada para consumo por outros sistemas.
    return {
        "category": category,
        "severity": severity,
        "title": title,
        "message": message,
        "score": None if score is None else float(score),
    }


def generate_vital_rule_alerts(vitals):
    """Gera alertas por regras clinicas simples em sinais vitais."""
    # Lista onde os alertas encontrados serao acumulados.
    alerts = []
    # Percorre cada medida de sinais vitais recebida.
    for vital in vitals:
        # Le frequencia cardiaca do input.
        heart_rate = vital.get("heart_rate")
        # Le saturacao de oxigenio do input.
        spo2 = vital.get("spo2")
        # Le pressao sistolica do input.
        systolic_bp = vital.get("systolic_bp")
        # Le pressao diastolica do input.
        diastolic_bp = vital.get("diastolic_bp")
        # Alerta de frequencia cardiaca muito alta ou muito baixa.
        if heart_rate is not None and (heart_rate > 120 or heart_rate < 45):
            alerts.append(_alert("vital_signs", "high", "Anomalia em frequencia cardiaca", f"Frequencia cardiaca fora do esperado: {heart_rate} bpm."))
        # Alerta de baixa oxigenacao.
        if spo2 is not None and spo2 < 90:
            alerts.append(_alert("vital_signs", "high", "Anomalia em oxigenacao", f"Saturacao de oxigenio baixa: {spo2}%."))
        # Alerta de pressao sistolica muito baixa ou muito alta.
        if systolic_bp is not None and (systolic_bp < 90 or systolic_bp > 180):
            alerts.append(_alert("vital_signs", "medium", "Anomalia em pressao arterial", f"Pressao sistolica fora do esperado: {systolic_bp} mmHg."))
        # Alerta de pressao diastolica muito baixa ou muito alta.
        if diastolic_bp is not None and (diastolic_bp < 50 or diastolic_bp > 120):
            alerts.append(_alert("vital_signs", "medium", "Anomalia em pressao arterial", f"Pressao diastolica fora do esperado: {diastolic_bp} mmHg."))
    # Retorna todos os alertas de regras clinicas.
    return alerts


def generate_prescription_rule_alerts(prescriptions):
    """Gera alertas por regras simples em prescricoes."""
    # Lista onde os alertas encontrados serao acumulados.
    alerts = []
    # Conta quantas prescricoes foram enviadas para o paciente.
    total = len(prescriptions)
    # Alerta para muitas prescricoes simultaneas no exemplo recebido.
    if total >= 8:
        alerts.append(_alert("prescriptions", "medium", "Volume incomum de prescricoes", f"Paciente possui {total} prescricoes no periodo informado."))
    # Lista temporaria para medicamentos prescritos por via IV.
    iv_drugs = []
    # Percorre prescricoes para procurar rotas de maior criticidade.
    for prescription in prescriptions:
        # Le a rota de administracao em maiusculo.
        route = str(prescription.get("route", "")).upper()
        # Le o medicamento para montar mensagem explicativa.
        drug = prescription.get("drug", "medicamento nao informado")
        # Guarda medicamentos intravenosos para gerar um unico alerta resumido.
        if "IV" in route:
            iv_drugs.append(str(drug))
    # Gera um alerta resumido se houver prescricoes intravenosas.
    if iv_drugs:
        # Seleciona alguns nomes para a mensagem nao ficar longa.
        examples = ", ".join(sorted(set(iv_drugs))[:5])
        # Adiciona indicacao de que existem outros medicamentos alem dos exemplos.
        suffix = "..." if len(set(iv_drugs)) > 5 else ""
        # Adiciona alerta unico de complexidade terapeutica por via IV.
        alerts.append(_alert("prescriptions", "low", "Prescricoes intravenosas", f"Paciente possui {len(iv_drugs)} prescricao(oes) IV. Exemplos: {examples}{suffix}."))
    # Retorna alertas gerados por regra.
    return alerts


def generate_transfer_rule_alerts(transfers):
    """Gera alertas por regras simples em movimentacao do paciente."""
    # Lista onde os alertas encontrados serao acumulados.
    alerts = []
    # Conta transferencias recebidas no input.
    total = len(transfers)
    # Alerta para muitas movimentacoes na internacao.
    if total >= 4:
        alerts.append(_alert("transfers", "medium", "Movimentacao frequente", f"Paciente possui {total} movimentacoes registradas."))
    # Conta passagens por UTI.
    icu_count = sum(1 for transfer in transfers if "ICU" in str(transfer.get("careunit", "")).upper() or "INTENSIVE" in str(transfer.get("careunit", "")).upper())
    # Alerta quando existe passagem por UTI.
    if icu_count > 0:
        alerts.append(_alert("transfers", "medium", "Passagem por unidade critica", f"Paciente possui {icu_count} movimentacao(oes) envolvendo UTI ou unidade intensiva."))
    # Retorna os alertas de movimentacao.
    return alerts


def generate_model_alert(category, prediction, score):
    """Gera alerta quando o modelo considera a entrada anomala."""
    # Se a predicao for normal, nao ha alerta.
    if prediction != -1:
        return None
    # Define severidade alta para scores bem negativos.
    severity = "high" if score < -0.05 else "medium"
    # Define titulos por categoria.
    titles = {
        "vital_signs": "Anomalia detectada pelo modelo em sinais vitais",
        "prescriptions": "Anomalia detectada pelo modelo em prescricoes",
        "transfers": "Anomalia detectada pelo modelo em movimentacao",
    }
    # Define mensagens por categoria.
    messages = {
        "vital_signs": "O padrao de sinais vitais difere do comportamento aprendido.",
        "prescriptions": "A evolucao das prescricoes difere do comportamento aprendido.",
        "transfers": "O padrao de movimentacao difere do comportamento aprendido.",
    }
    # Retorna o alerta padronizado.
    return _alert(category, severity, titles[category], messages[category], score)
