"""Arquivo de teste manual da biblioteca de deteccao de anomalias."""

from datetime import datetime, timedelta

from clinical_anomaly_detection import ClinicalAnomalyDetector


def build_extreme_prescriptions():
    """Monta prescricoes extremas para forcar anomalia no modelo de prescricoes."""
    # Define a data/hora inicial das prescricoes sinteticas.
    base_time = datetime(2150, 1, 1, 8, 0, 0)
    # Lista onde as prescricoes extremas serao acumuladas.
    prescriptions = []
    # Cria muitas prescricoes para sair da distribuicao aprendida no MIMIC-IV demo.
    for index in range(120):
        # Calcula o horario de inicio da prescricao atual.
        start_time = base_time + timedelta(minutes=index)
        # Calcula o horario de fim da prescricao atual.
        stop_time = start_time + timedelta(minutes=30)
        # Adiciona uma prescricao com medicamento variado, dose extrema e via IV.
        prescriptions.append(
            {
                "starttime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "stoptime": stop_time.strftime("%Y-%m-%d %H:%M:%S"),
                "drug": f"Extreme Drug {index % 80}",
                "dose_val_rx": str(20000 + index * 100),
                "dose_unit_rx": "mg",
                "route": "IV",
            }
        )
    # Retorna a lista completa para uso no paciente anomalo.
    return prescriptions


def build_anomalous_patient():
    """Monta um paciente ficticio com sinais de anomalia para demonstracao."""
    # Retorna o dicionario no formato esperado pela biblioteca.
    return {
        # Lista de sinais vitais em ordem temporal.
        "vitals": [
            # Primeira medida com valores relativamente comuns.
            {
                "charttime": "2150-01-01 08:00:00",
                "heart_rate": 82,
                "spo2": 96,
                "systolic_bp": 120,
                "diastolic_bp": 75,
                "mean_bp": 90,
            },
            # Segunda medida com taquicardia, baixa saturacao e hipotensao.
            {
                "charttime": "2150-01-01 10:00:00",
                "heart_rate": 132,
                "spo2": 86,
                "systolic_bp": 82,
                "diastolic_bp": 45,
                "mean_bp": 58,
            },
        ],
        # Lista extrema de prescricoes para diferenciar tambem no modelo.
        "prescriptions": build_extreme_prescriptions(),
        # Lista de movimentacoes durante a internacao.
        "transfers": [
            # Entrada inicial em unidade comum.
            {
                "eventtype": "admit",
                "careunit": "Medicine",
                "intime": "2150-01-01 06:00:00",
                "outtime": "2150-01-01 08:00:00",
            },
            # Transferencia rapida para UTI.
            {
                "eventtype": "transfer",
                "careunit": "Medical Intensive Care Unit",
                "intime": "2150-01-01 08:00:00",
                "outtime": "2150-01-01 11:00:00",
            },
            # Saida para outra unidade apos poucas horas.
            {
                "eventtype": "transfer",
                "careunit": "Stepdown Unit",
                "intime": "2150-01-01 11:00:00",
                "outtime": "2150-01-01 14:00:00",
            },
            # Retorno para unidade intensiva.
            {
                "eventtype": "transfer",
                "careunit": "Surgical Intensive Care Unit",
                "intime": "2150-01-01 14:00:00",
                "outtime": "2150-01-01 18:00:00",
            },
        ],
    }


def build_normal_patient():
    """Monta um paciente ficticio com padrao esperado para comparacao."""
    # Retorna um dicionario com valores clinicos mais estaveis.
    return {
        # Lista de sinais vitais com valores dentro de faixas usuais.
        "vitals": [
            # Primeira medida do paciente estavel.
            {
                "charttime": "2150-01-01 08:00:00",
                "heart_rate": 78,
                "spo2": 97,
                "systolic_bp": 118,
                "diastolic_bp": 72,
                "mean_bp": 88,
            },
            # Segunda medida com pouca variacao em relacao a anterior.
            {
                "charttime": "2150-01-01 10:00:00",
                "heart_rate": 80,
                "spo2": 96,
                "systolic_bp": 122,
                "diastolic_bp": 74,
                "mean_bp": 90,
            },
        ],
        # Lista curta de prescricoes sem mudancas grandes.
        "prescriptions": [
            # Prescricao oral simples.
            {
                "starttime": "2150-01-01 08:00:00",
                "stoptime": "2150-01-02 08:00:00",
                "drug": "Example Analgesic",
                "dose_val_rx": "500",
                "dose_unit_rx": "mg",
                "route": "PO",
            }
        ],
        # Movimentacao simples, sem passagem por UTI.
        "transfers": [
            # Permanencia em unidade comum por periodo mais longo.
            {
                "eventtype": "admit",
                "careunit": "Medicine",
                "intime": "2150-01-01 06:00:00",
                "outtime": "2150-01-03 06:00:00",
            }
        ],
    }


def print_alerts(case_name, alerts):
    """Imprime os alertas de um caso de teste em formato legivel."""
    # Imprime titulo do caso avaliado.
    print(f"\nAlertas gerados para {case_name}:")
    # Caso nao existam alertas, informa explicitamente.
    if not alerts:
        print("- Nenhum alerta gerado.")
    # Percorre os alertas retornados.
    for alert in alerts:
        # Imprime cada alerta em formato legivel.
        print(f"- [{alert['severity']}] {alert['category']}: {alert['title']} | {alert['message']} | score={alert['score']}")


def main():
    """Executa treino ou carga dos modelos e imprime alertas do paciente exemplo."""
    # Cria o detector apontando para o dataset e para a pasta de modelos.
    detector = ClinicalAnomalyDetector(
        dataset_path="dataset/mimic-iv-clinical-database-demo-2.2",
        model_dir="models",
    )
    # Treina apenas na primeira execucao; depois reutiliza modelos salvos.
    status = detector.train_if_needed()
    # Mostra se os modelos foram treinados ou carregados.
    print(f"Status dos modelos: {status}")
    # Monta um paciente ficticio sem anomalias evidentes.
    normal_patient = build_normal_patient()
    # Executa a deteccao para o paciente sem anomalia.
    normal_alerts = detector.predict_patient(normal_patient)
    # Imprime os resultados do paciente sem anomalia.
    print_alerts("paciente sem anomalia", normal_alerts)
    # Gera imagens explicativas do paciente sem anomalia.
    normal_report = detector.generate_patient_report(normal_patient, output_dir="reports/normal")
    # Imprime os caminhos das imagens geradas.
    print("\nImagens do paciente sem anomalia:")
    # Percorre cada imagem criada no relatorio.
    for image in normal_report["images"]:
        # Mostra nome logico e caminho do PNG.
        print(f"- {image['name']}: {image['path']}")
    # Monta um paciente ficticio com anomalias evidentes.
    anomalous_patient = build_anomalous_patient()
    # Executa a deteccao para o paciente com anomalia.
    anomalous_alerts = detector.predict_patient(anomalous_patient)
    # Imprime os resultados do paciente com anomalia.
    print_alerts("paciente com anomalia", anomalous_alerts)
    # Gera imagens explicativas do paciente com anomalia.
    anomalous_report = detector.generate_patient_report(anomalous_patient, output_dir="reports/anomalous")
    # Imprime os caminhos das imagens geradas.
    print("\nImagens do paciente com anomalia:")
    # Percorre cada imagem criada no relatorio.
    for image in anomalous_report["images"]:
        # Mostra nome logico e caminho do PNG.
        print(f"- {image['name']}: {image['path']}")


# Garante que o main rode somente quando este arquivo for executado diretamente.
if __name__ == "__main__":
    # Chama a funcao principal de teste.
    main()
