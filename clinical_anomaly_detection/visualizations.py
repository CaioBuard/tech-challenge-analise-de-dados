"""Geracao de graficos para explicar as anomalias detectadas."""

from pathlib import Path

import matplotlib

# Usa backend sem tela, adequado para biblioteca e execucao em servidor.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _image_result(name, path):
    """Monta o retorno padrao de uma imagem gerada."""
    # Retorna nome logico e caminho para o codigo chamador usar a imagem.
    return {"name": name, "path": str(path)}


def _prepare_output_dir(output_dir):
    """Garante que a pasta de saida exista."""
    # Converte o caminho recebido para Path.
    path = Path(output_dir)
    # Cria a pasta, incluindo pais, caso ainda nao exista.
    path.mkdir(parents=True, exist_ok=True)
    # Retorna a pasta pronta para uso.
    return path


def plot_vital_timeseries(patient_input, output_dir):
    """Gera grafico temporal dos sinais vitais do paciente."""
    # Prepara a pasta de saida.
    output_path = _prepare_output_dir(output_dir)
    # Converte a lista de sinais vitais em dataframe.
    vitals = pd.DataFrame(patient_input.get("vitals", []))
    # Define o caminho final da imagem.
    image_path = output_path / "vital_timeseries.png"
    # Cria a figura do matplotlib.
    plt.figure(figsize=(11, 5))
    # Verifica se existem dados para desenhar.
    if vitals.empty:
        # Escreve mensagem quando nao ha sinais vitais.
        plt.text(0.5, 0.5, "Sem sinais vitais informados", ha="center", va="center")
    else:
        # Converte o horario para datetime.
        vitals["charttime"] = pd.to_datetime(vitals.get("charttime"), errors="coerce")
        # Ordena os pontos no tempo.
        vitals = vitals.sort_values("charttime")
        # Desenha frequencia cardiaca quando existir.
        if "heart_rate" in vitals:
            plt.plot(vitals["charttime"], vitals["heart_rate"], marker="o", label="Heart rate (bpm)", color="#d62728")
        # Desenha saturacao quando existir.
        if "spo2" in vitals:
            plt.plot(vitals["charttime"], vitals["spo2"], marker="o", label="SpO2 (%)", color="#1f77b4")
        # Desenha pressao sistolica quando existir.
        if "systolic_bp" in vitals:
            plt.plot(vitals["charttime"], vitals["systolic_bp"], marker="o", label="Systolic BP (mmHg)", color="#2ca02c")
        # Destaca pontos clinicamente criticos em vermelho escuro.
        for _, row in vitals.iterrows():
            # Marca frequencia cardiaca fora da regra clinica.
            if pd.notna(row.get("heart_rate")) and (row.get("heart_rate") > 120 or row.get("heart_rate") < 45):
                plt.scatter(row["charttime"], row["heart_rate"], s=120, color="#8b0000", zorder=5)
            # Marca saturacao baixa.
            if pd.notna(row.get("spo2")) and row.get("spo2") < 90:
                plt.scatter(row["charttime"], row["spo2"], s=120, color="#8b0000", zorder=5)
            # Marca pressao sistolica fora da regra clinica.
            if pd.notna(row.get("systolic_bp")) and (row.get("systolic_bp") < 90 or row.get("systolic_bp") > 180):
                plt.scatter(row["charttime"], row["systolic_bp"], s=120, color="#8b0000", zorder=5)
        # Mostra legenda dos sinais.
        plt.legend()
    # Define titulo do grafico.
    plt.title("Sinais vitais do paciente")
    # Define rotulo do eixo X.
    plt.xlabel("Tempo")
    # Define rotulo do eixo Y.
    plt.ylabel("Valor observado")
    # Adiciona grade discreta.
    plt.grid(True, alpha=0.25)
    # Ajusta margens para evitar cortes.
    plt.tight_layout()
    # Salva a imagem em PNG.
    plt.savefig(image_path, dpi=140)
    # Fecha a figura para liberar memoria.
    plt.close()
    # Retorna o caminho da imagem gerada.
    return _image_result("vital_timeseries", image_path)


def plot_feature_comparison(reference_features, patient_features, output_dir):
    """Gera grafico comparando paciente com distribuicao do dataset."""
    # Prepara a pasta de saida.
    output_path = _prepare_output_dir(output_dir)
    # Define o caminho final da imagem.
    image_path = output_path / "feature_comparison.png"
    # Escolhe features clinicas faceis de explicar.
    columns = ["heart_rate", "spo2", "systolic_bp", "diastolic_bp", "mean_bp"]
    # Mantem somente colunas disponiveis no dataset de referencia.
    columns = [column for column in columns if column in reference_features.columns and column in patient_features.columns]
    # Cria a figura.
    plt.figure(figsize=(11, 5))
    # Verifica se ha dados suficientes.
    if not columns or reference_features.empty or patient_features.empty:
        # Escreve mensagem quando nao for possivel comparar.
        plt.text(0.5, 0.5, "Sem dados suficientes para comparacao", ha="center", va="center")
    else:
        # Cria lista de valores de referencia, removendo nulos.
        data = [reference_features[column].dropna() for column in columns]
        # Desenha boxplots do dataset.
        plt.boxplot(data, tick_labels=columns, vert=True, patch_artist=True)
        # Calcula uma linha media do paciente para cada feature.
        patient_values = [patient_features[column].mean() for column in columns]
        # Desenha os valores do paciente como pontos vermelhos.
        plt.scatter(range(1, len(columns) + 1), patient_values, color="#d62728", s=90, label="Paciente")
        # Adiciona legenda.
        plt.legend()
    # Define titulo.
    plt.title("Comparacao do paciente com o modelo")
    # Define rotulo do eixo Y.
    plt.ylabel("Valor da feature")
    # Adiciona grade discreta.
    plt.grid(True, axis="y", alpha=0.25)
    # Ajusta margens.
    plt.tight_layout()
    # Salva a imagem.
    plt.savefig(image_path, dpi=140)
    # Fecha a figura.
    plt.close()
    # Retorna o caminho da imagem gerada.
    return _image_result("feature_comparison", image_path)


def plot_anomaly_scores(scores_by_category, output_dir):
    """Gera grafico de barras com score de anomalia por categoria."""
    # Prepara a pasta de saida.
    output_path = _prepare_output_dir(output_dir)
    # Define o caminho final da imagem.
    image_path = output_path / "anomaly_scores.png"
    # Cria a figura.
    plt.figure(figsize=(9, 4.5))
    # Define categorias na ordem do trabalho.
    categories = ["vital_signs", "prescriptions", "transfers"]
    # Busca os scores na ordem definida.
    scores = [scores_by_category.get(category) for category in categories]
    # Define cores: vermelho para negativo, verde para positivo, cinza para ausente.
    colors = ["#d62728" if score is not None and score < 0 else "#2ca02c" if score is not None else "#999999" for score in scores]
    # Troca None por zero apenas para conseguir desenhar a barra.
    plot_scores = [0 if score is None else score for score in scores]
    # Desenha barras horizontais.
    plt.barh(categories, plot_scores, color=colors)
    # Desenha a linha vertical que separa normal de anomalo.
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    # Escreve o valor numerico ao lado de cada barra.
    for index, score in enumerate(scores):
        # Define texto exibido para score ausente.
        text = "sem score" if score is None else f"{score:.3f}"
        # Posiciona texto perto da barra.
        plt.text(plot_scores[index], index, f" {text}", va="center")
    # Define titulo.
    plt.title("Score de anomalia por categoria")
    # Define rotulo do eixo X.
    plt.xlabel("Score do Modelo")
    # Adiciona grade discreta.
    plt.grid(True, axis="x", alpha=0.25)
    # Ajusta margens.
    plt.tight_layout()
    # Salva a imagem.
    plt.savefig(image_path, dpi=140)
    # Fecha a figura.
    plt.close()
    # Retorna o caminho da imagem gerada.
    return _image_result("anomaly_scores", image_path)


def generate_visual_report(patient_input, reference_features, patient_features, scores_by_category, output_dir):
    """Gera todas as imagens explicativas de um paciente."""
    # Lista onde as imagens geradas serao acumuladas.
    images = []
    # Gera grafico temporal dos sinais vitais.
    images.append(plot_vital_timeseries(patient_input, output_dir))
    # Gera boxplot comparando paciente e dataset.
    images.append(plot_feature_comparison(reference_features, patient_features, output_dir))
    # Gera grafico dos scores por categoria.
    images.append(plot_anomaly_scores(scores_by_category, output_dir))
    # Retorna a lista de imagens para o codigo chamador.
    return images
