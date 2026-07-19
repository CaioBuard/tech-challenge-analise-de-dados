"""Fachada principal importavel por outros codigos."""

from pathlib import Path

from .alerts import (
    generate_model_alert,
    generate_prescription_rule_alerts,
    generate_transfer_rule_alerts,
    generate_vital_rule_alerts,
)
from .config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_DIR,
    PRESCRIPTION_MODEL_FILE,
    TRANSFER_MODEL_FILE,
    VITAL_MODEL_FILE,
)
from .data_loader import MimicDataLoader
from .features import (
    build_prescription_patient_features,
    build_prescription_training_features,
    build_transfer_patient_features,
    build_transfer_training_features,
    build_vital_patient_features,
    build_vital_training_features,
)
from .models import AnomalyModelManager
from .visualizations import generate_visual_report


class ClinicalAnomalyDetector:
    """Servico principal para treino, carga e predicao de anomalias clinicas."""

    def __init__(self, dataset_path=DEFAULT_DATASET_PATH, model_dir=DEFAULT_MODEL_DIR):
        # Guarda o caminho do dataset como Path.
        self.dataset_path = Path(dataset_path)
        # Guarda o caminho dos modelos como Path.
        self.model_dir = Path(model_dir)
        # Cria o carregador de dados do MIMIC-IV demo.
        self.loader = MimicDataLoader(self.dataset_path)
        # Cria o gerenciador de modelos persistidos.
        self.model_manager = AnomalyModelManager(self.model_dir)
        # Define os nomes logicos e arquivos de cada modelo.
        self.model_files = {
            "vital_signs": VITAL_MODEL_FILE,
            "prescriptions": PRESCRIPTION_MODEL_FILE,
            "transfers": TRANSFER_MODEL_FILE,
        }
        # Indica se os modelos ja estao carregados em memoria.
        self.loaded = False

    def train(self):
        """Forca novo treinamento e sobrescreve modelos salvos."""
        # Carrega eventos de sinais vitais.
        chartevents = self.loader.load_chartevents()
        # Carrega prescricoes.
        prescriptions = self.loader.load_prescriptions()
        # Carrega transferencias.
        transfers = self.loader.load_transfers()
        # Cria features de sinais vitais.
        vital_features = build_vital_training_features(chartevents)
        # Cria features de prescricoes.
        prescription_features = build_prescription_training_features(prescriptions)
        # Cria features de movimentacao.
        transfer_features = build_transfer_training_features(transfers)
        # Junta os tres conjuntos de treino em um dicionario.
        training_sets = {
            "vital_signs": vital_features,
            "prescriptions": prescription_features,
            "transfers": transfer_features,
        }
        # Treina e salva os modelos.
        status = self.model_manager.train_and_save(training_sets, self.model_files)
        # Marca que os modelos ficaram carregados apos o treino.
        self.loaded = True
        # Retorna status para quem chamou.
        return status

    def train_if_needed(self):
        """Treina somente quando os modelos ainda nao existem no disco."""
        # Verifica se todos os modelos ja foram salvos.
        if self.model_manager.all_models_exist(self.model_files):
            # Tenta carregar os modelos existentes para reutilizar.
            try:
                # Carrega os modelos existentes para reutilizar.
                status = self.load_models()
                # Retorna status indicando reutilizacao.
                return status
            # Captura incompatibilidade de versao ou artefato antigo do scikit-learn.
            except (AttributeError, ValueError, ModuleNotFoundError):
                # Retreina os modelos no ambiente atual quando os arquivos antigos falham.
                return self.train()
        # Se algum arquivo estiver faltando, treina tudo novamente.
        return self.train()

    def load_models(self):
        """Carrega modelos ja treinados do disco."""
        # Pede ao gerenciador para carregar todos os modelos esperados.
        status = self.model_manager.load(self.model_files)
        # Marca que os modelos estao prontos em memoria.
        self.loaded = True
        # Retorna status para mensagens externas.
        return status

    def predict_patient(self, patient_input):
        """Recebe dados de um paciente e retorna alertas de anomalia."""
        # Carrega modelos automaticamente caso ainda nao estejam em memoria.
        if not self.loaded:
            # Treina apenas se necessario; caso contrario, reutiliza modelos salvos.
            self.train_if_needed()
        # Busca sinais vitais do input; se nao existir, usa lista vazia.
        vitals = patient_input.get("vitals", [])
        # Busca prescricoes do input; se nao existir, usa lista vazia.
        prescriptions = patient_input.get("prescriptions", [])
        # Busca movimentacoes do input; se nao existir, usa lista vazia.
        transfers = patient_input.get("transfers", [])
        # Lista final de alertas.
        alerts = []
        # Adiciona alertas por regras clinicas de sinais vitais.
        alerts.extend(generate_vital_rule_alerts(vitals))
        # Adiciona alertas por regras de prescricoes.
        alerts.extend(generate_prescription_rule_alerts(prescriptions))
        # Adiciona alertas por regras de movimentacao.
        alerts.extend(generate_transfer_rule_alerts(transfers))
        # Cria features de sinais vitais para o paciente.
        vital_features = build_vital_patient_features(vitals)
        # Cria features de prescricoes para o paciente.
        prescription_features = build_prescription_patient_features(prescriptions)
        # Cria features de movimentacao para o paciente.
        transfer_features = build_transfer_patient_features(transfers)
        # Junta as features por nome de modelo para executar de forma padronizada.
        feature_sets = {
            "vital_signs": vital_features,
            "prescriptions": prescription_features,
            "transfers": transfer_features,
        }
        # Tenta predizer com modelos carregados.
        try:
            # Aplica os tres modelos e adiciona os alertas anomalos.
            alerts.extend(self._predict_all_models(feature_sets))
        # Captura falhas tipicas de modelo salvo em versao diferente do scikit-learn.
        except (AttributeError, ValueError, ModuleNotFoundError):
            # Retreina os modelos no ambiente atual.
            self.train()
            # Repete a predicao com os modelos recem-treinados.
            alerts.extend(self._predict_all_models(feature_sets))
        # Retorna a lista consolidada para outro codigo consumir.
        return alerts

    def generate_patient_report(self, patient_input, output_dir="reports"):
        """Gera alertas e imagens explicativas para um paciente."""
        # Carrega modelos automaticamente caso ainda nao estejam em memoria.
        if not self.loaded:
            # Treina apenas se necessario; caso contrario, reutiliza modelos salvos.
            self.train_if_needed()
        # Gera os alertas usando o fluxo principal da biblioteca.
        alerts = self.predict_patient(patient_input)
        # Cria features de treino de sinais vitais para usar como referencia visual.
        reference_features = build_vital_training_features(self.loader.load_chartevents())
        # Cria features de sinais vitais para o paciente informado.
        vital_features = build_vital_patient_features(patient_input.get("vitals", []))
        # Cria features de prescricoes para o paciente informado.
        prescription_features = build_prescription_patient_features(patient_input.get("prescriptions", []))
        # Cria features de movimentacao para o paciente informado.
        transfer_features = build_transfer_patient_features(patient_input.get("transfers", []))
        # Junta as features do paciente por categoria.
        feature_sets = {
            "vital_signs": vital_features,
            "prescriptions": prescription_features,
            "transfers": transfer_features,
        }
        # Calcula score medio por categoria para o grafico.
        scores_by_category = self._score_all_models(feature_sets)
        # Gera as imagens explicativas em disco.
        images = generate_visual_report(
            patient_input=patient_input,
            reference_features=reference_features,
            patient_features=vital_features,
            scores_by_category=scores_by_category,
            output_dir=output_dir,
        )
        # Retorna alertas e imagens para outro codigo usar.
        return {"alerts": alerts, "images": images, "scores": scores_by_category}

    def _score_all_models(self, feature_sets):
        """Calcula score medio de cada modelo para graficos explicativos."""
        # Dicionario onde cada categoria recebe seu score medio.
        scores_by_category = {}
        # Percorre cada modelo e suas features.
        for model_name, features in feature_sets.items():
            # Se nao houver features, registra score ausente.
            if features.empty:
                # Guarda None para indicar que nao houve dados naquela categoria.
                scores_by_category[model_name] = None
                # Pula para o proximo modelo.
                continue
            # Executa o modelo para obter scores.
            _, scores = self.model_manager.predict(model_name, features)
            # Calcula a media dos scores, pois um paciente pode ter varias linhas.
            scores_by_category[model_name] = float(scores.mean()) if len(scores) else None
        # Retorna scores prontos para visualizacao.
        return scores_by_category

    def _predict_all_models(self, feature_sets):
        """Executa todos os modelos e retorna alertas gerados por eles."""
        # Lista temporaria para alertas de modelos.
        alerts = []
        # Percorre cada modelo e suas respectivas features.
        for model_name, features in feature_sets.items():
            # Executa o modelo atual e acumula os alertas.
            alerts.extend(self._predict_and_alert(model_name, features))
        # Retorna todos os alertas de modelo.
        return alerts

    def _predict_and_alert(self, model_name, features):
        """Executa um modelo e transforma predicoes anomalas em alertas."""
        # Lista temporaria de alertas deste modelo.
        alerts = []
        # Executa predicao e score.
        predictions, scores = self.model_manager.predict(model_name, features)
        # Percorre cada linha avaliada pelo modelo.
        for prediction, score in zip(predictions, scores):
            # Converte anomalia de modelo em alerta textual.
            alert = generate_model_alert(model_name, prediction, score)
            # Adiciona somente quando o modelo marcou anomalia.
            if alert is not None:
                alerts.append(alert)
        # Retorna alertas deste modelo.
        return alerts
