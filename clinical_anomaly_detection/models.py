"""Treino, persistencia e predicao dos modelos de anomalia."""

from pathlib import Path
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_pipeline(random_state=42):
    """Cria o pipeline padrao de deteccao de anomalias."""
    # O imputador preenche valores faltantes usando a mediana da coluna.
    imputer = SimpleImputer(strategy="median")
    # O scaler padroniza as features para media 0 e desvio 1.
    scaler = StandardScaler()
    # O IsolationForest aprende o comportamento comum sem precisar de rotulos.
    model = IsolationForest(n_estimators=200, contamination=0.08, random_state=random_state)
    # O pipeline garante que treino e predicao usem os mesmos passos.
    return Pipeline([("imputer", imputer), ("scaler", scaler), ("model", model)])


class AnomalyModelManager:
    """Gerencia modelos treinados, salvos e carregados."""

    def __init__(self, model_dir):
        # Guarda a pasta onde os modelos serao persistidos.
        self.model_dir = Path(model_dir)
        # Garante que a pasta exista.
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Dicionario em memoria com os modelos carregados.
        self.models = {}

    def model_path(self, model_name):
        """Retorna o caminho completo de um arquivo de modelo."""
        # Junta a pasta de modelos com o nome do arquivo.
        return self.model_dir / model_name

    def all_models_exist(self, model_files):
        """Verifica se todos os arquivos de modelo ja existem."""
        # Retorna True somente se cada arquivo estiver presente no disco.
        return all(self.model_path(model_file).exists() for model_file in model_files.values())

    def train_and_save(self, training_sets, model_files):
        """Treina os modelos e salva cada um em disco."""
        # Limpa modelos antigos carregados em memoria.
        self.models = {}
        # Percorre cada conjunto de treino recebido.
        for name, features in training_sets.items():
            # Valida se existem linhas suficientes para treinar.
            if features.empty:
                # Interrompe com mensagem clara se o dataset nao tiver dados.
                raise ValueError(f"Nao ha dados de treino para o modelo: {name}")
            # Cria um pipeline novo para este tipo de dado.
            pipeline = _build_pipeline()
            # Treina o modelo com as features numericas.
            pipeline.fit(features)
            # Guarda o modelo em memoria.
            self.models[name] = pipeline
            # Cria um pacote com o modelo e a versao do scikit-learn usada no treino.
            artifact = {"sklearn_version": sklearn.__version__, "pipeline": pipeline}
            # Salva o modelo em disco para reutilizacao futura.
            joblib.dump(artifact, self.model_path(model_files[name]))
        # Retorna um texto simples para o script informar o usuario.
        return "trained"

    def load(self, model_files):
        """Carrega modelos ja treinados do disco."""
        # Limpa o dicionario para evitar misturar modelos antigos.
        self.models = {}
        # Percorre cada arquivo esperado.
        for name, model_file in model_files.items():
            # Monta o caminho completo do arquivo.
            path = self.model_path(model_file)
            # Verifica se o modelo existe antes de carregar.
            if not path.exists():
                # Erro claro quando predict for chamado antes de treinar.
                raise FileNotFoundError(f"Modelo nao encontrado: {path}")
            # Trata aviso de versao inconsistente como erro para poder retreinar.
            try:
                # Cria um contexto temporario para controlar warnings.
                with warnings.catch_warnings():
                    # Faz o scikit-learn avisar como excecao quando o joblib for de outra versao.
                    warnings.simplefilter("error", InconsistentVersionWarning)
                    # Carrega o artefato salvo com joblib.
                    artifact = joblib.load(path)
            # Captura o warning convertido em excecao.
            except InconsistentVersionWarning as exc:
                # Converte para ValueError, que a camada de servico usa para retreinar.
                raise ValueError(f"Modelo incompativel com a versao atual do scikit-learn: {path}") from exc
            # Modelos novos sao salvos como dicionario com metadados.
            if isinstance(artifact, dict) and "pipeline" in artifact:
                # Le a versao usada no momento do treino.
                trained_version = artifact.get("sklearn_version")
                # Compara a versao treinada com a versao instalada agora.
                if trained_version != sklearn.__version__:
                    # Interrompe a carga para que a camada de servico retreine.
                    raise ValueError(f"Modelo {path} foi treinado com scikit-learn {trained_version}, mas o ambiente usa {sklearn.__version__}.")
                # Guarda apenas o pipeline em memoria.
                self.models[name] = artifact["pipeline"]
            else:
                # Mantem compatibilidade com modelos antigos salvos antes dos metadados.
                self.models[name] = artifact
        # Retorna um texto simples para o script informar reutilizacao.
        return "loaded"

    def predict(self, model_name, features):
        """Prediz anomalias e scores para um conjunto de features."""
        # Verifica se o modelo solicitado esta carregado.
        if model_name not in self.models:
            # Erro claro para facilitar depuracao.
            raise ValueError(f"Modelo nao carregado: {model_name}")
        # Se nao houver linhas, retorna arrays vazios.
        if features.empty:
            # Retorna predicoes e scores vazios para manter contrato da funcao.
            return np.array([]), np.array([])
        # Busca o pipeline em memoria.
        model = self.models[model_name]
        # Predicao retorna -1 para anomalia e 1 para normal.
        predictions = model.predict(features)
        # decision_function retorna scores menores para pontos mais anomalos.
        scores = model.decision_function(features)
        # Entrega predicoes e scores para a camada de alertas.
        return predictions, scores
