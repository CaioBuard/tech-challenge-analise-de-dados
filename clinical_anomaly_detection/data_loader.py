"""Funcoes de leitura dos arquivos do MIMIC-IV demo."""

from pathlib import Path

import pandas as pd


class MimicDataLoader:
    """Carrega tabelas especificas do dataset MIMIC-IV demo."""

    def __init__(self, dataset_path):
        # Converte o caminho recebido para Path, que facilita montar subpastas.
        self.dataset_path = Path(dataset_path)

    def _read_csv_gz(self, relative_path, usecols=None):
        """Le um arquivo csv.gz relativo ao caminho raiz do dataset."""
        # Monta o caminho completo do arquivo solicitado.
        file_path = self.dataset_path / relative_path
        # Verifica se o arquivo existe antes de tentar ler.
        if not file_path.exists():
            # Erro claro para ajudar quem estiver executando o projeto.
            raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")
        # Le o arquivo compactado; o pandas entende .gz automaticamente.
        return pd.read_csv(file_path, compression="gzip", usecols=usecols)

    def load_chartevents(self):
        """Carrega eventos de sinais vitais da UTI."""
        # Le apenas as colunas necessarias para economizar memoria.
        return self._read_csv_gz(
            Path("icu") / "chartevents.csv.gz",
            usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        )

    def load_prescriptions(self):
        """Carrega prescricoes hospitalares."""
        # Le as colunas usadas para criar features de evolucao terapeutica.
        return self._read_csv_gz(
            Path("hosp") / "prescriptions.csv.gz",
            usecols=[
                "subject_id",
                "hadm_id",
                "starttime",
                "stoptime",
                "drug",
                "dose_val_rx",
                "dose_unit_rx",
                "route",
            ],
        )

    def load_transfers(self):
        """Carrega movimentacoes do paciente durante a internacao."""
        # Le transferencias entre unidades e seus horarios.
        return self._read_csv_gz(
            Path("hosp") / "transfers.csv.gz",
            usecols=["subject_id", "hadm_id", "eventtype", "careunit", "intime", "outtime"],
        )
