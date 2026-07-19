"""Criacao de features para treino e predicao."""

import numpy as np
import pandas as pd

from .config import (
    PRESCRIPTION_FEATURE_COLUMNS,
    TRANSFER_FEATURE_COLUMNS,
    VITAL_FEATURE_COLUMNS,
    VITAL_ITEMID_MAP,
)


def _to_numeric_series(values):
    """Converte valores textuais de dose para numeros quando possivel."""
    # Transforma a serie em texto para permitir extrair numeros de strings como "2-4".
    text_values = values.astype(str)
    # Extrai o primeiro numero encontrado em cada valor textual.
    extracted = text_values.str.extract(r"([-+]?\d*\.?\d+)")[0]
    # Converte o texto extraido para float; valores invalidos viram NaN.
    return pd.to_numeric(extracted, errors="coerce")


def _ensure_feature_columns(frame, columns):
    """Garante que todas as colunas esperadas existam e estejam na ordem correta."""
    # Trabalha em uma copia para nao alterar o dataframe original fora da funcao.
    result = frame.copy()
    # Percorre cada coluna esperada pelo modelo.
    for column in columns:
        # Cria a coluna faltante com NaN, para o imputador do modelo preencher depois.
        if column not in result.columns:
            result[column] = np.nan
    # Retorna apenas as colunas usadas no treinamento, na ordem correta.
    return result[columns]


def build_vital_training_features(chartevents):
    """Cria features de treino para sinais vitais a partir de chartevents."""
    # Mantem apenas os itemids que representam batimento, oxigenacao e pressao.
    filtered = chartevents[chartevents["itemid"].isin(VITAL_ITEMID_MAP)].copy()
    # Remove linhas sem valor numerico, pois elas nao ajudam o modelo.
    filtered = filtered.dropna(subset=["valuenum"])
    # Converte o itemid para o nome clinico usado pela biblioteca.
    filtered["metric"] = filtered["itemid"].map(VITAL_ITEMID_MAP)
    # Converte o horario para datetime, permitindo ordenar a serie temporal.
    filtered["charttime"] = pd.to_datetime(filtered["charttime"], errors="coerce")
    # Remove horarios invalidos.
    filtered = filtered.dropna(subset=["charttime"])
    # Cria uma tabela larga: uma linha por horario, uma coluna por sinal vital.
    wide = filtered.pivot_table(
        index=["subject_id", "hadm_id", "stay_id", "charttime"],
        columns="metric",
        values="valuenum",
        aggfunc="mean",
    ).reset_index()
    # Ordena por paciente/internacao/UTI/tempo para calcular variacoes.
    wide = wide.sort_values(["subject_id", "hadm_id", "stay_id", "charttime"])
    # Calcula variacao de frequencia cardiaca em relacao a medida anterior.
    wide["heart_rate_delta"] = wide.groupby(["subject_id", "hadm_id", "stay_id"])["heart_rate"].diff()
    # Calcula variacao de saturacao em relacao a medida anterior.
    wide["spo2_delta"] = wide.groupby(["subject_id", "hadm_id", "stay_id"])["spo2"].diff()
    # Calcula variacao de pressao sistolica em relacao a medida anterior.
    wide["systolic_bp_delta"] = wide.groupby(["subject_id", "hadm_id", "stay_id"])["systolic_bp"].diff()
    # Retorna somente as colunas numericas esperadas pelo modelo.
    return _ensure_feature_columns(wide, VITAL_FEATURE_COLUMNS)


def build_vital_patient_features(vitals):
    """Cria features de sinais vitais para um novo paciente recebido como input."""
    # Converte a lista de dicionarios em dataframe.
    frame = pd.DataFrame(vitals)
    # Se nao houver sinais vitais, retorna uma linha vazia com as colunas esperadas.
    if frame.empty:
        return pd.DataFrame(columns=VITAL_FEATURE_COLUMNS)
    # Converte o horario para datetime para preservar a ordem temporal.
    frame["charttime"] = pd.to_datetime(frame.get("charttime"), errors="coerce")
    # Ordena as medidas do paciente pelo tempo.
    frame = frame.sort_values("charttime")
    # Calcula variacao de frequencia cardiaca entre medidas consecutivas.
    frame["heart_rate_delta"] = frame.get("heart_rate", pd.Series(dtype=float)).diff()
    # Calcula variacao de saturacao entre medidas consecutivas.
    frame["spo2_delta"] = frame.get("spo2", pd.Series(dtype=float)).diff()
    # Calcula variacao de pressao sistolica entre medidas consecutivas.
    frame["systolic_bp_delta"] = frame.get("systolic_bp", pd.Series(dtype=float)).diff()
    # Retorna somente as colunas numericas esperadas pelo modelo.
    return _ensure_feature_columns(frame, VITAL_FEATURE_COLUMNS)


def build_prescription_training_features(prescriptions):
    """Cria features de treino para evolucao de prescricoes."""
    # Trabalha em copia para preservar o dataframe original.
    frame = prescriptions.copy()
    # Converte o inicio da prescricao para datetime.
    frame["starttime"] = pd.to_datetime(frame["starttime"], errors="coerce")
    # Remove prescricoes sem data inicial.
    frame = frame.dropna(subset=["starttime"])
    # Cria uma data sem hora para agregar prescricoes por dia de internacao.
    frame["date"] = frame["starttime"].dt.date
    # Extrai a dose numerica sempre que possivel.
    frame["dose_numeric"] = _to_numeric_series(frame["dose_val_rx"])
    # Marca prescricoes intravenosas.
    frame["is_iv"] = frame["route"].fillna("").str.upper().str.contains("IV", regex=False).astype(int)
    # Ordena para calcular mudancas sequenciais de dose e rota.
    frame = frame.sort_values(["subject_id", "hadm_id", "drug", "starttime"])
    # Calcula mudanca absoluta de dose dentro do mesmo medicamento.
    frame["dose_delta"] = frame.groupby(["subject_id", "hadm_id", "drug"])["dose_numeric"].diff().abs()
    # Marca mudancas relevantes de dose.
    frame["dose_changed"] = (frame["dose_delta"] > 0).astype(int)
    # Marca mudancas de rota dentro do mesmo medicamento.
    frame["route_changed"] = (
        frame.groupby(["subject_id", "hadm_id", "drug"])["route"].transform(lambda s: s.fillna("").ne(s.fillna("").shift()).astype(int))
        - 1
    ).clip(lower=0)
    # Agrega prescricoes por paciente, internacao e dia.
    grouped = frame.groupby(["subject_id", "hadm_id", "date"]).agg(
        total_prescriptions=("drug", "count"),
        unique_drugs=("drug", "nunique"),
        iv_prescriptions=("is_iv", "sum"),
        avg_dose=("dose_numeric", "mean"),
        max_dose=("dose_numeric", "max"),
        dose_change_count=("dose_changed", "sum"),
        route_change_count=("route_changed", "sum"),
    )
    # Remove o indice hierarquico criado pelo groupby.
    grouped = grouped.reset_index()
    # Retorna somente as colunas esperadas pelo modelo.
    return _ensure_feature_columns(grouped, PRESCRIPTION_FEATURE_COLUMNS)


def build_prescription_patient_features(prescriptions):
    """Cria features de prescricoes para um novo paciente recebido como input."""
    # Converte a lista de prescricoes em dataframe.
    frame = pd.DataFrame(prescriptions)
    # Se nao houver prescricoes, retorna dataframe vazio.
    if frame.empty:
        return pd.DataFrame(columns=PRESCRIPTION_FEATURE_COLUMNS)
    # Converte dose para valor numerico.
    frame["dose_numeric"] = _to_numeric_series(frame.get("dose_val_rx", pd.Series(dtype=object)))
    # Marca prescricoes intravenosas.
    frame["is_iv"] = frame.get("route", pd.Series(dtype=object)).fillna("").str.upper().str.contains("IV", regex=False).astype(int)
    # Ordena por medicamento e inicio quando a coluna existir.
    if "starttime" in frame.columns:
        frame["starttime"] = pd.to_datetime(frame["starttime"], errors="coerce")
        frame = frame.sort_values(["drug", "starttime"], na_position="last")
    # Calcula mudancas de dose por medicamento.
    frame["dose_changed"] = frame.groupby("drug")["dose_numeric"].diff().abs().gt(0).astype(int) if "drug" in frame.columns else 0
    # Calcula mudancas de rota por medicamento.
    frame["route_changed"] = frame.groupby("drug")["route"].transform(lambda s: s.fillna("").ne(s.fillna("").shift()).astype(int)).sub(1).clip(lower=0) if {"drug", "route"}.issubset(frame.columns) else 0
    # Cria uma unica linha resumindo a evolucao de prescricoes do paciente.
    result = pd.DataFrame(
        [
            {
                "total_prescriptions": len(frame),
                "unique_drugs": frame["drug"].nunique() if "drug" in frame.columns else 0,
                "iv_prescriptions": frame["is_iv"].sum(),
                "avg_dose": frame["dose_numeric"].mean(),
                "max_dose": frame["dose_numeric"].max(),
                "dose_change_count": frame["dose_changed"].sum(),
                "route_change_count": frame["route_changed"].sum(),
            }
        ]
    )
    # Retorna as colunas no formato usado pelo modelo.
    return _ensure_feature_columns(result, PRESCRIPTION_FEATURE_COLUMNS)


def build_transfer_training_features(transfers):
    """Cria features de treino para movimentacao do paciente."""
    # Trabalha em copia para evitar efeitos colaterais.
    frame = transfers.copy()
    # Converte horarios de entrada e saida para datetime.
    frame["intime"] = pd.to_datetime(frame["intime"], errors="coerce")
    frame["outtime"] = pd.to_datetime(frame["outtime"], errors="coerce")
    # Calcula permanencia em horas.
    frame["stay_hours"] = (frame["outtime"] - frame["intime"]).dt.total_seconds() / 3600
    # Substitui permanencias faltantes ou negativas por zero.
    frame["stay_hours"] = frame["stay_hours"].clip(lower=0).fillna(0)
    # Marca unidades de terapia intensiva pelo texto da unidade.
    frame["is_icu"] = frame["careunit"].fillna("").str.upper().str.contains("ICU|INTENSIVE", regex=True).astype(int)
    # Marca transferencias muito rapidas, que podem indicar instabilidade ou fluxo incomum.
    frame["is_quick_transfer"] = ((frame["stay_hours"] > 0) & (frame["stay_hours"] < 6)).astype(int)
    # Agrega dados por internacao.
    grouped = frame.groupby(["subject_id", "hadm_id"]).agg(
        transfer_count=("careunit", "count"),
        unique_careunits=("careunit", "nunique"),
        total_los_hours=("stay_hours", "sum"),
        avg_stay_hours=("stay_hours", "mean"),
        min_stay_hours=("stay_hours", "min"),
        icu_transfer_count=("is_icu", "sum"),
        quick_transfer_count=("is_quick_transfer", "sum"),
    )
    # Remove indice hierarquico.
    grouped = grouped.reset_index()
    # Retorna somente as colunas numericas esperadas.
    return _ensure_feature_columns(grouped, TRANSFER_FEATURE_COLUMNS)


def build_transfer_patient_features(transfers):
    """Cria features de movimentacao para um novo paciente recebido como input."""
    # Converte a lista de movimentacoes em dataframe.
    frame = pd.DataFrame(transfers)
    # Se nao houver movimentacoes, retorna dataframe vazio.
    if frame.empty:
        return pd.DataFrame(columns=TRANSFER_FEATURE_COLUMNS)
    # Converte horarios de entrada e saida.
    frame["intime"] = pd.to_datetime(frame.get("intime"), errors="coerce")
    frame["outtime"] = pd.to_datetime(frame.get("outtime"), errors="coerce")
    # Calcula tempo em horas em cada unidade.
    frame["stay_hours"] = (frame["outtime"] - frame["intime"]).dt.total_seconds() / 3600
    # Corrige valores invalidos para zero.
    frame["stay_hours"] = frame["stay_hours"].clip(lower=0).fillna(0)
    # Marca unidades de UTI.
    frame["is_icu"] = frame.get("careunit", pd.Series(dtype=object)).fillna("").str.upper().str.contains("ICU|INTENSIVE", regex=True).astype(int)
    # Marca transferencias muito rapidas.
    frame["is_quick_transfer"] = ((frame["stay_hours"] > 0) & (frame["stay_hours"] < 6)).astype(int)
    # Resume toda a movimentacao em uma linha.
    result = pd.DataFrame(
        [
            {
                "transfer_count": len(frame),
                "unique_careunits": frame["careunit"].nunique() if "careunit" in frame.columns else 0,
                "total_los_hours": frame["stay_hours"].sum(),
                "avg_stay_hours": frame["stay_hours"].mean(),
                "min_stay_hours": frame["stay_hours"].min(),
                "icu_transfer_count": frame["is_icu"].sum(),
                "quick_transfer_count": frame["is_quick_transfer"].sum(),
            }
        ]
    )
    # Retorna as colunas no formato do treinamento.
    return _ensure_feature_columns(result, TRANSFER_FEATURE_COLUMNS)
