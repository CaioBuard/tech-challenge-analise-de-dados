"""Fluxo de processamento usando LangGraph."""

from pathlib import Path
from typing import Any, Dict, List, TypedDict
import uuid

from langgraph.graph import END, StateGraph

from zip_processor import classify_files, extract_zip, process_classified_files


class ZipFlowState(TypedDict, total=False):
    """Estado trafegado entre etapas do fluxo."""

    zip_bytes: bytes
    generated_root: str
    upload_id: str
    temp_dir: str
    files: List[Any]
    classified_files: List[Dict[str, Any]]
    results: List[Dict[str, Any]]


def run_zip_flow(zip_bytes, generated_root):
    """Executa o fluxo de upload usando LangGraph."""
    # Define etapa de extracao.
    def extract_node(state):
        # Extrai ZIP para pasta temporaria.
        temp_dir, files = extract_zip(state["zip_bytes"])
        # Atualiza estado com arquivos encontrados.
        return {"temp_dir": str(temp_dir), "files": files}

    # Define etapa de classificacao.
    def classify_node(state):
        # Classifica arquivos por extensao.
        classified_files = classify_files(state["files"])
        # Atualiza estado com arquivos classificados.
        return {"classified_files": classified_files}

    # Define etapa de processamento.
    def process_node(state):
        # Cria um identificador unico para as imagens deste upload.
        upload_id = str(uuid.uuid4())
        # Cria pasta de saida deste upload.
        output_root = Path(state["generated_root"]) / upload_id
        # Garante que a pasta exista.
        output_root.mkdir(parents=True, exist_ok=True)
        # Processa arquivos classificados pelo proprio grafo.
        results = process_classified_files(state["classified_files"], output_root)
        # Retorna campos finais.
        return {"upload_id": upload_id, "results": results}

    # Cria grafo de estados.
    graph = StateGraph(ZipFlowState)
    # Adiciona etapa de extracao.
    graph.add_node("extract", extract_node)
    # Adiciona etapa de classificacao.
    graph.add_node("classify", classify_node)
    # Adiciona etapa de processamento.
    graph.add_node("process", process_node)
    # Define ponto de entrada do grafo.
    graph.set_entry_point("extract")
    # Liga extracao para classificacao.
    graph.add_edge("extract", "classify")
    # Liga classificacao para processamento.
    graph.add_edge("classify", "process")
    # Finaliza depois do processamento.
    graph.add_edge("process", END)
    # Compila o grafo.
    app = graph.compile()
    # Executa o grafo.
    final_state = app.invoke({"zip_bytes": zip_bytes, "generated_root": str(generated_root)})
    # Marca o motor usado no retorno.
    final_state["engine"] = "langgraph"
    # Retorna resultado final.
    return final_state
