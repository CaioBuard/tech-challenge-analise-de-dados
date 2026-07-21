"""Aplicacao Flask para upload de ZIP e exibicao de resultados."""

from pathlib import Path
import sys
import traceback

from flask import Flask, abort, render_template, request, send_from_directory, url_for

# Adiciona a raiz do projeto ao path para importar a biblioteca principal.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Garante que a raiz esteja no sys.path.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Adiciona a pasta web ao path para importar modulos locais.
WEB_ROOT = Path(__file__).resolve().parent

# Garante que a pasta web esteja no sys.path.
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

from flow import run_zip_flow


# Pasta onde imagens geradas pela web serao salvas.
GENERATED_ROOT = WEB_ROOT / "generated"

# Cria a aplicacao Flask usando templates e static dentro de web/.
app = Flask(
    __name__,
    template_folder=str(WEB_ROOT / "templates"),
    static_folder=str(WEB_ROOT / "static"),
)


def _image_url(path):
    """Converte caminho de imagem gerada para URL servida pelo Flask."""
    # Resolve o caminho recebido.
    image_path = Path(path).resolve()
    # Resolve a raiz permitida das imagens geradas.
    generated_root = GENERATED_ROOT.resolve()
    # Bloqueia caminhos fora da pasta generated.
    if generated_root not in image_path.parents and image_path != generated_root:
        return ""
    # Calcula caminho relativo para a rota /generated.
    relative_path = image_path.relative_to(generated_root).as_posix()
    # Retorna URL Flask para servir a imagem.
    return url_for("generated_file", filename=relative_path)


def _prepare_result_for_template(result):
    """Prepara resultado do fluxo para ser renderizado pelo Jinja2."""
    # Cria uma copia superficial para evitar alterar o resultado original.
    prepared = dict(result)
    # Lista de arquivos processados pronta para o template.
    prepared_results = []
    # Percorre cada item gerado pelo fluxo.
    for item in result.get("results", []):
        # Cria uma copia do item atual.
        prepared_item = dict(item)
        # Prepara URLs de imagens quando o item for planilha.
        if prepared_item.get("type") == "xls":
            # Lista de imagens com URL publica.
            prepared_images = []
            # Percorre imagens geradas pela biblioteca.
            for image in prepared_item.get("images", []):
                # Copia metadados da imagem.
                prepared_image = dict(image)
                # Adiciona URL que o navegador consegue carregar.
                prepared_image["url"] = _image_url(image.get("path", ""))
                # Guarda imagem preparada.
                prepared_images.append(prepared_image)
            # Substitui imagens brutas pelas imagens preparadas.
            prepared_item["images"] = prepared_images
        # Adiciona item preparado.
        prepared_results.append(prepared_item)
    # Atualiza lista de resultados.
    prepared["results"] = prepared_results
    # Retorna estrutura pronta para Jinja2.
    return prepared


@app.get("/")
def index():
    """Mostra pagina inicial de upload."""
    # Renderiza o template inicial.
    return render_template("index.html")


@app.post("/upload")
def upload():
    """Recebe ZIP, executa LangGraph e mostra resultados."""
    # Busca arquivo enviado pelo formulario.
    uploaded_file = request.files.get("zip_file")
    # Valida se existe arquivo.
    if uploaded_file is None or uploaded_file.filename == "":
        return render_template("error.html", message="Nenhum arquivo enviado."), 400
    # Valida extensao do arquivo.
    if not uploaded_file.filename.lower().endswith(".zip"):
        return render_template("error.html", message="Envie um arquivo .zip."), 400
    # Processa upload com tratamento de erro amigavel.
    try:
        # Le os bytes do ZIP enviado.
        zip_bytes = uploaded_file.read()
        # Executa o fluxo LangGraph.
        result = run_zip_flow(zip_bytes, GENERATED_ROOT)
        # Prepara URLs e metadados para o template.
        prepared_result = _prepare_result_for_template(result)
        # Renderiza a pagina de resultado.
        return render_template("result.html", result=prepared_result)
    # Em caso de erro, mostra mensagem simples e registra stack trace no terminal.
    except Exception as exc:
        # Imprime erro completo no terminal para facilitar depuracao.
        traceback.print_exc()
        # Renderiza pagina de erro.
        return render_template("error.html", message=str(exc)), 500


@app.get("/generated/<path:filename>")
def generated_file(filename):
    """Serve imagens geradas pela biblioteca."""
    # Resolve caminho pedido pelo navegador.
    file_path = (GENERATED_ROOT / filename).resolve()
    # Resolve raiz permitida.
    generated_root = GENERATED_ROOT.resolve()
    # Bloqueia acesso fora de generated.
    if generated_root not in file_path.parents and file_path != generated_root:
        abort(403)
    # Serve arquivo usando send_from_directory.
    return send_from_directory(generated_root, filename)


def main():
    """Sobe o servidor Flask local."""
    # Garante que a pasta de arquivos gerados exista.
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    # Mostra URL no terminal.
    print("Servidor rodando em http://localhost:8000")
    # Inicia Flask sem modo debug para evitar processo duplicado no Windows.
    app.run(host="localhost", port=8000, debug=False, use_reloader=False)


# Executa o servidor quando o arquivo for chamado diretamente.
if __name__ == "__main__":
    # Chama funcao principal.
    main()
