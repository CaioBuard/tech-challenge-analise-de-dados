"""Aplicacao Flask para upload de ZIP e exibicao de resultados."""

from html import escape
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


# Pasta onde a aplicacao salva arquivos gerados para cada upload.
OUTPUT_ROOT = WEB_ROOT / "generated"

# Cria a aplicacao Flask usando templates e static dentro de web/.
app = Flask(
    __name__,
    template_folder=str(WEB_ROOT / "templates"),
    static_folder=str(WEB_ROOT / "static"),
)


def _image_url(path):
    """Converte um arquivo gerado para URL servida pelo Flask."""
    image_path = Path(path).resolve()
    output_root = OUTPUT_ROOT.resolve()

    if output_root not in image_path.parents and image_path != output_root:
        return ""

    relative_path = image_path.relative_to(output_root).as_posix()
    return url_for("generated_file", filename=relative_path)


def _generated_url(path):
    """Converte qualquer relatorio dentro de generated para uma URL publica."""
    return _image_url(path)


def _render_text_content(text_content):
    """Converte texto simples em HTML seguro para a pagina de resultados."""
    if not text_content:
        return ""

    html_parts = []
    in_list = False

    for raw_line in str(text_content).splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue

        upper_line = line.upper()
        if upper_line.startswith("TRANSCRICAO"):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h3>{escape('TRANSCRICAO')}</h3>")
        elif upper_line.startswith("ANALISE"):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h3>{escape('ANALISE CLINICA PRELIMINAR')}</h3>")
        elif line.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{escape(line[2:])}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<p>{escape(line)}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "".join(html_parts)


def _prepare_result_for_template(result):
    """Prepara resultado do fluxo para ser renderizado pelo Jinja2."""
    prepared = dict(result)
    prepared_results = []

    for item in result.get("results", []):
        if item is None:
            continue

        prepared_item = dict(item)

        if prepared_item.get("type") == "xls":
            prepared_images = []
            for image in prepared_item.get("images", []):
                prepared_image = dict(image)
                prepared_image["url"] = _image_url(image.get("path", ""))
                prepared_images.append(prepared_image)
            prepared_item["images"] = prepared_images
        elif prepared_item.get("type") == "video":
            evaluation = dict(prepared_item.get("evaluation", {}))
            report_paths = evaluation.get("report_paths", {})
            evaluation["report_urls"] = {
                name: _generated_url(path)
                for name, path in report_paths.items()
            }
            prepared_item["evaluation"] = evaluation
        elif prepared_item.get("type") in {"text", "audio"}:
            if not prepared_item.get("formatted_text"):
                text_content = prepared_item.get("text")
                if not text_content and prepared_item.get("path"):
                    candidate_path = Path(prepared_item["path"])
                    if candidate_path.is_file():
                        try:
                            text_content = candidate_path.read_text(encoding="utf-8")
                        except (OSError, UnicodeDecodeError):
                            text_content = None

                if not text_content:
                    text_content = prepared_item.get("message") or prepared_item.get("content")

                prepared_item["formatted_text"] = _render_text_content(text_content)

        prepared_results.append(prepared_item)

    prepared["results"] = prepared_results
    return prepared


@app.get("/")
def index():
    """Mostra pagina inicial de upload."""
    return render_template("index.html")


@app.post("/upload")
def upload():
    """Recebe ZIP, executa LangGraph e mostra resultados."""
    uploaded_file = request.files.get("zip_file")
    if uploaded_file is None or uploaded_file.filename == "":
        return render_template("error.html", message="Nenhum arquivo enviado."), 400
    if not uploaded_file.filename.lower().endswith(".zip"):
        return render_template("error.html", message="Envie um arquivo .zip."), 400

    try:
        zip_bytes = uploaded_file.read()
        print(f"[WEB] Processando upload: {uploaded_file.filename}", flush=True)
        result = run_zip_flow(zip_bytes, OUTPUT_ROOT)
        print(f"[WEB] Upload concluido: {uploaded_file.filename}", flush=True)
        prepared_result = _prepare_result_for_template(result)
        return render_template("result.html", result=prepared_result)
    except Exception as exc:
        traceback.print_exc()
        return render_template("error.html", message=str(exc)), 500


@app.get("/generated/<path:filename>")
def generated_file(filename):
    """Serve arquivos gerados pela aplicacao."""
    file_path = (OUTPUT_ROOT / filename).resolve()
    output_root = OUTPUT_ROOT.resolve()

    if output_root not in file_path.parents and file_path != output_root:
        abort(403)

    return send_from_directory(output_root, filename)


def main():
    """Sobe o servidor Flask local."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Servidor rodando em http://localhost:8000")
    app.run(host="localhost", port=8000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
