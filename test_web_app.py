from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.app import _prepare_result_for_template


def test_prepare_result_for_template_formats_text_content(tmp_path):
    text_file = tmp_path / "resultado.txt"
    text_file.write_text(
        "TRANSCRICAO\nPaciente: dor no peito\n\nANALISE\nQueixa principal: dor no peito\n- Sintoma: dor\n",
        encoding="utf-8",
    )

    result = {
        "results": [
            {
                "type": "text",
                "filename": text_file.name,
                "path": str(text_file),
                "message": "Texto pronto para exibição.",
            }
        ]
    }

    prepared = _prepare_result_for_template(result)
    item = prepared["results"][0]

    assert item["formatted_text"]
    assert "TRANSCRICAO" in item["formatted_text"]
    assert "<h3>TRANSCRICAO</h3>" in item["formatted_text"]
    assert "<li>Sintoma: dor</li>" in item["formatted_text"]


def test_prepare_result_for_template_ignores_none_items():
    result = {"results": [None, {"type": "audio", "filename": "audio.wav", "message": "Texto"}]}

    prepared = _prepare_result_for_template(result)

    assert len(prepared["results"]) == 1
    assert prepared["results"][0]["type"] == "audio"
