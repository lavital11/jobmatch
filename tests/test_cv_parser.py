import io

import pytest

from jobmatch.cv_parser import CVParseError, extract_text, normalize_whitespace


def _make_minimal_pdf_bytes(content_stream: bytes) -> bytes:
    """Build a tiny, syntactically valid single-page PDF with the given content stream."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            f"<< /Length {len(content_stream)} >>\nstream\n".encode("latin-1")
            + content_stream
            + b"\nendstream"
        ),
    ]

    buf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf += f"{i} 0 obj\n".encode("latin-1")
        buf += body
        buf += b"\nendobj\n"

    xref_offset = len(buf)
    buf += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    buf += b"0000000000 65535 f \n"
    for off in offsets:
        buf += f"{off:010d} 00000 n \n".encode("latin-1")
    buf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    ).encode("latin-1")

    return bytes(buf)


def _make_text_pdf_bytes(text: str) -> bytes:
    content_stream = f"BT /F1 24 Tf 72 712 Td ({text}) Tj ET".encode("latin-1")
    return _make_minimal_pdf_bytes(content_stream)


def _make_docx_bytes(paragraph_text: str, table_cell_text: str) -> bytes:
    from docx import Document

    document = Document()
    document.add_paragraph(paragraph_text)
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Skill"
    table.cell(0, 1).text = table_cell_text

    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_extract_text_from_pdf_returns_page_text():
    pdf_bytes = _make_text_pdf_bytes("Hello JobMatch CV")

    text = extract_text(pdf_bytes, "resume.pdf")

    assert "Hello" in text
    assert "JobMatch" in text


def test_extract_text_from_scanned_pdf_raises_clear_error():
    pdf_bytes = _make_minimal_pdf_bytes(content_stream=b"")

    with pytest.raises(CVParseError, match="scanned image"):
        extract_text(pdf_bytes, "scanned_resume.pdf")


def test_extract_text_from_docx_includes_paragraphs_and_tables():
    docx_bytes = _make_docx_bytes(
        paragraph_text="Experience: Built a RAG pipeline.",
        table_cell_text="Python",
    )

    text = extract_text(docx_bytes, "resume.docx")

    assert "Built a RAG pipeline" in text
    assert "Python" in text


def test_extract_text_from_txt_decodes_utf8():
    text_bytes = "Junior developer with React experience.".encode("utf-8")

    text = extract_text(text_bytes, "resume.txt")

    assert text == "Junior developer with React experience."


def test_extract_text_from_empty_txt_raises_clear_error():
    with pytest.raises(CVParseError, match="empty"):
        extract_text(b"   \n  ", "resume.txt")


def test_extract_text_unsupported_extension_raises_clear_error():
    with pytest.raises(CVParseError, match="Unsupported file type"):
        extract_text(b"whatever bytes", "resume.xyz")


def test_extract_text_no_extension_raises_clear_error():
    with pytest.raises(CVParseError, match="Unsupported file type"):
        extract_text(b"whatever bytes", "resume")


def test_extract_text_normalizes_mid_sentence_line_breaks():
    # python-docx joins lines with "\n"; a sentence wrapped across two lines
    # should come back as a single space-joined line, not with the break kept.
    docx_bytes = _make_docx_bytes(
        paragraph_text="Experience building",
        table_cell_text="scalable backend systems",
    )

    text = extract_text(docx_bytes, "resume.docx")

    assert "\n" not in text


def test_normalize_whitespace_collapses_newlines_tabs_and_runs_of_spaces():
    messy = "Hello\nWorld\t\tfoo   bar\r\nbaz"

    assert normalize_whitespace(messy) == "Hello World foo bar baz"


def test_normalize_whitespace_strips_leading_and_trailing_whitespace():
    assert normalize_whitespace("   padded text   ") == "padded text"


def test_normalize_whitespace_is_idempotent():
    messy = "a\n\nb   c\td"
    once = normalize_whitespace(messy)
    twice = normalize_whitespace(once)

    assert once == twice


def test_normalize_whitespace_makes_line_wrapped_and_single_line_text_match():
    line_wrapped = "Built a REST\nAPI using\nFastAPI and Postgres."
    single_line = "Built a REST API using FastAPI and Postgres."

    assert normalize_whitespace(line_wrapped) == normalize_whitespace(single_line)
