from pathlib import Path
import textwrap


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT_MARGIN = 54
TOP_MARGIN = 54
BOTTOM_MARGIN = 54
FONT_SIZE = 11
LINE_HEIGHT = 14
MAX_TEXT_WIDTH = 92


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def markdown_to_lines(markdown_text: str):
    lines = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# "):
            lines.append(("title", line[2:].strip()))
            lines.append(("blank", ""))
        elif line.startswith("## "):
            lines.append(("section", line[3:].strip()))
        elif line.startswith("### "):
            lines.append(("subsection", line[4:].strip()))
        elif line.startswith("- "):
            wrapped = textwrap.wrap(line[2:].strip(), width=MAX_TEXT_WIDTH - 4) or [""]
            first = True
            for chunk in wrapped:
                prefix = "- " if first else "  "
                lines.append(("body", f"{prefix}{chunk}"))
                first = False
        elif line.strip() == "":
            lines.append(("blank", ""))
        else:
            wrapped = textwrap.wrap(line.strip(), width=MAX_TEXT_WIDTH) or [""]
            for chunk in wrapped:
                lines.append(("body", chunk))
    return lines


def style_for(kind):
    if kind == "title":
        return ("Helvetica-Bold", 18)
    if kind == "section":
        return ("Helvetica-Bold", 14)
    if kind == "subsection":
        return ("Helvetica-Bold", 12)
    return ("Helvetica", FONT_SIZE)


def paginate(lines):
    pages = []
    current = []
    y = PAGE_HEIGHT - TOP_MARGIN
    for kind, text in lines:
        if kind == "blank":
            needed = LINE_HEIGHT
        else:
            _, size = style_for(kind)
            needed = size + 4
        if y - needed < BOTTOM_MARGIN:
            pages.append(current)
            current = []
            y = PAGE_HEIGHT - TOP_MARGIN
        current.append((kind, text, y))
        y -= needed
    if current:
        pages.append(current)
    return pages


def build_page_stream(page_items):
    commands = ["BT"]
    for kind, text, y in page_items:
        if kind == "blank":
            continue
        font_name, font_size = style_for(kind)
        font_ref = "/F1" if font_name == "Helvetica" else "/F2"
        escaped = escape_pdf_text(text)
        commands.append(f"{font_ref} {font_size} Tf")
        commands.append(f"1 0 0 1 {LEFT_MARGIN} {int(y)} Tm")
        commands.append(f"({escaped}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def write_pdf(markdown_path: Path, pdf_path: Path):
    lines = markdown_to_lines(markdown_path.read_text(encoding="utf-8"))
    pages = paginate(lines)

    objects = []

    def add_object(data: bytes):
        objects.append(data)
        return len(objects)

    font1_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font2_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_ids = []
    content_ids = []

    for page in pages:
        stream = build_page_stream(page)
        content_id = add_object(
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )
        content_ids.append(content_id)
        page_id = add_object(b"")
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_id = add_object(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")
    )

    for index, page_id in enumerate(page_ids):
        content_id = content_ids[index]
        objects[page_id - 1] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font1_id} 0 R /F2 {font2_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode("latin-1")
        )

    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("latin-1"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_start = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    pdf_path.write_bytes(output)


if __name__ == "__main__":
    write_pdf(
        Path("FINAL_RESULTS_AND_OBSERVATIONS.md"),
        Path("FINAL_RESULTS_AND_OBSERVATIONS.pdf"),
    )
