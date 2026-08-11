from pathlib import Path
from io import BytesIO

from pypdf import PdfReader
from docx import Document


def load_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file.
    """

    pdf_stream = BytesIO(file_bytes)

    reader = PdfReader(pdf_stream)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def load_docx(file_bytes: bytes) -> str:
    """
    Extract text from a DOCX file.
    """

    doc_stream = BytesIO(file_bytes)

    document = Document(doc_stream)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def load_document(
    file_bytes: bytes,
    filename: str
) -> str:
    """
    Detect file type and extract text.
    """

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":

        return load_pdf(file_bytes)

    elif extension == ".docx":

        return load_docx(file_bytes)

    else:

        raise ValueError(
            "Unsupported file format. "
            "Please upload PDF or DOCX."
        )