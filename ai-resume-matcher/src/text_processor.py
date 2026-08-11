import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """
    Clean extracted document text.
    """

    # Remove null characters
    text = text.replace("\x00", " ")

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def split_text(text: str):
    """
    Split document into semantic chunks using
    LangChain RecursiveCharacterTextSplitter.
    """

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=700,

        chunk_overlap=100,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_text(text)

    return chunks