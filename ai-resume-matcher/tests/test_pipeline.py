from src.text_processor import (
    clean_text,
    split_text
)


def test_clean_text():

    text = """
    Hello      World


    Python
    """

    cleaned = clean_text(
        text
    )

    assert "Hello World" in cleaned

    assert "Python" in cleaned


def test_split_text():

    text = (
        "Python is a programming language. "
        * 100
    )

    chunks = split_text(
        text
    )

    assert len(chunks) > 1

    assert all(
        isinstance(chunk, str)
        for chunk in chunks
    )