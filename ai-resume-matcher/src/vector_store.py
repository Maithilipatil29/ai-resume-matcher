from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.embeddings import get_embedding_model


def create_documents(chunks: List[str]):

    documents = []

    for index, chunk in enumerate(chunks):

        document = Document(

            page_content=chunk,

            metadata={
                "chunk_id": index
            }
        )

        documents.append(document)

    return documents


def create_vector_store(chunks: List[str]):

    documents = create_documents(chunks)

    embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(

        documents,

        embeddings
    )

    return vector_store