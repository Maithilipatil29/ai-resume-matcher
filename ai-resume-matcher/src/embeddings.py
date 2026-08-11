import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


def get_embedding_model():

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = HuggingFaceEmbeddings(

        model_name=model_name,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings