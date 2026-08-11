import os

from dotenv import load_dotenv


load_dotenv()


def get_top_k():

    return int(
        os.getenv("TOP_K", "5")
    )


def retrieve_documents(
    vector_store,
    query: str
):

    k = get_top_k()

    results = vector_store.similarity_search_with_relevance_scores(

        query,

        k=k
    )

    return results