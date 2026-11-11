
import sys

from vectorstore.chroma_vectorstore import ChromaVectorStore


def search_vectorstore(query: str, top: int = 5):

    # Connect to local ChromaDB
    store = ChromaVectorStore(
        persist_directory="data/chroma_db"
    )

    # Search for similar documents
    results = store.client.similarity_search(
        query=query,
        k=top
    )

    print(f"Query: {query!r}")
    print(f"Top: {top}")
    print(f"Results: {len(results)}\n")

    for idx, result in enumerate(results, start=1):

        content = result.page_content

        snippet = content.strip().replace("\n", " ")

        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."

        print(f"Result {idx}")
        print(f"Content snippet: {snippet}")

        print(f"Metadata: {result.metadata}")

        print("  " + "-" * 60)

    if not results:
        print(
            "No results returned. "
            "Check whether documents have been ingested."
        )

    return results


def main():

    if len(sys.argv) < 2:
        print(
            'Usage: python -m rag.retrieval_debug "your query here"'
        )
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    search_vectorstore(query)


if __name__ == "__main__":
    main()
