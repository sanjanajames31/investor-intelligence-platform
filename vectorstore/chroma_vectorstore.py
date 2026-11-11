
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ChromaVectorStore:
    """Store and retrieve financial document chunks using ChromaDB."""

    def __init__(
        self,
        embeddings=None,
        persist_directory="data/chroma_db",
        collection_name="financial_documents"
    ):
        self.persist_directory = str(
            PROJECT_ROOT / persist_directory
        ) if not Path(persist_directory).is_absolute() else persist_directory

        self.collection_name = collection_name

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True
        )

        # Reuse the supplied embedding model.
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.client = Chroma(
            collection_name=collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

        print("ChromaDB initialized successfully!", flush=True)

    def upload_chunks(
        self,
        chunks,
        company="",
        year="",
        source_file=""
    ):
        """Upload semantic chunks to ChromaDB."""

        if not chunks:
            print("No chunks to upload.", flush=True)
            return

        prepared_chunks = []
        ids = []

        for index, chunk in enumerate(chunks):

            metadata = dict(chunk.metadata or {})

            metadata.update({
                "company": str(company),
                "year": str(year),
                "source_file": str(source_file)
            })

            prepared_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata
                )
            )

            # Generate a stable unique ID for each chunk.
            unique_text = (
                f"{company}_{year}_{source_file}_"
                f"{index}_{chunk.page_content}"
            )

            chunk_id = hashlib.sha256(
                unique_text.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)

        print(
            f"Uploading {len(prepared_chunks)} chunks to ChromaDB...",
            flush=True
        )

        self.client.add_documents(
            documents=prepared_chunks,
            ids=ids
        )

        total_chunks = self.client._collection.count()

        print(
            f"Successfully stored {len(prepared_chunks)} chunks!",
            flush=True
        )

        print(
            f"Total chunks in ChromaDB: {total_chunks}",
            flush=True
        )


class Retriever:
    """Retrieve relevant financial information from ChromaDB."""

    def __init__(self, client):
        self.client = client

    def invoke(
        self,
        query,
        company=None,
        year=None,
        top_k=20
    ):
        """Retrieve relevant chunks using similarity search."""

        conditions = []

        if company:
            conditions.append({
                "company": {"$eq": str(company)}
            })

        if year is not None:
            conditions.append({
                "year": {"$eq": str(year)}
            })

        if len(conditions) == 2:
            filter_dict = {
                "$and": conditions
            }
        elif len(conditions) == 1:
            filter_dict = conditions[0]
        else:
            filter_dict = None

        results = self.client.similarity_search(
            query=query,
            k=top_k,
            filter=filter_dict
        )

        print(
            f"Retrieved {len(results)} relevant chunks.",
            flush=True
        )

        return results
