
from pathlib import Path

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


def read_markdown(markdown_file: str) -> str:
    """Read Markdown content from a file."""

    file_path = Path(markdown_file)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Markdown file not found: {markdown_file}"
        )

    content = file_path.read_text(encoding="utf-8")

    if not content.strip():
        raise ValueError(
            f"Markdown file is empty: {markdown_file}"
        )

    return content


def chunk_markdown(
    markdown_file: str,
    embeddings
) -> list[Document]:
    """
    Generate semantic chunks from Markdown.

    Args:
        markdown_file: Markdown file path.
        embeddings: Hugging Face embedding model.

    Returns:
        List of semantic chunks.
    """

    print("Reading Markdown file...", flush=True)

    markdown_content = read_markdown(markdown_file)

    print("Starting semantic chunking...", flush=True)

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    chunks = splitter.create_documents([markdown_content])

    if not chunks:
        raise ValueError("Semantic chunking generated no chunks.")

    print(
        f"Semantic chunking completed! Generated {len(chunks)} chunks.",
        flush=True
    )

    return chunks


if __name__ == "__main__":

    # Project root directory
    repo_root = Path(__file__).resolve().parents[1]

    # Load free local Hugging Face embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    markdown_file = (
        repo_root / "data" / "markdown" / "apple_2024.md"
    )

    chunks = chunk_markdown(
        markdown_file=str(markdown_file),
        embeddings=embeddings
    )

    print(f"\nGenerated {len(chunks)} chunks\n")

    for index, chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print(f"Chunk {index + 1}")
        print("=" * 80)
        print(chunk.page_content[:1000])
        print()

