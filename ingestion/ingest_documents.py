
import re
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown

from vectorstore.chroma_vectorstore import ChromaVectorStore, Retriever
from rag.kpi_extractor_rag import extract_financial_metrics
from database.save_metrics import save_metrics


def parse_company(pdf_file: Path) -> str:
    """Extract company name from the PDF filename."""

    parts = pdf_file.stem.split("_")

    ignored_words = {
        "annual",
        "report",
        "financial",
        "10k",
        "10-k"
    }

    company_parts = []

    for part in parts:
        if part.isdigit() and len(part) == 4:
            continue

        if part.lower() not in ignored_words:
            company_parts.append(part)

    company = " ".join(company_parts).strip().lower()

    if not company:
        raise ValueError(
            f"Could not identify company from filename: {pdf_file.name}"
        )

    return company


def detect_fiscal_year(markdown_file: str) -> int:
    """
    Detect the actual fiscal year from annual report content.
    """

    content = Path(markdown_file).read_text(
        encoding="utf-8",
        errors="ignore"
    )

    beginning = content[:30000]

    patterns = [
        r"fiscal\s+year\s+(?:ended|ending)\s+"
        r"[^.\n]{0,100}?\b(20\d{2})\b",

        r"year\s+ended\s+"
        r"[^.\n]{0,100}?\b(20\d{2})\b",

        r"for\s+the\s+fiscal\s+year\s+"
        r"[^.\n]{0,100}?\b(20\d{2})\b",

        r"\bFY\s*[- ]?(20\d{2})\b",

        r"\b(20\d{2})\s+Form\s+10-K\b",

        r"Form\s+10-K\s+"
        r"[^.\n]{0,100}?\b(20\d{2})\b"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            beginning,
            flags=re.IGNORECASE
        )

        if match:
            year = int(match.group(1))

            if 2000 <= year <= 2099:
                print(
                    f"Detected actual fiscal year: {year}",
                    flush=True
                )
                return year

    raise ValueError(
        "Could not identify the actual fiscal year in the report. "
        "Please check the PDF's title page and reporting period. "
        "No data was saved."
    )


def ingest_document(
    pdf_path: str,
    embeddings,
    vector_store
) -> None:
    """Convert, chunk, retrieve and save one annual report."""

    pdf_file = Path(pdf_path)

    company = parse_company(pdf_file)

    print(
        f"\nIngesting {pdf_file.name}",
        flush=True
    )

    print(
        f"Detected company: {company}",
        flush=True
    )

    converter = PDFToMarkdownConverter()

    # STEP 1: Convert PDF to Markdown.
    print("STEP 1: Starting PDF conversion...", flush=True)

    markdown_file = converter.convert_pdf(
        pdf_path=str(pdf_file),
        output_dir="data/markdown"
    )

    print("STEP 2: PDF converted to Markdown!", flush=True)

    # STEP 2: Detect actual fiscal year.
    report_year = detect_fiscal_year(markdown_file)

    # STEP 3: Semantic chunking.
    print("STEP 3: Starting semantic chunking...", flush=True)

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )

    print(
        f"STEP 4: Generated {len(chunks)} chunks",
        flush=True
    )

    if not chunks:
        raise ValueError("No document chunks were generated.")

    # STEP 4: Store chunks in ChromaDB.
    print("STEP 5: Saving chunks to ChromaDB...", flush=True)

    vector_store.upload_chunks(
        chunks=chunks,
        company=company,
        year=str(report_year),
        source_file=pdf_file.name
    )

    print("ChromaDB storage completed!", flush=True)

    # STEP 5: Extract financial metrics.
    print("STEP 6: Extracting financial metrics...", flush=True)

    retriever = Retriever(vector_store.client)

    metrics = extract_financial_metrics(
        retriever=retriever,
        company=company,
        year=report_year,
        markdown_file=markdown_file
    )

    if not metrics:
        raise ValueError("Financial KPI extraction returned no results.")

    financial_fields = [
        "revenue",
        "net_income",
        "operating_income",
        "cash_flow",
        "total_assets",
        "total_liabilities"
    ]

    verified_candidates = [
        metrics.get(field)
        for field in financial_fields
        if metrics.get(field) is not None
    ]

    if not verified_candidates:
        raise ValueError(
            "No financial KPI values were extracted. "
            "Existing database records were not overwritten."
        )

    # STEP 6: Save using actual fiscal year.
    print("STEP 7: Saving metrics to PostgreSQL...", flush=True)

    save_metrics(
        company=company,
        year=report_year,
        metrics=metrics
    )

    print(
        f"INGESTION COMPLETED: {company} FY{report_year}",
        flush=True
    )


def ingest_directory(input_dir: str) -> None:
    """Ingest all PDF reports in a directory."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = ChromaVectorStore(
        embeddings=embeddings,
        persist_directory="data/chroma_db"
    )

    pdf_files = list(Path(input_dir).glob("*.pdf"))

    print(
        f"Found {len(pdf_files)} PDF(s)",
        flush=True
    )

    for pdf_file in pdf_files:
        try:
            ingest_document(
                pdf_path=str(pdf_file),
                embeddings=embeddings,
                vector_store=vector_store
            )
        except Exception as error:
            print(
                f"FAILED to ingest {pdf_file.name}: {error}",
                flush=True
            )


if __name__ == "__main__":
    ingest_directory("data/raw_pdfs")

