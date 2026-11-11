
import re
from pathlib import Path

from pydantic import BaseModel, Field

from llm.ollama_client import get_structured_completion
from vectorstore.chroma_vectorstore import ChromaVectorStore, Retriever


class FinancialMetrics(BaseModel):
    """Structured financial information extracted from annual reports."""

    revenue: str | int | float | None = None
    net_income: str | int | float | None = None
    operating_income: str | int | float | None = None
    cash_flow: str | int | float | None = None
    total_assets: str | int | float | None = None
    total_liabilities: str | int | float | None = None
    risk_factors: str | list[str] | None = None
    growth_drivers: str | list[str] | None = None


class BusinessInsights(BaseModel):
    """Business risks and growth drivers extracted by Ollama."""

    risk_factors: str | list[str] | None = Field(
        default=None,
        description="Important business and financial risks."
    )

    growth_drivers: str | list[str] | None = Field(
        default=None,
        description="Important factors supporting business growth."
    )


def clean_cell(value: str) -> str:
    """Remove Markdown formatting from a table cell or heading."""

    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*>\s*", "", value)
    value = value.replace("**", "").replace("__", "")
    value = value.replace("*", "").replace("_", "")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_heading(value: str) -> str:
    """Normalize Markdown heading text for matching."""

    value = clean_cell(value)
    value = re.sub(r"^#+\s*", "", value)
    return value.upper().strip()


def parse_number(value: str) -> str | None:
    """Validate and normalize a financial table value."""

    value = clean_cell(value)
    value = value.replace("$", "").replace(",", "").strip()

    if not value or value in {"—", "-", "–"}:
        return None

    # Handle negative values written as (12345).
    if re.fullmatch(r"\([\d]+(?:\.\d+)?\)", value):
        return "-" + value[1:-1]

    # Accept normal numbers and negative numbers.
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return value

    return None


def find_statement_table(
    lines: list[str],
    heading: str,
    year: int
) -> tuple[dict[str, str], str]:
    """
    Find a financial statement even when its heading spans
    multiple Markdown lines.

    Select the requested fiscal-year column from the table header.
    """

    heading_pattern = re.compile(
        heading,
        flags=re.IGNORECASE
    )

    # Find headings using a rolling window of six lines.
    heading_candidates = []

    for index in range(len(lines)):
        window = " ".join(
            normalize_heading(line)
            for line in lines[index:index + 6]
        )

        if heading_pattern.search(window):
            heading_candidates.append(index)

    if not heading_candidates:
        raise ValueError(
            f"Financial statement not found: {heading}"
        )

    # Try each matching heading. This avoids selecting a
    # table of contents entry instead of the actual statement.
    for heading_index in heading_candidates:

        table_lines = []

        for line in lines[heading_index + 1:heading_index + 120]:

            stripped = line.strip()

            # Stop at the next Markdown heading.
            if re.match(r"^\s*>\s*#+\s", stripped):
                break

            if re.match(r"^\s*#+\s", stripped):
                break

            # Remove an optional blockquote prefix.
            cleaned_line = re.sub(
                r"^\s*>\s?",
                "",
                stripped
            )

            if cleaned_line.startswith("|"):
                table_lines.append(cleaned_line)

        if not table_lines:
            continue

        year_column = None

        # Find the actual date/year column in the table header.
        for line in table_lines:

            cells = [
                clean_cell(cell)
                for cell in line.split("|")[1:-1]
            ]

            for column_index, cell in enumerate(cells):

                # Match an actual statement date, not a
                # table-of-contents description mentioning a year.
                date_pattern = (
                    rf"\b(?:September|Sept)\s+\d{{1,2}},?\s*{year}\b"
                )

                if re.search(date_pattern, cell, re.IGNORECASE):
                    year_column = column_index
                    break

            if year_column is not None:
                break

        if year_column is None:
            continue

        rows = {}

        # Extract financial values from the selected year column.
        for line in table_lines:

            cells = [
                clean_cell(cell)
                for cell in line.split("|")[1:-1]
            ]

            if len(cells) <= year_column:
                continue

            label = cells[0].strip().lower()

            if not label:
                continue

            value = parse_number(cells[year_column])

            if value is not None:
                rows[label] = value

        # Detect units near the statement heading.
        unit = "million"

        unit_text = " ".join(
            clean_cell(line)
            for line in lines[
                heading_index:heading_index + 12
            ]
        )

        unit_match = re.search(
            r"\bin\s+(billions|millions|thousands)\b",
            unit_text,
            flags=re.IGNORECASE
        )

        if unit_match:
            unit = unit_match.group(1).lower()

        if rows:
            print(
                f"Found statement: {heading}",
                flush=True
            )
            print(
                f"Selected FY{year} column: {year_column}",
                flush=True
            )

            return rows, unit

    raise ValueError(
        f"Could not find a valid FY{year} table for: {heading}"
    )


def extract_table_kpis(
    markdown_file: str,
    year: int
) -> dict:
    """Extract six numeric financial KPIs from Markdown statements."""

    path = Path(markdown_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Markdown file not found: {markdown_file}"
        )

    content = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = content.splitlines()

    print(
        f"Extracting financial tables for FY{year}...",
        flush=True
    )

    operations, operations_unit = find_statement_table(
        lines,
        r"CONSOLIDATED\s+STATEMENTS\s+OF\s+OPERATIONS",
        year
    )

    balance_sheet, balance_unit = find_statement_table(
        lines,
        r"CONSOLIDATED\s+BALANCE\s+SHEETS",
        year
    )

    cash_flow_statement, cash_unit = find_statement_table(
        lines,
        r"CONSOLIDATED\s+STATEMENTS\s+OF\s+CASH\s+FLOWS",
        year
    )

    required_rows = {
        "revenue": (
            operations,
            "total net sales",
            operations_unit
        ),
        "net_income": (
            operations,
            "net income",
            operations_unit
        ),
        "operating_income": (
            operations,
            "operating income",
            operations_unit
        ),
        "cash_flow": (
            cash_flow_statement,
            "cash generated by operating activities",
            cash_unit
        ),
        "total_assets": (
            balance_sheet,
            "total assets",
            balance_unit
        ),
        "total_liabilities": (
            balance_sheet,
            "total liabilities",
            balance_unit
        )
    }

    results = {}

    for field, (table, row_name, unit) in required_rows.items():

        value = table.get(row_name)

        if value is not None:
            results[field] = f"{value} {unit}"
        else:
            results[field] = None

        print(
            f"{field}: {results[field]}",
            flush=True
        )

    missing_fields = [
        field
        for field, value in results.items()
        if value is None
    ]

    if missing_fields:
        raise ValueError(
            "Some financial KPIs could not be extracted: "
            + ", ".join(missing_fields)
        )

    return results


def retrieve_context(
    retriever: Retriever,
    company: str,
    year: int,
    metadata_year: int | None = None
) -> str:
    """Retrieve relevant business information from ChromaDB."""

    search_year = (
        metadata_year
        if metadata_year is not None
        else year
    )

    queries = [
        "Major business risks risk factors",
        "Business growth drivers products services and market growth"
    ]

    all_documents = []
    seen_content = set()

    print("Retrieving business insights...", flush=True)

    for query in queries:

        documents = retriever.invoke(
            query=query,
            company=company,
            year=search_year,
            top_k=6
        )

        for document in documents:

            content = document.page_content.strip()

            if content and content not in seen_content:
                seen_content.add(content)
                all_documents.append(content)

    print(
        f"Retrieved {len(all_documents)} unique business chunks.",
        flush=True
    )

    return "\n\n".join(all_documents)


def build_insights_prompt(
    company: str,
    year: int,
    context: str
) -> str:
    """Build the Ollama prompt for risks and growth drivers."""

    return f"""
You are a financial analyst reading an official annual report.

COMPANY: {company}
FISCAL YEAR: {year}

DOCUMENT CONTEXT:
{context}

Extract only these two fields:

1. risk_factors
2. growth_drivers

RULES:

- Use only the provided document context.
- Do not use outside knowledge.
- Do not invent information.
- Extract risks specifically disclosed by the company.
- Extract growth drivers specifically supported by the report.
- Keep both fields concise and meaningful.
- If information is unavailable, return null.

Return the structured response matching the requested schema.
"""


def extract_financial_metrics(
    retriever: Retriever,
    company: str,
    year: int,
    metadata_year: int | None = None,
    markdown_file: str | None = None
) -> dict:
    """Extract numeric KPIs from tables and insights using Ollama."""

    print(
        f"Starting KPI extraction for {company}, fiscal year {year}...",
        flush=True
    )

    if not markdown_file:
        raise ValueError(
            "markdown_file is required for accurate financial KPI extraction."
        )

    # STEP 1: Extract six numeric KPIs directly from financial tables.
    financial_results = extract_table_kpis(
        markdown_file=markdown_file,
        year=year
    )

    # STEP 2: Retrieve business information from ChromaDB.
    context = retrieve_context(
        retriever=retriever,
        company=company,
        year=year,
        metadata_year=metadata_year
    )

    risk_growth_results = {
        "risk_factors": None,
        "growth_drivers": None
    }

    # STEP 3: Use Ollama only for qualitative business insights.
    if context.strip():

        prompt = build_insights_prompt(
            company=company,
            year=year,
            context=context
        )

        print(
            "Sending business context to Ollama...",
            flush=True
        )

        insights = get_structured_completion(
            prompt=prompt,
            response_model=BusinessInsights
        )

        if isinstance(insights, BusinessInsights):
            risk_growth_results = insights.model_dump()
        else:
            risk_growth_results = (
                BusinessInsights.model_validate(insights).model_dump()
            )

    else:
        print(
            "No business insight context found. "
            "Risk and growth fields will remain null.",
            flush=True
        )

    # STEP 4: Combine financial KPIs and business insights.
    results = {
        **financial_results,
        **risk_growth_results
    }

    print("\nFinal extracted KPI results:", flush=True)

    for key, value in results.items():
        print(f"{key}: {value}", flush=True)

    print(
        "\nFinancial KPI extraction completed!",
        flush=True
    )

    return results


def main() -> None:
    """Run financial KPI extraction and save results."""

    from database.save_metrics import save_metrics

    company = "apple"
    report_year = 2025
    stored_metadata_year = 2024

    markdown_file = "data/markdown/apple_2024.md"

    vector_store = ChromaVectorStore(
        persist_directory="data/chroma_db"
    )

    retriever = Retriever(vector_store.client)

    results = extract_financial_metrics(
        retriever=retriever,
        company=company,
        year=report_year,
        metadata_year=stored_metadata_year,
        markdown_file=markdown_file
    )

    print(
        "\nSaving financial metrics to PostgreSQL...",
        flush=True
    )

    save_metrics(
        company=company,
        year=report_year,
        metrics=results
    )

    print(
        "Financial metrics saved successfully!",
        flush=True
    )


if __name__ == "__main__":
    main()
