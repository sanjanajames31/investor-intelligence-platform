
from pathlib import Path

import pymupdf
import pymupdf4llm


class PDFToMarkdownConverter:
    """Convert PDF documents to Markdown."""

    def convert_pdf(self, pdf_path: str, output_dir: str) -> str:
        """
        Convert a PDF document to Markdown in batches.

        Args:
            pdf_path: Source PDF path.
            output_dir: Output markdown directory.

        Returns:
            Generated markdown file path.
        """
        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get total number of pages
        document = pymupdf.open(str(pdf_file))
        total_pages = len(document)
        document.close()

        all_markdown = []
        batch_size = 10

        # Convert PDF in batches of 10 pages
        for start in range(0, total_pages, batch_size):
            end = min(start + batch_size, total_pages)

            print(
                f"Converting pages {start + 1}-{end} "
                f"of {total_pages}...",
                flush=True
            )

            markdown_content = pymupdf4llm.to_markdown(
                str(pdf_file),
                pages=list(range(start, end))
            )

            all_markdown.append(markdown_content)

        # Combine all Markdown batches
        combined_markdown = "\n\n".join(all_markdown)

        markdown_file = output_path / f"{pdf_file.stem}.md"

        markdown_file.write_text(
            combined_markdown,
            encoding="utf-8"
        )

        print("PDF conversion completed!", flush=True)

        return str(markdown_file)

    def convert_directory(self, input_dir: str, output_dir: str) -> list[str]:
        """
        Convert all PDFs from a directory to Markdown.

        Args:
            input_dir: Directory containing PDF files.
            output_dir: Directory to save markdown files.

        Returns:
            List of generated markdown file paths.
        """
        input_path = Path(input_dir)
        markdown_files = []

        for pdf_file in input_path.glob("*.pdf"):
            markdown_file = self.convert_pdf(
                pdf_path=str(pdf_file),
                output_dir=output_dir
            )

            markdown_files.append(markdown_file)

        return markdown_files


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    input_dir = repo_root / "data" / "raw_pdfs"
    output_dir = repo_root / "data" / "markdown"

    converter = PDFToMarkdownConverter()

    markdown_files = converter.convert_directory(
        input_dir=str(input_dir),
        output_dir=str(output_dir)
    )

    print(f"Successfully converted {len(markdown_files)} PDF(s):")

    for markdown_file in markdown_files:
        print(f"  - {markdown_file}")

