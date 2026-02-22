"""PDF text extraction module."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
import pdfplumber
from loguru import logger


@dataclass
class PageText:
    """Represents extracted text from a single page."""
    page_number: int
    text: str
    has_images: bool
    word_count: int
    text_source: str = "embedded"  # "embedded" | "ocr" | "none"


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction."""
    file_path: str
    total_pages: int
    pages: list[PageText]
    extraction_method: str
    is_scanned: bool

    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def total_words(self) -> int:
        """Get total word count."""
        return sum(p.word_count for p in self.pages)


class PDFExtractor:
    """Extract text from PDF files.

    Uses PyMuPDF (fitz) as primary extractor, falls back to pdfplumber
    for complex layouts. Detects scanned PDFs that need OCR.
    """

    # Threshold for considering a PDF as scanned (needs OCR)
    MIN_TEXT_RATIO = 0.1  # If less than 10% of pages have text
    MIN_WORDS_PER_PAGE = 10

    def __init__(self, prefer_pdfplumber: bool = False):
        """Initialize PDF extractor.

        Args:
            prefer_pdfplumber: If True, use pdfplumber instead of PyMuPDF.
        """
        self.prefer_pdfplumber = prefer_pdfplumber

    def extract(self, file_path: str | Path) -> PDFExtractionResult:
        """Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            PDFExtractionResult with extracted text and metadata.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        logger.info(f"Extracting text from: {file_path.name}")

        if self.prefer_pdfplumber:
            return self._extract_with_pdfplumber(file_path)
        else:
            return self._extract_with_pymupdf(file_path)

    def _extract_with_pymupdf(self, file_path: Path) -> PDFExtractionResult:
        """Extract text using PyMuPDF (fitz)."""
        pages = []

        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                has_images = len(page.get_images()) > 0
                word_count = len(text.split()) if text else 0

                pages.append(PageText(
                    page_number=page_num,
                    text=text.strip(),
                    has_images=has_images,
                    word_count=word_count
                ))

            doc.close()

            # Determine if this is a scanned PDF
            pages_with_text = sum(1 for p in pages if p.word_count >= self.MIN_WORDS_PER_PAGE)
            is_scanned = (pages_with_text / total_pages) < self.MIN_TEXT_RATIO if total_pages > 0 else True

            return PDFExtractionResult(
                file_path=str(file_path),
                total_pages=total_pages,
                pages=pages,
                extraction_method="pymupdf",
                is_scanned=is_scanned
            )

        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying pdfplumber: {e}")
            return self._extract_with_pdfplumber(file_path)

    def _extract_with_pdfplumber(self, file_path: Path) -> PDFExtractionResult:
        """Extract text using pdfplumber (better for tables/complex layouts)."""
        pages = []

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                has_images = len(page.images) > 0 if hasattr(page, 'images') else False
                word_count = len(text.split()) if text else 0

                pages.append(PageText(
                    page_number=page_num,
                    text=text.strip(),
                    has_images=has_images,
                    word_count=word_count
                ))

        # Determine if this is a scanned PDF
        pages_with_text = sum(1 for p in pages if p.word_count >= self.MIN_WORDS_PER_PAGE)
        is_scanned = (pages_with_text / total_pages) < self.MIN_TEXT_RATIO if total_pages > 0 else True

        return PDFExtractionResult(
            file_path=str(file_path),
            total_pages=total_pages,
            pages=pages,
            extraction_method="pdfplumber",
            is_scanned=is_scanned
        )

    def extract_pages(self, file_path: str | Path,
                      start_page: int = 1,
                      end_page: int | None = None) -> Generator[PageText, None, None]:
        """Extract text from specific pages (for large files).

        Args:
            file_path: Path to the PDF file.
            start_page: Starting page number (1-indexed).
            end_page: Ending page number (inclusive), None for all remaining.

        Yields:
            PageText objects for each page.
        """
        file_path = Path(file_path)

        with fitz.open(file_path) as doc:
            end = end_page if end_page else len(doc)

            for page_num in range(start_page - 1, min(end, len(doc))):
                page = doc[page_num]
                text = page.get_text("text")
                has_images = len(page.get_images()) > 0
                word_count = len(text.split()) if text else 0

                yield PageText(
                    page_number=page_num + 1,
                    text=text.strip(),
                    has_images=has_images,
                    word_count=word_count
                )

    def needs_ocr(self, file_path: str | Path) -> bool:
        """Check if a PDF needs OCR (is scanned/image-based).

        Args:
            file_path: Path to the PDF file.

        Returns:
            True if the PDF appears to be scanned and needs OCR.
        """
        result = self.extract(file_path)
        return result.is_scanned


if __name__ == "__main__":
    # Test with a sample PDF
    import sys

    if len(sys.argv) > 1:
        extractor = PDFExtractor()
        result = extractor.extract(sys.argv[1])

        print(f"File: {result.file_path}")
        print(f"Pages: {result.total_pages}")
        print(f"Total words: {result.total_words}")
        print(f"Is scanned (needs OCR): {result.is_scanned}")
        print(f"Extraction method: {result.extraction_method}")
        print("\n--- First 500 characters ---")
        print(result.full_text[:500])
