#!/usr/bin/env python3
"""Text extraction from all supported file types.

Coordinates PDF, OCR, and audio extraction based on file type.

Usage:
    python scripts/extract_text.py data/raw/doj/dataset-1/
    python scripts/extract_text.py --file data/raw/doj/dataset-1/EFTA00000001.pdf
"""

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.extractors import PDFExtractor, OCRExtractor, AudioExtractor
from src.text_quality import is_text_layer_good, analyze_page_textiness

try:
    import fitz as fitz_render
    from PIL import Image as PILImage
except ImportError:
    fitz_render = None
    PILImage = None


# File type mappings
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
DOC_EXTENSIONS = {".doc", ".docx"}
TEXT_EXTENSIONS = {".txt", ".text"}

ALL_SUPPORTED = (PDF_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS |
                 VIDEO_EXTENSIONS | DOC_EXTENSIONS | TEXT_EXTENSIONS)


class TextExtractor:
    """Unified text extraction from multiple file types."""

    def __init__(self, output_dir: str | Path, use_ocr: bool = True,
                 force_ocr: bool = False, auto_quality: bool = True):
        """Initialize the extractor.

        Args:
            output_dir: Directory to save extracted text.
            use_ocr: Whether to use OCR for scanned PDFs/images.
            force_ocr: Always use OCR for PDFs, ignoring embedded text.
            auto_quality: Check text layer quality and fall back to OCR if bad.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_ocr = use_ocr
        self.force_ocr = force_ocr
        self.auto_quality = auto_quality

        # Initialize extractors lazily
        self._pdf_extractor = None
        self._ocr_extractor = None
        self._audio_extractor = None

    @property
    def pdf_extractor(self):
        if self._pdf_extractor is None:
            self._pdf_extractor = PDFExtractor()
        return self._pdf_extractor

    @property
    def ocr_extractor(self):
        if self._ocr_extractor is None:
            self._ocr_extractor = OCRExtractor()
        return self._ocr_extractor

    @property
    def audio_extractor(self):
        if self._audio_extractor is None:
            self._audio_extractor = AudioExtractor(model_name="medium")
        return self._audio_extractor

    def get_output_path(self, input_path: Path, suffix: str = ".json") -> Path:
        """Get output path for extracted text.

        Args:
            input_path: Input file path.
            suffix: Output file suffix.

        Returns:
            Output file path.
        """
        return self.output_dir / f"{input_path.stem}{suffix}"

    def is_processed(self, input_path: Path) -> bool:
        """Check if a file has already been processed.

        Args:
            input_path: Input file path.

        Returns:
            True if already processed.
        """
        output_path = self.get_output_path(input_path)
        return output_path.exists()

    def extract_from_file(self, file_path: Path, force: bool = False) -> dict | None:
        """Extract text from a single file.

        Args:
            file_path: Path to input file.
            force: Re-extract even if already processed.

        Returns:
            Extraction result dict or None if failed.
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix not in ALL_SUPPORTED:
            logger.warning(f"Unsupported file type: {suffix}")
            return None

        # Check if already processed
        if not force and self.is_processed(file_path):
            logger.debug(f"Already processed: {file_path.name}")
            return None

        logger.info(f"Processing: {file_path.name}")

        try:
            if suffix in PDF_EXTENSIONS:
                result = self._extract_pdf(file_path)
            elif suffix in IMAGE_EXTENSIONS:
                result = self._extract_image(file_path)
            elif suffix in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
                result = self._extract_audio(file_path)
            elif suffix in DOC_EXTENSIONS:
                result = self._extract_docx(file_path)
            elif suffix in TEXT_EXTENSIONS:
                result = self._extract_text(file_path)
            else:
                return None

            # Save result
            self._save_result(file_path, result)
            return result

        except Exception as e:
            logger.error(f"Failed to extract from {file_path.name}: {e}")
            return None

    def _ocr_result_to_dict(self, file_path: Path, ocr_result, method_note: str = "") -> dict:
        """Convert an OCR result to output dict."""
        method = "ocr" + (f"_{method_note}" if method_note else "")
        return {
            "file_path": str(file_path),
            "file_type": "pdf",
            "extraction_method": method,
            "total_pages": ocr_result.total_pages,
            "pages": [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "text_source": "ocr",
                    "confidence": p.confidence,
                }
                for p in ocr_result.pages
            ],
            "full_text": ocr_result.full_text,
        }

    def _extract_pdf(self, file_path: Path) -> dict:
        """Extract text from PDF with page-level hybrid routing.

        Pipeline:
        1. If --force-ocr → OCR all pages immediately
        2. PyMuPDF extracts embedded text from all pages
        3. Per-page classification:
           a. word_count >= 10 → embedded (good text layer)
           b. word_count < 10 AND no images → none (blank/vector page)
           c. word_count < 10 AND has_images → render + textiness check:
              - text_like / uncertain → OCR this page
              - photo_like / blank → skip OCR
        4. Quality gate on combined embedded text:
           - If bad → reclassify all embedded pages for OCR
        5. Selective OCR only on pages that need it
        6. Merge embedded + OCR pages into final result
        """
        # Force OCR mode: skip PDF text extraction entirely
        if self.force_ocr and self.use_ocr:
            logger.info(f"Force OCR mode, using OCR: {file_path.name}")
            ocr_result = self.ocr_extractor.extract_from_pdf(file_path)
            return self._ocr_result_to_dict(file_path, ocr_result, "forced")

        result = self.pdf_extractor.extract(file_path)

        # Classify each page
        embedded_pages = []   # pages with good embedded text
        ocr_candidates = []   # page numbers that need OCR
        none_pages = []       # blank/photo pages (no text)
        page_meta = {}        # per-page metadata for debugging

        # Identify pages needing further analysis
        needs_render = []  # (page_number, PageText) pairs needing image analysis
        for page in result.pages:
            if page.word_count >= 10:
                embedded_pages.append(page)
                page_meta[page.page_number] = {
                    "text_source": "embedded",
                    "page_type": "text",
                }
            elif not page.has_images:
                none_pages.append(page)
                page_meta[page.page_number] = {
                    "text_source": "none",
                    "page_type": "blank",
                }
            else:
                needs_render.append(page)

        # For pages with images but no embedded text: render and check textiness
        if needs_render and self.use_ocr:
            # Open PDF once with PyMuPDF for rendering (no poppler dependency)
            render_doc = fitz_render.open(str(file_path))
            for page in needs_render:
                try:
                    fitz_page = render_doc[page.page_number - 1]  # 0-indexed
                    mat = fitz_render.Matrix(150 / 72, 150 / 72)  # 150 DPI
                    pix = fitz_page.get_pixmap(matrix=mat)
                    pil_img = PILImage.frombytes(
                        "RGB", [pix.width, pix.height], pix.samples
                    )
                    textiness = analyze_page_textiness(pil_img)
                    page_meta[page.page_number] = {
                        "text_source": "pending",
                        "page_type": textiness.page_type,
                        "dark_ratio": round(textiness.dark_ratio, 4),
                        "line_count": textiness.line_count,
                    }
                    if textiness.page_type in ("text_like", "uncertain"):
                        ocr_candidates.append(page.page_number)
                    else:
                        # photo_like or blank → skip OCR
                        none_pages.append(page)
                        page_meta[page.page_number]["text_source"] = "none"
                except Exception as e:
                    logger.warning(
                        f"Textiness check failed for page {page.page_number}: {e}"
                    )
                    # Benefit of the doubt: OCR it
                    ocr_candidates.append(page.page_number)
                    page_meta[page.page_number] = {
                        "text_source": "pending",
                        "page_type": "error",
                    }
            render_doc.close()
        elif needs_render:
            # OCR disabled: mark as none
            for page in needs_render:
                none_pages.append(page)
                page_meta[page.page_number] = {
                    "text_source": "none",
                    "page_type": "image_no_ocr",
                }

        # Quality gate on embedded text (document-level)
        if self.auto_quality and self.use_ocr and embedded_pages:
            combined_text = "\n\n".join(p.text for p in embedded_pages if p.text)
            quality = is_text_layer_good(combined_text)
            if not quality.good:
                logger.info(
                    f"Bad embedded text ({quality.reason}), reclassifying "
                    f"{len(embedded_pages)} pages for OCR: {file_path.name}"
                )
                # Move all embedded pages to OCR candidates
                for page in embedded_pages:
                    ocr_candidates.append(page.page_number)
                    page_meta[page.page_number]["text_source"] = "pending"
                    page_meta[page.page_number]["quality_gate"] = quality.reason
                embedded_pages.clear()

        # Run selective OCR on candidate pages
        ocr_pages = {}  # page_number → OCRPageResult
        if ocr_candidates:
            logger.info(
                f"OCR'ing {len(ocr_candidates)}/{result.total_pages} pages: "
                f"{file_path.name}"
            )
            ocr_results = self.ocr_extractor.extract_pages_from_pdf(
                file_path, sorted(ocr_candidates)
            )
            for ocr_page in ocr_results:
                ocr_pages[ocr_page.page_number] = ocr_page
                page_meta[ocr_page.page_number]["text_source"] = "ocr"

        # Determine extraction method label
        has_embedded = len(embedded_pages) > 0
        has_ocr = len(ocr_pages) > 0
        if has_embedded and has_ocr:
            extraction_method = "hybrid"
        elif has_embedded:
            extraction_method = "pdf_text"
        elif has_ocr:
            extraction_method = "ocr_scanned"
        else:
            extraction_method = "pdf_text"  # all blank/photo pages

        # Assemble final page list in order
        final_pages = []
        for page in result.pages:
            pn = page.page_number
            meta = page_meta.get(pn, {})
            text_source = meta.get("text_source", "none")

            if text_source == "embedded":
                final_pages.append({
                    "page_number": pn,
                    "text": page.text,
                    "text_source": "embedded",
                    "word_count": page.word_count,
                    "page_type": meta.get("page_type", "text"),
                })
            elif text_source == "ocr" and pn in ocr_pages:
                ocr_page = ocr_pages[pn]
                final_pages.append({
                    "page_number": pn,
                    "text": ocr_page.text,
                    "text_source": "ocr",
                    "confidence": ocr_page.confidence,
                    "page_type": meta.get("page_type", "text_like"),
                    "dark_ratio": meta.get("dark_ratio"),
                    "line_count": meta.get("line_count"),
                })
            else:
                final_pages.append({
                    "page_number": pn,
                    "text": page.text,  # may be empty or minimal
                    "text_source": "none",
                    "word_count": page.word_count,
                    "page_type": meta.get("page_type", "blank"),
                    "dark_ratio": meta.get("dark_ratio"),
                    "line_count": meta.get("line_count"),
                })

        full_text = "\n\n".join(p["text"] for p in final_pages if p["text"])

        pages_embedded = sum(1 for p in final_pages if p["text_source"] == "embedded")
        pages_ocr = sum(1 for p in final_pages if p["text_source"] == "ocr")
        pages_none = sum(1 for p in final_pages if p["text_source"] == "none")

        logger.info(
            f"{file_path.name}: {extraction_method} "
            f"({pages_embedded} embedded, {pages_ocr} ocr, {pages_none} none)"
        )

        return {
            "file_path": str(file_path),
            "file_type": "pdf",
            "extraction_method": extraction_method,
            "total_pages": result.total_pages,
            "pages_embedded": pages_embedded,
            "pages_ocr": pages_ocr,
            "pages_none": pages_none,
            "pages": final_pages,
            "full_text": full_text,
        }

    def _extract_image(self, file_path: Path) -> dict:
        """Extract text from image using OCR."""
        text = self.ocr_extractor.extract_from_image(file_path)

        return {
            "file_path": str(file_path),
            "file_type": "image",
            "extraction_method": "ocr",
            "text": text,
            "full_text": text
        }

    def _extract_audio(self, file_path: Path) -> dict:
        """Extract text from audio/video using Whisper."""
        result = self.audio_extractor.transcribe(file_path)

        return {
            "file_path": str(file_path),
            "file_type": "audio",
            "extraction_method": "whisper",
            "duration_seconds": result.duration_seconds,
            "language": result.language,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text
                }
                for s in result.segments
            ],
            "full_text": result.full_text
        }

    def _extract_docx(self, file_path: Path) -> dict:
        """Extract text from Word document."""
        from docx import Document

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)

        return {
            "file_path": str(file_path),
            "file_type": "docx",
            "extraction_method": "python-docx",
            "paragraphs": paragraphs,
            "full_text": full_text
        }

    def _extract_text(self, file_path: Path) -> dict:
        """Extract text from plain text file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        return {
            "file_path": str(file_path),
            "file_type": "text",
            "extraction_method": "direct",
            "full_text": text
        }

    def _save_result(self, input_path: Path, result: dict):
        """Save extraction result to JSON."""
        output_path = self.get_output_path(input_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved: {output_path}")

    def process_directory(self, directory: Path, force: bool = False):
        """Process all supported files in a directory.

        Args:
            directory: Directory to process.
            force: Re-extract even if already processed.
        """
        directory = Path(directory)

        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")

        # Find all supported files
        files = []
        for ext in ALL_SUPPORTED:
            files.extend(directory.glob(f"**/*{ext}"))
            files.extend(directory.glob(f"**/*{ext.upper()}"))

        files = sorted(set(files))
        logger.info(f"Found {len(files)} supported files in {directory}")

        # Process files
        successful = 0
        failed = 0
        skipped = 0

        for file_path in tqdm(files, desc="Extracting"):
            if not force and self.is_processed(file_path):
                skipped += 1
                continue

            result = self.extract_from_file(file_path, force=force)
            if result:
                successful += 1
            else:
                failed += 1

        logger.info(f"Extraction complete: {successful} successful, {failed} failed, {skipped} skipped")

    def cleanup(self):
        """Clean up resources."""
        if self._ocr_extractor:
            self._ocr_extractor.cleanup()
        if self._audio_extractor:
            self._audio_extractor.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Extract text from files")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("--output", "-o", default="data/processed/text",
                       help="Output directory for extracted text")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Re-extract even if already processed")
    parser.add_argument("--no-ocr", action="store_true",
                       help="Disable OCR for scanned documents")
    parser.add_argument("--force-ocr", action="store_true",
                       help="Always use OCR for PDFs, ignoring embedded text (for garbage OCR layers)")
    parser.add_argument("--no-auto-quality", action="store_true",
                       help="Disable automatic text quality check (skip quality gate)")
    parser.add_argument("--file", action="store_true",
                       help="Process a single file instead of directory")
    args = parser.parse_args()

    extractor = TextExtractor(
        output_dir=args.output,
        use_ocr=not args.no_ocr,
        force_ocr=args.force_ocr,
        auto_quality=not args.no_auto_quality,
    )

    try:
        input_path = Path(args.input)

        if args.file or input_path.is_file():
            result = extractor.extract_from_file(input_path, force=args.force)
            if result:
                print(f"Extracted {result['file_type']} using {result['extraction_method']}")
                print(f"Text length: {len(result['full_text'])} characters")
        else:
            extractor.process_directory(input_path, force=args.force)

    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()
