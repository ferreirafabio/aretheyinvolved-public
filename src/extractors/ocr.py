"""OCR extraction module using LightOnOCR-2.

LightOnOCR-2-1B is a state-of-the-art document intelligence model that is:
- 3x smaller than DeepSeekOCR
- 1.73x faster than DeepSeekOCR
- State-of-the-art on OlmOCR benchmark
- Apache 2.0 licensed (open weights)

Hugging Face: https://huggingface.co/lightonai/LightOnOCR-2-1B
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import torch
from loguru import logger
from PIL import Image
from pdf2image import convert_from_path
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor


@dataclass
class OCRPageResult:
    """Result of OCR for a single page/image."""
    page_number: int
    text: str
    confidence: float
    image_path: str | None = None


@dataclass
class OCRResult:
    """Result of OCR extraction."""
    file_path: str
    total_pages: int
    pages: list[OCRPageResult]
    model_name: str

    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(p.text for p in self.pages if p.text)


class OCRExtractor:
    """Extract text from images and scanned PDFs using LightOnOCR-2.

    This extractor uses the LightOnOCR-2-1B model which provides
    state-of-the-art OCR performance with a small model size.
    """

    MODEL_NAME = "lightonai/LightOnOCR-2-1B"

    def __init__(self,
                 model_name: str | None = None,
                 device: str | None = None,
                 batch_size: int = 4):
        """Initialize OCR extractor.

        Args:
            model_name: Hugging Face model name. Defaults to LightOnOCR-2-1B.
            device: Device to use ('cuda', 'cpu', or None for auto).
            batch_size: Number of images to process at once.
        """
        self.model_name = model_name or self.MODEL_NAME
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Set dtype based on device
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        logger.info(f"Initializing OCR with {self.model_name} on {self.device}")

        self._processor = None
        self._model = None

    def _load_model(self):
        """Lazy load the model and processor."""
        if self._model is None:
            logger.info(f"Loading OCR model: {self.model_name}")

            self._processor = LightOnOcrProcessor.from_pretrained(
                self.model_name
            )

            self._model = LightOnOcrForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype
            ).to(self.device)

            self._model.eval()

            logger.info("OCR model loaded successfully")

    def extract_from_image(self, image: Image.Image | str | Path) -> str:
        """Extract text from a single image.

        Args:
            image: PIL Image, or path to image file.

        Returns:
            Extracted text.
        """
        self._load_model()

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Create conversation format as required by LightOnOCR
        conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]

        # Process inputs using chat template
        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move to device with correct dtype
        inputs = {
            k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }

        # Generate text
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False
            )

        # Decode output (skip input tokens)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self._processor.decode(generated_ids, skip_special_tokens=True)

        return text.strip()

    def extract_from_pdf(self, pdf_path: str | Path,
                         dpi: int = 200) -> OCRResult:
        """Extract text from a scanned PDF using OCR.

        Args:
            pdf_path: Path to the PDF file.
            dpi: Resolution for PDF to image conversion.

        Returns:
            OCRResult with extracted text from all pages.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Converting PDF to images: {pdf_path.name}")

        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)
        total_pages = len(images)

        logger.info(f"Processing {total_pages} pages with OCR")

        pages = []
        for i, image in enumerate(images, start=1):
            logger.debug(f"OCR processing page {i}/{total_pages}")

            try:
                text = self.extract_from_image(image)
                confidence = 0.95  # LightOnOCR-2 doesn't provide confidence scores

                pages.append(OCRPageResult(
                    page_number=i,
                    text=text,
                    confidence=confidence
                ))
            except Exception as e:
                logger.error(f"OCR failed for page {i}: {e}")
                pages.append(OCRPageResult(
                    page_number=i,
                    text="",
                    confidence=0.0
                ))

        return OCRResult(
            file_path=str(pdf_path),
            total_pages=total_pages,
            pages=pages,
            model_name=self.model_name
        )

    def extract_pages_from_pdf(
        self,
        pdf_path: str | Path,
        page_numbers: list[int],
        dpi: int = 200,
    ) -> list[OCRPageResult]:
        """OCR only specific pages of a PDF.

        Args:
            pdf_path: Path to the PDF file.
            page_numbers: 1-indexed page numbers to OCR.
            dpi: Resolution for PDF to image conversion.

        Returns:
            List of OCRPageResult for the requested pages.
        """
        pdf_path = Path(pdf_path)
        if not page_numbers:
            return []

        self._load_model()
        results = []

        for page_num in page_numbers:
            logger.debug(f"OCR page {page_num} of {pdf_path.name}")
            try:
                # Convert only this page to image
                images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=page_num,
                    last_page=page_num,
                )
                text = self.extract_from_image(images[0])
                results.append(OCRPageResult(
                    page_number=page_num,
                    text=text,
                    confidence=0.95,
                ))
            except Exception as e:
                logger.error(f"OCR failed for page {page_num}: {e}")
                results.append(OCRPageResult(
                    page_number=page_num,
                    text="",
                    confidence=0.0,
                ))

        return results

    def extract_from_images(self,
                           image_paths: list[str | Path]) -> Generator[OCRPageResult, None, None]:
        """Extract text from multiple images.

        Args:
            image_paths: List of paths to image files.

        Yields:
            OCRPageResult for each image.
        """
        self._load_model()

        for i, img_path in enumerate(image_paths, start=1):
            try:
                text = self.extract_from_image(img_path)
                yield OCRPageResult(
                    page_number=i,
                    text=text,
                    confidence=0.95,
                    image_path=str(img_path)
                )
            except Exception as e:
                logger.error(f"OCR failed for {img_path}: {e}")
                yield OCRPageResult(
                    page_number=i,
                    text="",
                    confidence=0.0,
                    image_path=str(img_path)
                )

    def cleanup(self):
        """Free GPU memory by unloading the model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("OCR model unloaded")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        extractor = OCRExtractor()

        file_path = Path(sys.argv[1])

        if file_path.suffix.lower() == ".pdf":
            result = extractor.extract_from_pdf(file_path)
            print(f"File: {result.file_path}")
            print(f"Pages: {result.total_pages}")
            print(f"Model: {result.model_name}")
            print("\n--- Full Text ---")
            print(result.full_text[:2000])
        else:
            text = extractor.extract_from_image(file_path)
            print(f"File: {file_path}")
            print("\n--- Extracted Text ---")
            print(text)

        extractor.cleanup()
