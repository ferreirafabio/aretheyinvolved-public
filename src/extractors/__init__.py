"""Text extraction modules for different file types."""

from .pdf import PDFExtractor

try:
    from .ocr import OCRExtractor
except ImportError:
    OCRExtractor = None

try:
    from .audio import AudioExtractor
except ImportError:
    AudioExtractor = None

__all__ = ["PDFExtractor", "OCRExtractor", "AudioExtractor"]
