"""Main pipeline orchestrating all NER + LLM extraction stages.

Pipeline flow (monolithic):
    raw_text → Stage 0: Deterministic Clean → clean_text
                             ↓
              Stage 1: XLM-R NER (high recall)
                             ↓
              [if suspicious] Stage 1.5: LLM Recovery
                             ↓
              Stage 2: LLM Classifier (closed-set)
                             ↓
              Stage 3: Hard Validator (offsets vs raw_text)
                             ↓
              Stage 4: LLM Repair (corrupted spans only)
                             ↓
              names.json (original_text + normalized_name)

Multi-tier split:
    Tier 1 (Tier 1 GPU): Stage 0 + Stage 1 → _ner.json
    Tier 2 (Tier 2 GPU):     Stage 1.5 + Stage 2 + Stage 3 + Stage 4 → _names.json

Stage 1.5 (LLM Recovery) triggers when:
- Document has substantial text (>500 chars) BUT
- NER found very few names (0-2) AND
- Text appears to have OCR artifacts (>2% garbage chars)
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


_PIPELINE_VERSION_CACHE: str | None = None


def _get_pipeline_version() -> str:
    """Get git short SHA as pipeline version marker (cached after first call)."""
    global _PIPELINE_VERSION_CACHE
    if _PIPELINE_VERSION_CACHE is not None:
        return _PIPELINE_VERSION_CACHE
    try:
        _PIPELINE_VERSION_CACHE = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        _PIPELINE_VERSION_CACHE = "unknown"
    return _PIPELINE_VERSION_CACHE

from .deterministic_cleaner import CleaningResult, clean_document, same_length_clean
from .xlmr_extractor import XLMRNERExtractor, MockXLMRExtractor, NERResult, NERSpan, create_ner_extractor
from .llm_classifier import (
    LLMSpanClassifier, MockLLMClassifier, ClassificationResult, BatchClassifyItem, create_classifier
)
from .hard_validator import HardValidator, ValidationResult, ValidatedSpan
from .llm_repair import (
    LLMNameRepairer, MockLLMRepairer, RepairResult, RepairedName, BatchRepairItem, create_repairer
)
from .llm_recovery import (
    LLMNameRecovery, MockLLMRecovery, RecoveryResult, RecoveredSpan,
    is_suspicious_document, should_trigger_recovery, compute_noise_score,
    create_recovery, recovered_to_classified
)
from .shared_model import SharedModelManager


@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""
    # Stage 1: NER
    ner_model: str = "xlm-roberta-large-finetuned-conll03-english"
    ner_confidence_threshold: float = 0.3
    ner_batch_size: int = 32

    # Stage 1.5: LLM Recovery (for suspicious documents)
    enable_recovery: bool = True  # Enable LLM recovery for corrupted docs
    recovery_min_text_length: int = 500  # Min text length to trigger recovery
    recovery_max_ner_names: int = 2  # Max NER names to trigger recovery
    recovery_min_garbage_ratio: float = 0.02  # Min garbage ratio to trigger

    # Stage 2: LLM Classifier
    classifier_model: str = "Qwen/Qwen2.5-32B-Instruct"
    classifier_batch_size: int = 50

    # Stage 4: LLM Repair
    repair_model: str = "Qwen/Qwen2.5-32B-Instruct"
    repair_batch_size: int = 80

    # Inter-document batching (Tier 2)
    inter_batch_size: int = 1  # 1 = sequential (backward compat), >1 = batched

    # LLM backend
    llm_backend: str = "transformers"  # "transformers" or "vllm"

    # General
    use_gpu: bool = True
    mock: bool = False  # Use mock NER (testing only, never in production)
    quantize_4bit: bool = True
    save_intermediate: bool = False  # Save intermediate stage outputs


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    stage0_clean_length: int = 0
    stage0_noise_score: float = 0.0  # OCR noise score (0=clean, 1=corrupted)
    stage1_ner_spans: int = 0
    stage1_person_spans: int = 0
    stage1_5_recovery_triggered: bool = False
    stage1_5_recovered: int = 0
    stage2_classified: int = 0
    stage2_dropped: int = 0
    stage3_validated: int = 0
    stage3_failed: int = 0
    stage3_needs_repair: int = 0
    stage4_repaired: int = 0
    final_names: int = 0


@dataclass
class InterBatchMetrics:
    """Metrics from inter-document batched processing."""
    total_docs: int = 0
    batches_submitted: int = 0
    actual_batch_sizes: list[int] = field(default_factory=list)
    recovery_triggered: int = 0
    classify_parse_failures: int = 0
    repair_parse_failures: int = 0


@dataclass
class NerOnlyResult:
    """Result from Tier 1 (Stage 0 + Stage 1 only).

    Contains all data needed by Tier 2 to continue processing.
    """
    source_file: str
    raw_text: str
    clean_text: str
    page_boundaries: list[int]
    noise_score: float
    clean_text_length: int
    spans: list  # list of NERSpan dicts
    total_spans: int
    person_spans: int
    ner_model: str
    pipeline_version: str


# Tier 2 skip gate thresholds
TIER2_SKIP_MIN_TEXT_LENGTH = 50
TIER2_SKIP_ESCAPE_HATCH_LENGTH = 200


@dataclass
class PipelineResult:
    """Final result from the extraction pipeline."""
    source_file: str
    names: list[RepairedName]
    stats: PipelineStats
    document_type: str | None = None

    # Intermediate results (if save_intermediate=True)
    cleaning_result: CleaningResult | None = None
    ner_result: NERResult | None = None
    recovery_result: RecoveryResult | None = None
    classification_result: ClassificationResult | None = None
    validation_result: ValidationResult | None = None
    repair_result: RepairResult | None = None


class ExtractionPipeline:
    """Main pipeline for NER + LLM name extraction.

    This pipeline implements a 5-stage process:
    - Stage 0: Deterministic cleaning (same-length, preserves offsets)
    - Stage 1: XLM-R NER (high recall person extraction)
    - Stage 2: LLM classification (filter non-persons, assign roles)
    - Stage 3: Hard validation (verify spans exist in raw text)
    - Stage 4: LLM repair (fix OCR errors in corrupted spans)
    """

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()

        # Lazy-loaded components
        self._ner_extractor = None
        self._recovery = None
        self._classifier = None
        self._validator = None
        self._repairer = None

        # Shared LLM model manager (classifier, recovery & repairer share same model)
        self._shared_llm_model = None

    def _get_ner_extractor(self):
        """Get or create NER extractor."""
        if self._ner_extractor is None:
            self._ner_extractor = create_ner_extractor(
                use_gpu=self.config.use_gpu,
                mock=self.config.mock,
                model_name=self.config.ner_model,
                confidence_threshold=self.config.ner_confidence_threshold,
                batch_size=self.config.ner_batch_size,
            )
        return self._ner_extractor

    def _get_shared_llm_model(self):
        """Get or create shared LLM backend.

        Both classifier and repairer use the same Qwen model to avoid
        loading it twice (~4 min saved, ~18GB VRAM saved).

        Returns an LLMBackend instance (transformers or vLLM).
        """
        if self._shared_llm_model is None and self.config.use_gpu:
            import torch
            if torch.cuda.is_available():
                from .llm_backend.factory import create_backend
                self._shared_llm_model = create_backend(
                    backend=self.config.llm_backend,
                    quantize_4bit=self.config.quantize_4bit,
                )
        return self._shared_llm_model

    def _get_recovery(self):
        """Get or create LLM recovery module."""
        if self._recovery is None:
            if not self.config.enable_recovery:
                self._recovery = MockLLMRecovery()
            else:
                shared_model = self._get_shared_llm_model()
                if shared_model:
                    self._recovery = LLMNameRecovery(
                        model_name=self.config.classifier_model,
                        quantize_4bit=self.config.quantize_4bit,
                        shared_model=shared_model
                    )
                else:
                    self._recovery = create_recovery(
                        use_llm=self.config.use_gpu,
                        model_name=self.config.classifier_model,
                        quantize_4bit=self.config.quantize_4bit
                    )
        return self._recovery

    def _get_classifier(self):
        """Get or create LLM classifier."""
        if self._classifier is None:
            shared_model = self._get_shared_llm_model()
            if shared_model:
                # Use shared model
                self._classifier = LLMSpanClassifier(
                    model_name=self.config.classifier_model,
                    quantize_4bit=self.config.quantize_4bit,
                    shared_model=shared_model
                )
            else:
                # Fallback to own model (CPU or mock)
                self._classifier = create_classifier(
                    use_llm=self.config.use_gpu,
                    model_name=self.config.classifier_model,
                    quantize_4bit=self.config.quantize_4bit
                )
        return self._classifier

    def _get_validator(self):
        """Get or create hard validator."""
        if self._validator is None:
            self._validator = HardValidator(strict=True)
        return self._validator

    def _get_repairer(self):
        """Get or create LLM repairer."""
        if self._repairer is None:
            shared_model = self._get_shared_llm_model()
            if shared_model:
                # Use shared model
                self._repairer = LLMNameRepairer(
                    model_name=self.config.repair_model,
                    quantize_4bit=self.config.quantize_4bit,
                    shared_model=shared_model
                )
            else:
                # Fallback to own model (CPU or mock)
                self._repairer = create_repairer(
                    use_llm=self.config.use_gpu,
                    model_name=self.config.repair_model,
                    quantize_4bit=self.config.quantize_4bit
                )
        return self._repairer

    def process_document(self,
                         ocr_data: dict,
                         source_file: str = "") -> PipelineResult:
        """Process a single document through all pipeline stages.

        Args:
            ocr_data: OCR JSON with 'full_text', 'pages', or 'paragraphs'.
            source_file: Source file name for tracking.

        Returns:
            PipelineResult with extracted names and statistics.
        """
        stats = PipelineStats()

        # =========================================
        # Stage 0: Deterministic Cleaning
        # =========================================
        logger.info(f"Stage 0: Deterministic cleaning...")
        cleaning_result = clean_document(ocr_data)
        stats.stage0_clean_length = len(cleaning_result.clean_text)
        stats.stage0_noise_score = compute_noise_score(cleaning_result.raw_text)

        if not cleaning_result.clean_text.strip():
            logger.warning(f"No text found in document: {source_file}")
            return PipelineResult(
                source_file=source_file,
                names=[],
                stats=stats
            )

        logger.info(f"  Cleaned {stats.stage0_clean_length} chars (noise score: {stats.stage0_noise_score:.2f})")

        # =========================================
        # Stage 1: XLM-R NER
        # =========================================
        logger.info(f"Stage 1: XLM-R NER extraction...")
        ner_extractor = self._get_ner_extractor()
        ner_result = ner_extractor.extract_spans(
            text=cleaning_result.clean_text,
            source_file=source_file,
            page_boundaries=cleaning_result.page_boundaries,
            filter_types=['PER']  # Only person entities
        )
        stats.stage1_ner_spans = ner_result.total_spans
        stats.stage1_person_spans = ner_result.person_spans

        logger.info(f"  Found {ner_result.total_spans} spans ({ner_result.person_spans} persons)")

        # =========================================
        # Stage 1.5: LLM Recovery (for suspicious documents)
        # =========================================
        recovery_result = None
        all_spans = list(ner_result.spans)  # Start with NER spans

        # Check if document looks suspicious (few names despite substantial text + noise)
        should_recover = (
            self.config.enable_recovery and
            should_trigger_recovery(
                text_length=stats.stage0_clean_length,
                ner_name_count=ner_result.person_spans,
                noise_score=stats.stage0_noise_score
            )
        )

        if should_recover:
            logger.info(f"Stage 1.5: LLM Recovery (suspicious document)...")
            recovery = self._get_recovery()
            recovery_result = recovery.recover_names(
                raw_text=cleaning_result.raw_text,
                clean_text=cleaning_result.clean_text,
                source_file=source_file,
                page_boundaries=cleaning_result.page_boundaries
            )
            stats.stage1_5_recovery_triggered = True
            stats.stage1_5_recovered = recovery_result.total_recovered

            logger.info(f"  Recovered {recovery_result.total_recovered} additional name candidates")

            # Convert recovered spans to NER span format and merge
            for recovered_span in recovery_result.recovered_spans:
                # Check for duplicates (same position)
                is_duplicate = any(
                    s.start == recovered_span.start and s.end == recovered_span.end
                    for s in all_spans
                )
                if not is_duplicate:
                    # Create NER-compatible span
                    from .xlmr_extractor import NERSpan
                    all_spans.append(NERSpan(
                        text=recovered_span.text,
                        start=recovered_span.start,
                        end=recovered_span.end,
                        entity_type="PER",
                        confidence=recovered_span.confidence,
                        page_number=recovered_span.page_number
                    ))

        # Update NER result with merged spans
        ner_result.spans = all_spans

        if not all_spans:
            logger.warning(f"No spans found (NER + Recovery): {source_file}")
            return PipelineResult(
                source_file=source_file,
                names=[],
                stats=stats,
                cleaning_result=cleaning_result if self.config.save_intermediate else None,
                ner_result=ner_result if self.config.save_intermediate else None,
                recovery_result=recovery_result if self.config.save_intermediate else None
            )

        # =========================================
        # Stage 2: LLM Classification
        # =========================================
        logger.info(f"Stage 2: LLM classification...")
        classifier = self._get_classifier()
        classification_result = classifier.classify_spans(
            document_text=cleaning_result.clean_text,
            spans=ner_result.spans,
            source_file=source_file,
            batch_size=self.config.classifier_batch_size
        )
        stats.stage2_classified = classification_result.total_spans
        stats.stage2_dropped = classification_result.dropped_spans

        logger.info(f"  Classified {classification_result.total_spans}, dropped {classification_result.dropped_spans}")

        # Filter to only person spans that aren't dropped
        person_spans = [c for c in classification_result.classified_spans
                        if c.is_person and not c.drop]

        if not person_spans:
            logger.warning(f"No person spans after classification: {source_file}")
            return PipelineResult(
                source_file=source_file,
                names=[],
                stats=stats,
                document_type=classification_result.document_type,
                cleaning_result=cleaning_result if self.config.save_intermediate else None,
                ner_result=ner_result if self.config.save_intermediate else None,
                classification_result=classification_result if self.config.save_intermediate else None
            )

        # =========================================
        # Stage 3: Hard Validation
        # =========================================
        logger.info(f"Stage 3: Hard validation...")
        validator = self._get_validator()
        validation_result = validator.validate(
            raw_text=cleaning_result.raw_text,  # Validate against RAW text
            classified_spans=person_spans,
            source_file=source_file
        )
        stats.stage3_validated = validation_result.passed
        stats.stage3_failed = validation_result.failed
        stats.stage3_needs_repair = validation_result.needs_repair

        logger.info(f"  Validated {validation_result.passed}, failed {validation_result.failed}, needs repair {validation_result.needs_repair}")

        if not validation_result.validated_spans:
            logger.warning(f"No spans passed validation: {source_file}")
            return PipelineResult(
                source_file=source_file,
                names=[],
                stats=stats,
                document_type=classification_result.document_type,
                cleaning_result=cleaning_result if self.config.save_intermediate else None,
                ner_result=ner_result if self.config.save_intermediate else None,
                classification_result=classification_result if self.config.save_intermediate else None,
                validation_result=validation_result if self.config.save_intermediate else None
            )

        # =========================================
        # Stage 4: LLM Repair
        # =========================================
        logger.info(f"Stage 4: LLM repair...")
        repairer = self._get_repairer()
        repair_result = repairer.repair_names(
            validated_spans=validation_result.validated_spans,
            source_file=source_file,
            batch_size=self.config.repair_batch_size
        )
        stats.stage4_repaired = repair_result.repaired_count
        stats.final_names = repair_result.total_names

        logger.info(f"  Final: {repair_result.total_names} names ({repair_result.repaired_count} repaired)")

        return PipelineResult(
            source_file=source_file,
            names=repair_result.repaired_names,
            stats=stats,
            document_type=classification_result.document_type,
            cleaning_result=cleaning_result if self.config.save_intermediate else None,
            ner_result=ner_result if self.config.save_intermediate else None,
            classification_result=classification_result if self.config.save_intermediate else None,
            validation_result=validation_result if self.config.save_intermediate else None,
            repair_result=repair_result if self.config.save_intermediate else None
        )

    def process_ner_only(self,
                         ocr_data: dict,
                         source_file: str = "") -> NerOnlyResult:
        """Run only Stage 0 + Stage 1 (Tier 1: Tier 1 GPU workload).

        Produces a NerOnlyResult that can be serialized to _ner.json
        and later consumed by process_from_ner().

        Args:
            ocr_data: OCR JSON with 'full_text', 'pages', or 'paragraphs'.
            source_file: Source file name for tracking.

        Returns:
            NerOnlyResult with cleaning + NER data for Tier 2.
        """
        # Stage 0: Deterministic Cleaning
        cleaning_result = clean_document(ocr_data)
        clean_length = len(cleaning_result.clean_text)
        noise_score = compute_noise_score(cleaning_result.raw_text)

        # Stage 1: XLM-R NER
        spans = []
        total_spans = 0
        person_spans = 0

        if cleaning_result.clean_text.strip():
            ner_extractor = self._get_ner_extractor()
            ner_result = ner_extractor.extract_spans(
                text=cleaning_result.clean_text,
                source_file=source_file,
                page_boundaries=cleaning_result.page_boundaries,
                filter_types=['PER']
            )
            total_spans = ner_result.total_spans
            person_spans = ner_result.person_spans
            spans = [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "entity_type": s.entity_type,
                    "confidence": float(s.confidence) if s.confidence is not None else None,
                    "page_number": s.page_number
                }
                for s in ner_result.spans
            ]

        return NerOnlyResult(
            source_file=source_file,
            raw_text=cleaning_result.raw_text,
            clean_text=cleaning_result.clean_text,
            page_boundaries=cleaning_result.page_boundaries or [0],
            noise_score=noise_score,
            clean_text_length=clean_length,
            spans=spans,
            total_spans=total_spans,
            person_spans=person_spans,
            ner_model=self.config.ner_model,
            pipeline_version=_get_pipeline_version(),
        )

    def process_from_ner(self,
                         ner_data: dict,
                         ocr_data: dict,
                         source_file: str = "") -> PipelineResult:
        """Run Stages 1.5-4 from a saved _ner.json (Tier 2: Tier 2 GPU workload).

        Re-runs Stage 0 (deterministic clean) from the original text JSON
        to get the correct raw_text/clean_text pair, then uses spans from
        the _ner.json to continue with LLM stages.

        Args:
            ner_data: Parsed _ner.json data (spans, stats).
            ocr_data: Original text JSON (same format as process_document).
            source_file: Source file name for tracking.

        Returns:
            PipelineResult with extracted names (same as process_document).
        """
        stats = PipelineStats()
        stats.stage0_clean_length = ner_data.get("clean_text_length", 0)
        stats.stage0_noise_score = ner_data.get("noise_score", 0.0)
        stats.stage1_ner_spans = ner_data.get("total_spans", 0)
        stats.stage1_person_spans = ner_data.get("person_spans", 0)

        # Tier 2 skip gate: skip if essentially empty
        # Never skip if clean_text_length >= 200 (escape hatch for noisy
        # narrative docs where XLM-R missed names — let LLM recovery try)
        if (stats.stage1_person_spans == 0
                and stats.stage0_clean_length < TIER2_SKIP_ESCAPE_HATCH_LENGTH):
            logger.info(
                f"Tier 2 skip gate: len={stats.stage0_clean_length}, "
                f"spans=0 -> skipping LLM stages"
            )
            return PipelineResult(
                source_file=source_file,
                names=[],
                stats=stats,
            )

        # Re-run Stage 0 to get correct raw_text/clean_text pair
        # (cheap CPU operation, avoids storing raw_text in _ner.json)
        cleaning_result = clean_document(ocr_data)
        raw_text = cleaning_result.raw_text
        clean_text = cleaning_result.clean_text
        page_boundaries = cleaning_result.page_boundaries

        # Reconstruct NER spans from _ner.json
        from .xlmr_extractor import NERSpan
        all_spans = [
            NERSpan(
                text=s["text"],
                start=s["start"],
                end=s["end"],
                entity_type=s["entity_type"],
                confidence=s.get("confidence", 0.0),
                page_number=s.get("page_number"),
            )
            for s in ner_data.get("spans", [])
        ]

        # =========================================
        # Stage 1.5: LLM Recovery (for suspicious documents)
        # =========================================
        recovery_result = None

        should_recover = (
            self.config.enable_recovery and
            should_trigger_recovery(
                text_length=stats.stage0_clean_length,
                ner_name_count=stats.stage1_person_spans,
                noise_score=stats.stage0_noise_score,
            )
        )

        if should_recover:
            logger.info("Stage 1.5: LLM Recovery (suspicious document)...")
            recovery = self._get_recovery()
            recovery_result = recovery.recover_names(
                raw_text=raw_text,
                clean_text=clean_text,
                source_file=source_file,
                page_boundaries=page_boundaries,
            )
            stats.stage1_5_recovery_triggered = True
            stats.stage1_5_recovered = recovery_result.total_recovered
            logger.info(f"  Recovered {recovery_result.total_recovered} additional name candidates")

            for recovered_span in recovery_result.recovered_spans:
                is_duplicate = any(
                    s.start == recovered_span.start and s.end == recovered_span.end
                    for s in all_spans
                )
                if not is_duplicate:
                    all_spans.append(NERSpan(
                        text=recovered_span.text,
                        start=recovered_span.start,
                        end=recovered_span.end,
                        entity_type="PER",
                        confidence=recovered_span.confidence,
                        page_number=recovered_span.page_number,
                    ))

        if not all_spans:
            logger.warning(f"No spans found (NER + Recovery): {source_file}")
            return PipelineResult(
                source_file=source_file,
                names=[],
                stats=stats,
            )

        # =========================================
        # Stage 2: LLM Classification
        # =========================================
        logger.info("Stage 2: LLM classification...")
        classifier = self._get_classifier()

        # Build a NERResult to pass to classifier
        ner_result = NERResult(
            source_file=source_file,
            spans=all_spans,
            total_spans=len(all_spans),
            person_spans=sum(1 for s in all_spans if s.entity_type == "PER"),
        )

        classification_result = classifier.classify_spans(
            document_text=clean_text,
            spans=ner_result.spans,
            source_file=source_file,
            batch_size=self.config.classifier_batch_size,
        )
        stats.stage2_classified = classification_result.total_spans
        stats.stage2_dropped = classification_result.dropped_spans

        logger.info(f"  Classified {classification_result.total_spans}, dropped {classification_result.dropped_spans}")

        person_spans = [c for c in classification_result.classified_spans
                        if c.is_person and not c.drop]

        if not person_spans:
            logger.warning(f"No person spans after classification: {source_file}")
            return PipelineResult(
                source_file=source_file,
                names=[],
                stats=stats,
                document_type=classification_result.document_type,
            )

        # =========================================
        # Stage 3: Hard Validation
        # =========================================
        logger.info("Stage 3: Hard validation...")
        validator = self._get_validator()
        validation_result = validator.validate(
            raw_text=raw_text,
            classified_spans=person_spans,
            source_file=source_file,
        )
        stats.stage3_validated = validation_result.passed
        stats.stage3_failed = validation_result.failed
        stats.stage3_needs_repair = validation_result.needs_repair

        logger.info(f"  Validated {validation_result.passed}, failed {validation_result.failed}, needs repair {validation_result.needs_repair}")

        if not validation_result.validated_spans:
            logger.warning(f"No spans passed validation: {source_file}")
            return PipelineResult(
                source_file=source_file,
                names=[],
                stats=stats,
                document_type=classification_result.document_type,
            )

        # =========================================
        # Stage 4: LLM Repair
        # =========================================
        logger.info("Stage 4: LLM repair...")
        repairer = self._get_repairer()
        repair_result = repairer.repair_names(
            validated_spans=validation_result.validated_spans,
            source_file=source_file,
            batch_size=self.config.repair_batch_size,
        )
        stats.stage4_repaired = repair_result.repaired_count
        stats.final_names = repair_result.total_names

        logger.info(f"  Final: {repair_result.total_names} names ({repair_result.repaired_count} repaired)")

        return PipelineResult(
            source_file=source_file,
            names=repair_result.repaired_names,
            stats=stats,
            document_type=classification_result.document_type,
        )

    def process_batch_from_ner(
        self,
        batch_inputs: list[tuple[dict, dict, str]],
        inter_batch_size: int = 8,
    ) -> list[PipelineResult]:
        """Process multiple documents through Stages 1.5-4 with inter-doc batching.

        Stages 0 and 1.5 run per-file (cheap CPU + rare gated LLM).
        Stages 2 and 4 run batched across documents for GPU efficiency.
        Stage 3 runs per-file (free CPU offset comparison).

        Args:
            batch_inputs: List of (ner_data, ocr_data, source_file) tuples.
            inter_batch_size: Max prompts per batched generate() call.

        Returns:
            List of PipelineResult in the same order as batch_inputs.
        """
        if not batch_inputs:
            return []

        metrics = InterBatchMetrics(total_docs=len(batch_inputs))

        # === Phase 1: Per-file Stage 0 re-clean + Stage 1.5 recovery ===
        # Each doc gets its raw_text/clean_text and NER spans reconstructed.
        # Recovery (1.5) is sequential (rare, uses do_sample=True).
        doc_data: list[dict | None] = [None] * len(batch_inputs)
        results: list[PipelineResult | None] = [None] * len(batch_inputs)

        for i, (ner_data, ocr_data, source_file) in enumerate(batch_inputs):
            stats = PipelineStats()
            stats.stage0_clean_length = ner_data.get("clean_text_length", 0)
            stats.stage0_noise_score = ner_data.get("noise_score", 0.0)
            stats.stage1_ner_spans = ner_data.get("total_spans", 0)
            stats.stage1_person_spans = ner_data.get("person_spans", 0)

            # Tier 2 skip gate
            if (stats.stage1_person_spans == 0
                    and stats.stage0_clean_length < TIER2_SKIP_ESCAPE_HATCH_LENGTH):
                results[i] = PipelineResult(
                    source_file=source_file, names=[], stats=stats,
                )
                continue

            # Re-run Stage 0 (cheap CPU)
            cleaning_result = clean_document(ocr_data)
            raw_text = cleaning_result.raw_text
            clean_text = cleaning_result.clean_text
            page_boundaries = cleaning_result.page_boundaries

            # Reconstruct NER spans
            all_spans = [
                NERSpan(
                    text=s["text"], start=s["start"], end=s["end"],
                    entity_type=s["entity_type"],
                    confidence=s.get("confidence", 0.0),
                    page_number=s.get("page_number"),
                )
                for s in ner_data.get("spans", [])
            ]

            # Stage 1.5: LLM Recovery (sequential, rare ~5%)
            should_recover = (
                self.config.enable_recovery and
                should_trigger_recovery(
                    text_length=stats.stage0_clean_length,
                    ner_name_count=stats.stage1_person_spans,
                    noise_score=stats.stage0_noise_score,
                )
            )

            if should_recover:
                metrics.recovery_triggered += 1
                recovery = self._get_recovery()
                recovery_result = recovery.recover_names(
                    raw_text=raw_text, clean_text=clean_text,
                    source_file=source_file,
                    page_boundaries=page_boundaries,
                )
                stats.stage1_5_recovery_triggered = True
                stats.stage1_5_recovered = recovery_result.total_recovered

                for recovered_span in recovery_result.recovered_spans:
                    is_duplicate = any(
                        s.start == recovered_span.start and s.end == recovered_span.end
                        for s in all_spans
                    )
                    if not is_duplicate:
                        all_spans.append(NERSpan(
                            text=recovered_span.text,
                            start=recovered_span.start,
                            end=recovered_span.end,
                            entity_type="PER",
                            confidence=recovered_span.confidence,
                            page_number=recovered_span.page_number,
                        ))

            if not all_spans:
                results[i] = PipelineResult(
                    source_file=source_file, names=[], stats=stats,
                )
                continue

            doc_data[i] = {
                "raw_text": raw_text,
                "clean_text": clean_text,
                "all_spans": all_spans,
                "stats": stats,
                "source_file": source_file,
            }

        # === Phase 2: Batched Stage 2 Classification ===
        classify_items = []
        classify_idx_map = []  # maps classify_items index -> doc index

        for i, dd in enumerate(doc_data):
            if dd is not None:
                classify_items.append(BatchClassifyItem(
                    document_text=dd["clean_text"],
                    spans=dd["all_spans"],
                    source_file=dd["source_file"],
                ))
                classify_idx_map.append(i)

        if classify_items:
            classifier = self._get_classifier()
            classification_results = classifier.classify_spans_batch(
                classify_items, inter_batch_size=inter_batch_size,
                classifier_batch_size=self.config.classifier_batch_size,
            )
            metrics.batches_submitted += 1
            metrics.actual_batch_sizes.append(len(classify_items))
        else:
            classification_results = {}

        # === Phase 3: Per-file Stage 3 Validation + collect repair items ===
        validator = self._get_validator()
        repair_items = []
        repair_idx_map = []  # maps repair_items index -> doc index

        for ci, doc_idx in enumerate(classify_idx_map):
            dd = doc_data[doc_idx]
            stats = dd["stats"]
            source_file = dd["source_file"]
            raw_text = dd["raw_text"]

            cr = classification_results.get(source_file)
            if cr is None:
                results[doc_idx] = PipelineResult(
                    source_file=source_file, names=[], stats=stats,
                )
                continue

            stats.stage2_classified = cr.total_spans
            stats.stage2_dropped = cr.dropped_spans

            person_spans = [c for c in cr.classified_spans if c.is_person and not c.drop]
            if not person_spans:
                results[doc_idx] = PipelineResult(
                    source_file=source_file, names=[], stats=stats,
                    document_type=cr.document_type,
                )
                continue

            # Stage 3: Hard Validation (per-file, free CPU)
            validation_result = validator.validate(
                raw_text=raw_text,
                classified_spans=person_spans,
                source_file=source_file,
            )
            stats.stage3_validated = validation_result.passed
            stats.stage3_failed = validation_result.failed
            stats.stage3_needs_repair = validation_result.needs_repair

            if not validation_result.validated_spans:
                results[doc_idx] = PipelineResult(
                    source_file=source_file, names=[], stats=stats,
                    document_type=cr.document_type,
                )
                continue

            # Store for batched repair
            repair_items.append(BatchRepairItem(
                validated_spans=validation_result.validated_spans,
                source_file=source_file,
            ))
            repair_idx_map.append(doc_idx)
            # Store document_type for final result
            dd["document_type"] = cr.document_type

        # === Phase 4: Batched Stage 4 Repair ===
        if repair_items:
            repairer = self._get_repairer()
            repair_results = repairer.repair_names_batch(
                repair_items, inter_batch_size=inter_batch_size,
                repair_batch_size=self.config.repair_batch_size,
            )
            metrics.batches_submitted += 1
            metrics.actual_batch_sizes.append(len(repair_items))
        else:
            repair_results = {}

        # === Assemble final results ===
        for ri, doc_idx in enumerate(repair_idx_map):
            dd = doc_data[doc_idx]
            stats = dd["stats"]
            source_file = dd["source_file"]

            rr = repair_results.get(source_file)
            if rr is None:
                results[doc_idx] = PipelineResult(
                    source_file=source_file, names=[], stats=stats,
                    document_type=dd.get("document_type"),
                )
                continue

            stats.stage4_repaired = rr.repaired_count
            stats.final_names = rr.total_names

            results[doc_idx] = PipelineResult(
                source_file=source_file,
                names=rr.repaired_names,
                stats=stats,
                document_type=dd.get("document_type"),
            )

        # Fill any remaining None results (shouldn't happen but safety net)
        for i in range(len(results)):
            if results[i] is None:
                results[i] = PipelineResult(
                    source_file=batch_inputs[i][2], names=[],
                    stats=PipelineStats(),
                )

        # Log batch metrics
        shared_model = self._get_shared_llm_model()
        gpu_stats = ""
        if shared_model and hasattr(shared_model, 'batch_stats'):
            bs = shared_model.batch_stats
            gpu_stats = (
                f" gpu_batches={bs.total_batches} gpu_prompts={bs.total_prompts} "
                f"oom_splits={bs.oom_splits} padding_waste={bs.padding_waste_tokens}"
            )
        logger.info(
            f"Inter-batch metrics: docs={metrics.total_docs} "
            f"recovery_triggered={metrics.recovery_triggered} "
            f"batch_sizes={metrics.actual_batch_sizes}{gpu_stats}"
        )

        return results

    def process_text(self,
                     text: str,
                     source_file: str = "") -> PipelineResult:
        """Process raw text through the pipeline.

        Convenience method for processing text directly.

        Args:
            text: Raw document text.
            source_file: Source file name.

        Returns:
            PipelineResult.
        """
        ocr_data = {"full_text": text}
        return self.process_document(ocr_data, source_file)

    def cleanup(self):
        """Free GPU memory from all components."""
        if self._ner_extractor is not None:
            self._ner_extractor.cleanup()
            self._ner_extractor = None
            logger.info("NER model unloaded")

        # Clear references to classifier, recovery and repairer first
        self._classifier = None
        self._recovery = None
        self._repairer = None

        # Then cleanup the shared LLM model (they no longer reference it)
        if self._shared_llm_model is not None:
            self._shared_llm_model.cleanup()
            self._shared_llm_model = None
            logger.info("Shared LLM model unloaded")

        logger.info("Pipeline cleanup complete")


def save_pipeline_result(result: PipelineResult,
                         output_dir: Path,
                         save_intermediate: bool = False) -> dict[str, Path]:
    """Save pipeline result to JSON files.

    Args:
        result: Pipeline result to save.
        output_dir: Directory to save files.
        save_intermediate: Whether to save intermediate stage outputs.

    Returns:
        Dict mapping output type to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get base filename
    source_stem = Path(result.source_file).stem if result.source_file else "document"

    saved_files = {}

    # Save final names
    names_path = output_dir / f"{source_stem}_names.json"
    names_data = {
        "source_file": result.source_file,
        "document_type": result.document_type,
        "pipeline_version": _get_pipeline_version(),
        "total_names": len(result.names),
        "stats": {
            "stage0_clean_length": result.stats.stage0_clean_length,
            "stage0_noise_score": result.stats.stage0_noise_score,
            "stage1_ner_spans": result.stats.stage1_ner_spans,
            "stage1_person_spans": result.stats.stage1_person_spans,
            "stage1_5_recovery_triggered": result.stats.stage1_5_recovery_triggered,
            "stage1_5_recovered": result.stats.stage1_5_recovered,
            "stage2_classified": result.stats.stage2_classified,
            "stage2_dropped": result.stats.stage2_dropped,
            "stage3_validated": result.stats.stage3_validated,
            "stage3_failed": result.stats.stage3_failed,
            "stage3_needs_repair": result.stats.stage3_needs_repair,
            "stage4_repaired": result.stats.stage4_repaired,
            "final_names": result.stats.final_names
        },
        "names": [
            {
                "original_text": n.original_text,
                "normalized_name": n.normalized_name,
                "start": n.start,
                "end": n.end,
                "role": n.role,
                "all_roles": n.all_roles,
                "confidence": float(n.confidence) if n.confidence is not None else None,
                "page_number": n.page_number,
                "was_repaired": n.was_repaired
            }
            for n in result.names
        ]
    }

    # Atomic write: temp file + rename (prevents corrupt JSON on preemption)
    tmp_path = names_path.with_suffix(".tmp.json")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(names_data, f, ensure_ascii=False)
    os.rename(tmp_path, names_path)
    saved_files['names'] = names_path

    # Save intermediate results if requested
    if save_intermediate:
        if result.cleaning_result:
            clean_path = output_dir / f"{source_stem}_clean.json"
            clean_data = {
                "raw_length": len(result.cleaning_result.raw_text),
                "clean_length": len(result.cleaning_result.clean_text),
                "page_boundaries": result.cleaning_result.page_boundaries,
                "clean_text": result.cleaning_result.clean_text
            }
            with open(clean_path, 'w', encoding='utf-8') as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)
            saved_files['clean'] = clean_path

        if result.ner_result:
            ner_path = output_dir / f"{source_stem}_ner.json"
            ner_data = {
                "model": result.ner_result.model_name,
                "total_spans": result.ner_result.total_spans,
                "person_spans": result.ner_result.person_spans,
                "spans": [
                    {
                        "text": s.text,
                        "start": s.start,
                        "end": s.end,
                        "entity_type": s.entity_type,
                        "confidence": float(s.confidence) if s.confidence is not None else None,
                        "page_number": s.page_number
                    }
                    for s in result.ner_result.spans
                ]
            }
            with open(ner_path, 'w', encoding='utf-8') as f:
                json.dump(ner_data, f, indent=2, ensure_ascii=False)
            saved_files['ner'] = ner_path

        if result.classification_result:
            classification_path = output_dir / f"{source_stem}_classification.json"
            classification_data = {
                "source_file": result.classification_result.source_file,
                "document_type": result.classification_result.document_type,
                "total_spans": result.classification_result.total_spans,
                "person_spans": result.classification_result.person_spans,
                "dropped_spans": result.classification_result.dropped_spans,
                "classified_spans": [
                    {
                        "text": c.span.text,
                        "start": c.span.start,
                        "end": c.span.end,
                        "entity_type": c.span.entity_type,
                        "ner_confidence": float(c.span.confidence) if c.span.confidence is not None else None,
                        "page_number": c.span.page_number,
                        "is_person": c.is_person,
                        "role": c.role,
                        "all_roles": c.all_roles,
                        "drop": c.drop,
                        "drop_reason": c.drop_reason,
                        "classification_confidence": c.classification_confidence
                    }
                    for c in result.classification_result.classified_spans
                ]
            }
            with open(classification_path, 'w', encoding='utf-8') as f:
                json.dump(classification_data, f, indent=2, ensure_ascii=False)
            saved_files['classification'] = classification_path

    return saved_files


def save_ner_only_result(result: NerOnlyResult,
                          output_dir: Path) -> Path:
    """Save Tier 1 NER-only result to a _ner.json file.

    The output includes all data needed by Tier 2 (process_from_ner):
    spans, noise score, clean text, page boundaries, and raw text reference.

    Args:
        result: NerOnlyResult from process_ner_only().
        output_dir: Directory to save to.

    Returns:
        Path to the saved _ner.json file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stem = Path(result.source_file).stem if result.source_file else "document"
    ner_path = output_dir / f"{source_stem}_ner.json"

    ner_data = {
        "source_file": result.source_file,
        "pipeline_version": result.pipeline_version,
        "ner_model": result.ner_model,
        "clean_text_length": result.clean_text_length,
        "noise_score": result.noise_score,
        "page_boundaries": result.page_boundaries,
        "total_spans": result.total_spans,
        "person_spans": result.person_spans,
        "spans": result.spans,
    }

    # Atomic write: write to temp file then rename (prevents corrupt JSON on preemption)
    tmp_path = ner_path.with_suffix(".tmp.json")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(ner_data, f, ensure_ascii=False)
    os.rename(tmp_path, ner_path)

    # Append to manifest (optional acceleration layer for downstream queries)
    _append_manifest(output_dir, source_stem, result)

    return ner_path


def _append_manifest(output_dir: Path, stem: str, result: NerOnlyResult) -> None:
    """Append a single entry to _ner_manifest.jsonl.

    Each line is a compact JSON with skip-gate fields so downstream tools
    (Tier 2 discovery, diagnostic scripts, monitoring) can query file
    characteristics without parsing every _ner.json.

    Safe for sequential SLURM workers writing to the same directory.
    Duplicates (from reruns) are harmless — readers use last-one-wins.
    """
    manifest_path = output_dir / "_ner_manifest.jsonl"
    entry = json.dumps({
        "stem": stem,
        "person_spans": result.person_spans,
        "total_spans": result.total_spans,
        "clean_text_length": result.clean_text_length,
        "noise_score": round(result.noise_score, 4) if result.noise_score else 0.0,
    }, ensure_ascii=False)
    try:
        with open(manifest_path, 'a', encoding='utf-8') as f:
            f.write(entry + '\n')
    except OSError:
        pass  # Non-critical — manifest is an optimization, not required


if __name__ == "__main__":
    # Test the pipeline
    sample_email = """
    From: Jeffrey Epstein <jepstein@mail.com>
    To: Ghislaine Maxwell <gmax@mail.com>
    Cc: Sarah Kellen

    Hi Ghislaine,

    Please confirm with Bill Clinton and Donald Trump about the
    meeting next week. Also, Alan Dershowitz called about the
    legal matter.

    Regards,
    Jeffrey
    """

    # Use mock components (no GPU)
    config = PipelineConfig(use_gpu=False, mock=True)
    pipeline = ExtractionPipeline(config)

    try:
        result = pipeline.process_text(sample_email, source_file="test_email.pdf")

        print(f"\n{'='*50}")
        print(f"Pipeline Results for: {result.source_file}")
        print(f"{'='*50}")
        print(f"Document type: {result.document_type}")
        print(f"\nStats:")
        print(f"  Stage 0 (clean): {result.stats.stage0_clean_length} chars")
        print(f"  Stage 1 (NER): {result.stats.stage1_ner_spans} spans, {result.stats.stage1_person_spans} persons")
        print(f"  Stage 2 (classify): {result.stats.stage2_classified} classified, {result.stats.stage2_dropped} dropped")
        print(f"  Stage 3 (validate): {result.stats.stage3_validated} passed, {result.stats.stage3_failed} failed, {result.stats.stage3_needs_repair} need repair")
        print(f"  Stage 4 (repair): {result.stats.stage4_repaired} repaired")
        print(f"  Final: {result.stats.final_names} names")

        print(f"\nExtracted names:")
        for name in result.names:
            repair_marker = "[R]" if name.was_repaired else "   "
            print(f"  {repair_marker} '{name.normalized_name}' ({name.role}) @ {name.start}-{name.end}")
            if name.original_text != name.normalized_name:
                print(f"       (original: '{name.original_text}')")

    finally:
        pipeline.cleanup()
