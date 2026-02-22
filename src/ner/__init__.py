"""Named Entity Recognition modules.

v2 pipeline: NER + LLM multi-stage extraction.
Core modules (deterministic_cleaner, hard_validator) have no heavy dependencies.
Pipeline modules require: torch, transformers.
"""

# Core v2 modules with no heavy dependencies
from .deterministic_cleaner import (
    same_length_clean,
    clean_document,
    CleaningResult,
    get_page_boundaries,
    get_page_number,
)

# LLM Validator (HeuristicValidator used by cleanup_garbage_names.py)
try:
    from .llm_validator import (
        HeuristicValidator,
        LlamaValidator,
        MockValidator,
        ValidationResult,
        create_validator,
    )
except ImportError:
    HeuristicValidator = None
    LlamaValidator = None
    MockValidator = None
    ValidationResult = None
    create_validator = None

# Pipeline v2 modules (require torch)
try:
    from .xlmr_extractor import (
        XLMRNERExtractor,
        MockXLMRExtractor,
        NERSpan,
        NERResult,
        create_ner_extractor,
    )
    from .llm_classifier import (
        LLMSpanClassifier,
        MockLLMClassifier,
        ClassifiedSpan,
        ClassificationResult,
        create_classifier,
    )
    from .hard_validator import (
        HardValidator,
        ValidatedSpan,
        ValidationResult as HardValidationResult,
        ValidationError,
        needs_repair,
        validate_span,
        validate_spans_strict,
    )
    from .llm_repair import (
        LLMNameRepairer,
        MockLLMRepairer,
        RepairedName,
        RepairResult,
        create_repairer,
        basic_normalize,
    )
    from .llm_recovery import (
        LLMNameRecovery,
        MockLLMRecovery,
        RecoveredSpan,
        RecoveryResult,
        is_suspicious_document,
        should_trigger_recovery,
        compute_noise_score,
        create_recovery,
    )
    from .pipeline import (
        ExtractionPipeline,
        NerOnlyResult,
        PipelineConfig,
        PipelineResult,
        PipelineStats,
        save_ner_only_result,
        save_pipeline_result,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Pipeline v2 modules unavailable (missing torch): {e}")
    XLMRNERExtractor = None
    MockXLMRExtractor = None
    NERSpan = None
    NERResult = None
    create_ner_extractor = None
    LLMSpanClassifier = None
    MockLLMClassifier = None
    ClassifiedSpan = None
    ClassificationResult = None
    create_classifier = None
    HardValidator = None
    ValidatedSpan = None
    HardValidationResult = None
    ValidationError = None
    needs_repair = None
    validate_span = None
    validate_spans_strict = None
    LLMNameRepairer = None
    MockLLMRepairer = None
    RepairedName = None
    RepairResult = None
    create_repairer = None
    basic_normalize = None
    LLMNameRecovery = None
    MockLLMRecovery = None
    RecoveredSpan = None
    RecoveryResult = None
    is_suspicious_document = None
    should_trigger_recovery = None
    compute_noise_score = None
    create_recovery = None
    ExtractionPipeline = None
    PipelineConfig = None
    PipelineResult = None
    PipelineStats = None
    save_pipeline_result = None

__all__ = [
    # Core v2 (no heavy deps)
    "same_length_clean",
    "clean_document",
    "CleaningResult",
    "get_page_boundaries",
    "get_page_number",
    # LLM Validator
    "HeuristicValidator",
    "LlamaValidator",
    "MockValidator",
    "ValidationResult",
    "create_validator",
    # Pipeline v2 - XLM-R NER (Stage 1)
    "XLMRNERExtractor",
    "MockXLMRExtractor",
    "NERSpan",
    "NERResult",
    "create_ner_extractor",
    # Pipeline v2 - LLM Classifier (Stage 2)
    "LLMSpanClassifier",
    "MockLLMClassifier",
    "ClassifiedSpan",
    "ClassificationResult",
    "create_classifier",
    # Pipeline v2 - Hard Validator (Stage 3)
    "HardValidator",
    "ValidatedSpan",
    "HardValidationResult",
    "ValidationError",
    "needs_repair",
    "validate_span",
    "validate_spans_strict",
    # Pipeline v2 - LLM Repair (Stage 4)
    "LLMNameRepairer",
    "MockLLMRepairer",
    "RepairedName",
    "RepairResult",
    "create_repairer",
    "basic_normalize",
    # Pipeline v2 - LLM Recovery (Stage 1.5)
    "LLMNameRecovery",
    "MockLLMRecovery",
    "RecoveredSpan",
    "RecoveryResult",
    "is_suspicious_document",
    "should_trigger_recovery",
    "compute_noise_score",
    "create_recovery",
    # Pipeline v2 - Main Pipeline
    "ExtractionPipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStats",
    "save_pipeline_result",
]
