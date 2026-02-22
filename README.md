# AreTheyInvolved

A searchable index of names extracted from the publicly released Epstein files using a multi-stage AI pipeline. Built for research and transparency.

**Website**: [aretheyinvolved.com](https://aretheyinvolved.com)

## Why Open Source?

This is the public release of the code behind [aretheyinvolved.com](https://aretheyinvolved.com). Tools that deal with sensitive public-interest documents should be transparent and auditable. By publishing the extraction pipeline, anyone can verify that the system works correctly, identify potential errors, and understand exactly how names and roles are extracted from documents. The code is published as working production code, unpolished but complete.

## What This Does

When the Epstein files were released, social media was flooded with fake screenshots and unverified claims. This project provides a **systematic, verifiable way** to search the actual documents using modern AI/ML techniques.

- **882,934 documents** processed (849,706 DOJ + 33,228 House Oversight)
- **200,000+ unique names** extracted with role classifications
- **Exact, fuzzy, and phonetic search** across all documents
- **Full provenance**: every name links back to its source document and page

## Data Sources

| Source | Documents | Pages |
|--------|-----------|-------|
| [DOJ Epstein Disclosures](https://www.justice.gov/epstein/doj-disclosures) | 849,706 files (12 datasets) | ~3.5M |
| [House Oversight Committee](https://oversight.house.gov/release/oversight-committee-releases-epstein-records-provided-by-the-department-of-justice/) | 33,228 files | 33,228 |

> **Note:** Dataset 9 (DS9, EFTA00039025-EFTA01262781) has never been accessible on the DOJ website (returns HTTP 404). Currently working on DS9. All other datasets (DS1-DS8, DS10-DS12) are fully processed.

## Pipeline Architecture

The system uses a 7-stage pipeline to extract, verify, and index person names from documents:

```
                              Raw PDF / Image
                                    │
                         ┌──────────▼──────────┐
                         │  Has text layer?     │
                         └────┬────────────┬────┘
                         YES  │            │  NO
                              │            │
                ┌─────────────▼──┐  ┌──────▼──────────────────┐
                │ Extract text   │  │ OCR                     │
                │ from PDF       │  │ (LightOnOCR-2-1B)       │
                └───────────┬────┘  └──────────┬──────────────┘
                            │                  │
                            └────────┬─────────┘
                                     │
                ┌────────────────────▼────────────────────────┐
                │       STAGE 2: DETERMINISTIC CLEANING       │
                │                                             │
                │  Unicode normalize, strip noise              │
                │  Remove box-drawing, broken tokens           │
                │  PRESERVES character offsets                  │
                └────────────────────┬────────────────────────┘
                                     │
      ┌──────────────────────────────▼──────────────────────────────┐
      │                          NER TIER 1                         │
      │                                                             │
      │  ┌───────────────────────────────────────────────────────┐  │
      │  │  STAGE 2.1: XLM-RoBERTa NER (high recall)            │  │
      │  │  Scans each page for potential person names           │  │
      │  └───────────────────────────────────────────────────────┘  │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
      ┌──────────────────────────────▼──────────────────────────────┐
      │                          NER TIER 2                         │
      │                                                             │
      │  ┌───────────────────────────────────────────────────────┐  │
      │  │  STAGE 2.2: LLM Recovery (conditional)               │  │
      │  │  Find names NER missed in corrupted/noisy docs        │  │
      │  │  Gated: text>500 chars, 0-2 names, noise>0.15        │  │
      │  └───────────────────────────────────────────────────────┘  │
      │                                                             │
      │  ┌───────────────────────────────────────────────────────┐  │
      │  │  STAGE 2.3: LLM Classifier (Qwen3-32B-AWQ)           │  │
      │  │  Closed-set classification per candidate:             │  │
      │  │  • is_person: true/false                              │  │
      │  │  • role: sender|recipient|passenger|mentioned         │  │
      │  │  • occupation extraction (lawyer, pilot, etc.)        │  │
      │  └───────────────────────────────────────────────────────┘  │
      │                                                             │
      │  ┌───────────────────────────────────────────────────────┐  │
      │  │  STAGE 2.4: Hard Validator                            │  │
      │  │  Verify raw_text[start:end] == span.text              │  │
      │  │  REJECTS anything the AI may have hallucinated        │  │
      │  └───────────────────────────────────────────────────────┘  │
      │                                                             │
      │  ┌───────────────────────────────────────────────────────┐  │
      │  │  STAGE 2.5: LLM Repair                               │  │
      │  │  Fix OCR errors in corrupted spans only               │  │
      │  │  (0>O, 1>I, 5>S, etc.)                                │  │
      │  └───────────────────────────────────────────────────────┘  │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
      ┌──────────────────────────────▼──────────────────────────────┐
      │                       POST-PROCESSING                       │
      │                                                             │
      │  ┌───────────────────────────────────────────────────────┐  │
      │  │  STAGE 3: JSON-Level Cleaning                         │  │
      │  │  • Strip OCR artifacts & garbage names                │  │
      │  │  • Normalize non-standard roles to "mentioned"         │  │
      │  │  • Deduplicate per document                           │  │
      │  └───────────────────────────────────────────────────────┘  │
      │                                                             │
      │  ┌───────────────────────────────────────────────────────┐  │
      │  │  STAGE 4: LLM Document Summaries                      │  │
      │  │  • MAP then REDUCE for multi-page docs                 │  │
      │  │  • DIRECT for single-page docs                        │  │
      │  │  • Short + long summaries per document                │  │
      │  │  • Document type & date extraction                    │  │
      │  └───────────────────────────────────────────────────────┘  │
      │                                                             │
      │  ┌───────────────────────────────────────────────────────┐  │
      │  │  STAGE 5: Database Import & Entity Resolution         │  │
      │  │  • Import names, roles, occupations to PG             │  │
      │  │  • Trigram-based name grouping (5 phases)             │  │
      │  │  • Cross-document entity deduplication                │  │
      │  └───────────────────────────────────────────────────────┘  │
      └─────────────────────────────────────────────────────────────┘
```

### Key Guarantees

- **No hallucinations**: Stage 2.4 hard validation ensures every extracted name exists verbatim at its claimed position in the original document (`raw_text[start:end] == span.text`)
- **Provenance preserved**: Output contains both `original_text` (exact from document) and `normalized_name` (human-readable)
- **Offset alignment**: Stage 2 cleaning preserves character offsets (`len(clean) == len(raw)`)
- **Verifiable**: Every name links to its source document, page number, and character offset

### Models Used

| Model | Purpose | Stage |
|-------|---------|-------|
| LightOnOCR-2-1B | Optical character recognition | 1 |
| XLM-RoBERTa (CoNLL-03) | Named entity recognition | 2.1 |
| Qwen3-32B-AWQ | Classification, repair, recovery, summaries | 2.2-2.5, 4 |

### Role Classifications

Each name is classified by how it appears in the document:

| Role | Description |
|------|-------------|
| `sender` | Document author (From field, signature) |
| `recipient` | Document recipient (To/CC fields) |
| `passenger` | Listed as passenger in flight logs |
| `mentioned` | Named within document text |
| `other` | Role unclear from context |

## Project Structure

```
aretheyinvolved/
├── src/
│   ├── ner/                        # NER pipeline modules
│   │   ├── pipeline.py             # Main orchestrator
│   │   ├── deterministic_cleaner.py # Stage 2: text cleaning
│   │   ├── xlmr_extractor.py      # Stage 2.1: XLM-R NER
│   │   ├── llm_recovery.py        # Stage 2.2: LLM recovery
│   │   ├── llm_classifier.py      # Stage 2.3: LLM classification
│   │   ├── hard_validator.py      # Stage 2.4: span validation
│   │   ├── llm_repair.py          # Stage 2.5: OCR repair
│   │   ├── shared_model.py        # Shared LLM model loading
│   │   └── llm_backend/           # vLLM / Transformers abstraction
│   ├── extractors/                 # Text extraction (OCR + PDF)
│   └── text_quality.py            # Text quality gating
├── scripts/
│   ├── extraction/                 # Pipeline CLI scripts
│   │   ├── extract_text.py         # Text extraction from PDFs/images
│   │   ├── extract_names_v2.py     # NER pipeline (Tier 1 + Tier 2)
│   │   ├── clean_names.py          # Post-processing & dedup
│   │   ├── generate_summaries.py   # LLM document summaries
│   │   └── validate_extractions.py # Hallucination validation
│   └── shared/constants.py        # Role priorities, document types
└── tests/                          # Test suite
```

## Search Features

- **Exact match**: Direct file ID or full name lookup
- **Fuzzy match**: Trigram similarity for typos and OCR errors
- **Phonetic match**: Handles name variations (e.g., "Smith" vs "Smyth")
- **Full-text search**: Search within document summaries
- **Tiered routing**: Automatic query classification for optimal performance

## Legal Disclaimer

- All data is from publicly released U.S. government documents
- **Presence in these files does NOT imply wrongdoing**
- Many individuals appear as witnesses, investigators, attorneys, or other non-criminal capacities
- This tool is for research and transparency purposes only
- Always verify information with original source documents
- See [aretheyinvolved.com/disclaimer](https://aretheyinvolved.com/disclaimer) for the full archival disclaimer

## License

MIT License - See LICENSE file for details.
