"""Shared constants used across extraction and database scripts."""

# Private artifact directories (relative to project root).
# Name JSONs are internal-only artifacts — never exposed publicly.
# The database is the sole runtime interface for frontend/API.
ARTIFACTS_PRIVATE_DIR = "data/processed/names_v2"
ARTIFACTS_SUMMARY_DIR = "data/processed/summaries"
ARTIFACTS_TEXT_DIR = "data/processed/text"

# Role priority for deduplication (lower = higher priority).
# Used by clean_names.py, generate_summaries.py, and update_database.py.
ROLE_PRIORITY = {
    "sender": 0,
    "recipient": 1,
    "passenger": 2,
    "mentioned": 3,
    "other": 4,
}

# Occupation synonyms: surface_form -> canonical occupation.
# Used by generate_summaries.py to normalize extracted occupations.
# NOTE: Legal status (defendant, plaintiff, witness, victim) and relational
# terms (associate, girlfriend, friend) are EXCLUDED from this table.
# Reserved for future: legal_status_mentions, relationship_mentions
OCCUPATION_SYNONYMS = {
    # Legal
    "attorney": "lawyer",
    "counsel": "lawyer",
    "solicitor": "lawyer",
    "defense attorney": "lawyer",
    "defense counsel": "lawyer",
    "prosecutor": "prosecutor",
    "district attorney": "prosecutor",
    "judge": "judge",
    "magistrate": "judge",
    # Law enforcement
    "detective": "detective",
    "special agent": "agent",
    "police officer": "police_officer",
    "officer": "police_officer",
    "sergeant": "police_officer",
    "investigator": "investigator",
    # Aviation
    "pilot": "pilot",
    "co-pilot": "pilot",
    "copilot": "pilot",
    "flight attendant": "flight_attendant",
    "stewardess": "flight_attendant",
    # Medical
    "doctor": "doctor",
    "physician": "doctor",
    "surgeon": "doctor",
    "psychiatrist": "psychiatrist",
    "nurse": "nurse",
    "dentist": "dentist",
    "masseuse": "masseuse",
    "massage therapist": "masseuse",
    # Domestic/service
    "assistant": "assistant",
    "personal assistant": "assistant",
    "secretary": "secretary",
    "housekeeper": "housekeeper",
    "maid": "housekeeper",
    "chauffeur": "driver",
    "driver": "driver",
    "nanny": "nanny",
    "bodyguard": "bodyguard",
    "butler": "butler",
    "chef": "chef",
    # Academic/media
    "professor": "professor",
    "teacher": "teacher",
    "journalist": "journalist",
    "reporter": "journalist",
    # Financial/business
    "financier": "financier",
    "banker": "banker",
    "accountant": "accountant",
}

# Document type classifications for summary extraction.
# Used by generate_summaries.py to classify documents.
DOCUMENT_TYPES = [
    "letter", "email", "memo", "legal_filing", "court_order",
    "deposition", "interview_transcript", "police_report",
    "flight_log", "financial_record", "fax", "phone_record",
    "photograph_description", "government_report", "media_article",
    "personal_note", "other",
]
