# Dread Rising Configuration

"""
GEMINI API RATE LIMITS (Free Tier):
===============================
Model                                    RPM    TPM       RPD
Gemini 2.5 Flash Preview 05-20          10     250,000   500
Gemini 2.5 Flash Preview TTS             3      10,000    15
Gemini 2.5 Pro Preview 06-05            --     --        --
Gemini 2.5 Pro Preview TTS               --     --        --
Gemini 2.5 Pro Experimental 03-25       5      250,000   25 (TPM: 1,000,000 TPD)
Gemini 2.0 Flash                        15     1,000,000 1,500
Gemini 2.0 Flash Preview Image Gen      10     200,000   100
Gemini 2.0 Flash Experimental           10     250,000   1,000
Gemini 2.0 Flash-Lite                   30     1,000,000 1,500
Gemini 1.5 Flash                        15     250,000   500
Gemini 1.5 Flash-8B                     15     250,000   500
Gemini 1.5 Pro                          --     --        --
Veo 2                                    --     --        --
Imagen 3                                 --     --        --
Gemma 3                                  30     15,000    14,400
Gemma 3n                                 30     15,000    14,400
Gemini Embedding Experimental 03-07     5      --        100

RPM = Requests Per Minute
TPM = Tokens Per Minute  
RPD = Requests Per Day

AVAILABLE MODELS:
================
Gemini 2.5 Flash Preview 05-20 (gemini-2.5-flash-preview-05-20)
- Input: Audio, images, videos, text | Output: Text
- Optimized for: Adaptive thinking, cost efficiency

Gemini 2.5 Flash Native Audio (gemini-2.5-flash-preview-native-audio-dialog)
- Input: Audio, videos, text | Output: Text and audio, interleaved
- Optimized for: High quality, natural conversational audio outputs

Gemini 2.5 Flash Preview TTS (gemini-2.5-flash-preview-tts)
- Input: Text | Output: Audio
- Optimized for: Low latency, controllable text-to-speech

Gemini 2.5 Pro Preview (gemini-2.5-pro-preview-06-05)
- Input: Audio, images, videos, text | Output: Text
- Optimized for: Enhanced thinking and reasoning, multimodal understanding

Gemini 2.0 Flash (gemini-2.0-flash)  CURRENT DEFAULT
- Input: Audio, images, videos, text | Output: Text
- Optimized for: Next generation features, speed, thinking, realtime streaming

Gemini 2.0 Flash Preview Image Generation (gemini-2.0-flash-preview-image-generation)
- Input: Audio, images, videos, text | Output: Text, images
- Optimized for: Conversational image generation and editing

Gemini 2.0 Flash-Lite (gemini-2.0-flash-lite)
- Input: Audio, images, videos, text | Output: Text
- Optimized for: Cost efficiency and low latency

Gemini 1.5 Flash (gemini-1.5-flash)
- Input: Audio, images, videos, text | Output: Text
- Optimized for: Fast and versatile performance

Gemini 1.5 Flash-8B (gemini-1.5-flash-8b)
- Input: Audio, images, videos, text | Output: Text
- Optimized for: High volume and lower intelligence tasks

Gemini 1.5 Pro (gemini-1.5-pro)
- Input: Audio, images, videos, text | Output: Text
- Optimized for: Complex reasoning tasks requiring more intelligence

Gemini 2.0 Flash Live (gemini-2.0-flash-live-001)
- Input: Audio, video, text | Output: Text, audio
- Optimized for: Low-latency bidirectional voice and video interactions

For URL Context tool, use: gemini-2.5-flash-preview-05-20
For Google Search grounding: Any model with tool support
For PDF analysis: gemini-2.0-flash (current default)
"""

# API Settings
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"  # i like the 2.5 flash
GEMINI_TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 4000

# when should you use which model
MODELS = {
    "default": "gemini-2.0-flash",              # General purpose, good rate limits
    "pdf_analysis": "gemini-2.0-flash",        # PDF document analysis
    "url_context": "gemini-2.5-flash-preview-05-20",  # URL context analysis (required for URL tool)
    "search_grounding": "gemini-2.0-flash",    # Google Search grounding
    "fast_analysis": "gemini-2.0-flash-lite",  # Quick analysis, cost efficient
    "complex_reasoning": "gemini-1.5-pro",     # Complex research tasks 
    "high_volume": "gemini-1.5-flash-8b"       # High volume processing
}

# Rate Limiting Settings (based on free tier limits)
RATE_LIMITS = {
    "gemini-2.0-flash": {"rpm": 15, "tpm": 1000000, "rpd": 1500},
    "gemini-2.0-flash-lite": {"rpm": 30, "tpm": 1000000, "rpd": 1500},
    "gemini-1.5-flash": {"rpm": 15, "tpm": 250000, "rpd": 500},
    "gemini-1.5-flash-8b": {"rpm": 15, "tpm": 250000, "rpd": 500},
    "gemini-2.5-flash-preview-05-20": {"rpm": 10, "tpm": 250000, "rpd": 500}
}

# Research Settings
MAX_ARXIV_RESULTS = 10
MAX_SEARCH_RESULTS = 5
DEFAULT_HIGHLIGHT_COLOR = "#FFFF00"

# UI Settings
PAGE_TITLE = "Dread Rising"
PAGE_ICON = "DR"
LAYOUT = "wide"

# File Settings
MAX_FILE_SIZE_MB = 50
SUPPORTED_FORMATS = [".pdf"]

# arXiv Settings
ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
ARXIV_RESULTS_PER_PAGE = 10

# Search Settings
SEARCH_ENGINES = ["duckduckgo", "google"]
DEFAULT_SEARCH_ENGINE = "duckduckgo"

# Search Rate Limiting (to avoid 429 errors)
SEARCH_MIN_INTERVAL = 5  # Minimum seconds between searches
SEARCH_MAX_RETRIES = 1     # Maximum retry attempts (fail fast)
SEARCH_RETRY_DELAY = 30   # Seconds to wait on rate limit
SEARCH_MAX_RESULTS_SAFE = 5  # Conservative max results to avoid rate limits

# Fallback Search URLs
FALLBACK_SEARCH_ENGINES = {
    "google_scholar": "https://scholar.google.com/scholar?q=",
    "arxiv": "https://arxiv.org/search/?query=",
    "pubmed": "https://pubmed.ncbi.nlm.nih.gov/?term=",
    "semantic_scholar": "https://www.semanticscholar.org/search?q="
}
