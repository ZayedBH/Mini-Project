import re

import numpy as np
from sentence_transformers import SentenceTransformer


INTENTS = [
    "generate python function",
    "generate python script",
    "convert pseudocode to python",
    "generate loop based python solution",
    "generate recursive python solution",
    "fix python syntax error",
    "fix undefined variable bug",
    "fix incorrect return statement",
    "fix wrong conditional expression",
    "fix off by one loop error",
    "simplify python logic",
    "convert verbose python loop to pythonic",
    "generate torch tensor operations",
    "fix torch tensor shape mismatch",
    "fix torch tensor dtype error",
    "fix missing backward in training loop",
    "fix optimizer step omission",
    "fix loss computation logic",
    "fix torch model forward pass bug",
    "generate simple pytorch nn module",
    "load dataset using huggingface datasets",
    "fix dataset split usage",
    "apply dataset map function",
    "fix dataset column access",
    "generate numpy array operations",
    "fix numpy indexing bug",
    "fix numpy shape mismatch",
    "fix numpy broadcasting error",
    "fix numpy aggregation bug",
    "simplify numpy expressions",
    "convert loop logic to numpy",
    "fix numpy axis misuse",
    "generate vectorized numpy solution",
    "load pandas dataframe",
    "fix pandas column selection bug",
    "fix pandas groupby usage",
    "fix pandas chained indexing",
    "handle pandas missing values",
    "fix pandas filtering condition",
    "simplify pandas transformation pipeline",
    "generate pandas aggregation code",
    "fix pandas datatype conversion",
    "generate matplotlib plot",
    "fix matplotlib dimension mismatch",
    "fix matplotlib labels or title",
    "fix matplotlib subplot usage",
    "generate sklearn training pipeline",
    "fix sklearn train test split",
    "fix sklearn missing fit call",
    "fix sklearn predict usage",
    "fix sklearn shape mismatch between x and y",
    "generate sklearn evaluation metrics",
]

GREETING_KEYWORDS = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
    "good afternoon",
]

SIMILARITY_THRESHOLD = 0.30

# Natural-language coding request patterns matched before embedding.
# Note: 'explain', 'fix', 'debug' are intentionally excluded — they are ambiguous
# (could be follow-ups) and are handled separately in generate_reply.
CODING_REQUEST_RE = re.compile(
    r'\b(write|create|make|build|generate|give me|show me|i need|'
    r'how (do i|to|can i)|calculate|compute|'
    r'code for|function for|script for|program for|example of|implement)\b',
    re.IGNORECASE,
)

# Follow-up / referential patterns — these mean the user is referring to prior output.
FOLLOW_UP_RE = re.compile(
    r'\b(this|that|it|above|the code|the function|the class|the error|those|them|these)\b|'
    r'^(explain|what does|what is|how does|why does|can you explain|could you explain|'
    r'fix this|debug this|optimise this|optimize this|refactor this|improve this|'
    r'add|change|update|modify|make it)',
    re.IGNORECASE,
)

# Load once at startup.
model = None
intent_vectors = None

try:
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    # nomic-embed-text requires task-type prefixes for correct similarity.
    prefixed_intents = ["search_document: " + intent for intent in INTENTS]
    intent_vectors = np.asarray(model.encode(prefixed_intents, normalize_embeddings=True), dtype=np.float32)
except Exception as e:
    print(f"Warning: Failed to load embedding model: {e}")
    print("Intent routing will default to 'out_of_scope' for all non-greeting queries.")


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    return re.sub(r"\s+", " ", lowered)


def _is_greeting(prompt: str) -> bool:
    normalized = _normalize_text(prompt)
    if not normalized:
        return False
    for keyword in GREETING_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', normalized):
            return True
    return False


def route_prompt(prompt: str):
    # 1) Greeting check
    if _is_greeting(prompt):
        return "greeting"

    # 2) Keyword pre-check — catch natural coding requests before embedding
    if CODING_REQUEST_RE.search(_normalize_text(prompt)):
        return "valid_intent"

    # 3) Embedding similarity check (if model loaded)
    if model is None or intent_vectors is None:
        return "valid_intent"

    query_vec = np.asarray(
        model.encode("search_query: " + prompt, normalize_embeddings=True), dtype=np.float32
    )
    similarities = np.dot(intent_vectors, query_vec)
    max_similarity = float(np.max(similarities))

    # 4) Return route
    if max_similarity > SIMILARITY_THRESHOLD:
        return "valid_intent"
    return "out_of_scope"
