"""Answer parsing and scoring for the Qwen3-8B evaluation suite."""
from collections import Counter
import json
import re
import string
import unicodedata

from .data import parse_answer_numbers, score_erased_prediction
from .schema import DATASETS

EVALUATION_SUITE = tuple(name for name in DATASETS if name.startswith("eval/"))


def _normalize_qa(text: str, unicode_punctuation: bool) -> str:
    text = "" if text is None else str(text).lower()
    if unicode_punctuation:
        text = "".join(ch for ch in unicodedata.normalize("NFKD", text)
                       if unicodedata.category(ch) != "Mn")
        text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def qa_exact_match(prediction: str, answers: list[str]) -> bool:
    """Accept an alias under ASCII punctuation deletion or Unicode punctuation spacing."""
    return any(_normalize_qa(prediction, mode) == _normalize_qa(answer, mode)
               for mode in (False, True) for answer in answers)


def qa_token_f1(prediction: str, answers: list[str]) -> float:
    """Return the best alias F1, counting repeated tokens after Unicode normalization."""
    tokens = _normalize_qa(prediction, True).split()
    best = 0.0
    for answer in answers:
        gold = _normalize_qa(answer, True).split()
        if not tokens or not gold:
            f1 = float(tokens == gold)
        else:
            overlap = sum((Counter(tokens) & Counter(gold)).values())
            precision, recall = overlap / len(tokens), overlap / len(gold)
            f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
        best = max(best, f1)
    return best


def parse_tool_name(text: str, candidates: list[str]) -> str | None:
    """Parse a candidate from JSON, a call, a label, or an unambiguous mention."""
    text = str(text or "").strip()
    candidates = sorted({str(name).strip() for name in candidates if str(name).strip()},
                        key=len, reverse=True)
    if not text or not candidates:
        return None
    cleaned = text.strip("`").strip()
    cleaned = re.sub(r"^```(?:json|text)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except ValueError:
        value = None
    name = None
    if isinstance(value, dict):
        name = next((str(value[key]).strip() for key in ("name", "api_name", "tool_name", "action")
                     if value.get(key)), None)
    elif isinstance(value, str):
        name = value.strip()
    if name in candidates:
        return name
    for pattern in (r"\[?\s*([A-Za-z_][A-Za-z0-9_./-]*)\s*\(",
                    r"(?im)^\s*(?:Action|API|Tool|Next tool/API name|Name)\s*:\s*"
                    r"([A-Za-z_][A-Za-z0-9_./-]*)\s*$"):
        match = re.search(pattern, cleaned)
        if match and match.group(1) in candidates:
            return match.group(1)
    bare = cleaned.strip("[](){}'\" \t\r\n")
    if bare in candidates:
        return bare
    found = [name for name in candidates if re.search(
        r"(?<![A-Za-z0-9_./-])" + re.escape(name) + r"(?![A-Za-z0-9_./-])", cleaned)]
    return found[0] if len(found) == 1 else None


def parse_prediction(dataset: str, text: str, sample: dict) -> str | list[str] | None:
    """Parse one generation using the task's answer format."""
    if dataset not in EVALUATION_SUITE:
        raise ValueError(f"Unsupported evaluation dataset: {dataset}")
    task = dataset.split("/")[1]
    if task == "niah":
        return parse_answer_numbers(text)
    if task == "bfcl":
        return parse_tool_name(text, sample["candidate_tool_names"])
    text = str(text).strip()
    if not text:
        return ""
    line = text.splitlines()[0].strip()
    prefix, separator, remainder = line.partition(":")
    return remainder.strip() if separator and prefix.strip().lower() in {"answer", "assistant"} else line


def score_prediction(dataset: str, text: str, sample: dict) -> dict:
    """Score a generation against erased answers; F1 applies only to QA tasks."""
    prediction = parse_prediction(dataset, text, sample)
    answers = sample["targets"]["erased_answers"]
    task = dataset.split("/")[1]
    f1 = None
    if task == "niah":
        em = score_erased_prediction(prediction, sample)
    elif task == "bfcl":
        em = prediction == answers[0].strip()
    else:
        em, f1 = qa_exact_match(prediction, answers), qa_token_f1(prediction, answers)
    return {"parsed_prediction": prediction, "em": bool(em), "f1": f1}


def aggregate_scores(scores: list[dict]) -> dict:
    """Average per-example scores for one task; reject empty or mixed-metric inputs."""
    if not scores:
        raise ValueError("Cannot aggregate an empty evaluation.")
    has_f1 = [score["f1"] is not None for score in scores]
    if any(has_f1) != all(has_f1):
        raise ValueError("Cannot mix QA and non-QA scores in one task aggregate.")
    count = len(scores)
    return {"samples": count, "em": sum(score["em"] for score in scores) / count,
            "f1": sum(score["f1"] for score in scores) / count if all(has_f1) else None}
