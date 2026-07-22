import html
import io
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from itertools import islice
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency may fail during import
    load_dataset = None

try:
    from gtts import gTTS
except Exception:  # pragma: no cover - optional dependency may fail during import
    gTTS = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency may fail during import
    OpenAI = None


st.set_page_config(
    page_title="단계별 영어 딕테이션 & 해석",
    page_icon="🎧",
    layout="wide",
)


APP_CSS = """
<style>
    .app-hero {
        padding: 1.1rem 1.3rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #123a5f 50%, #0f766e 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.18);
    }
    .app-hero h1 {
        margin: 0;
        font-size: 2.1rem;
        line-height: 1.15;
    }
    .app-hero p {
        margin: 0.45rem 0 0;
        opacity: 0.9;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.35rem 0 1rem;
    }
    .chip {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(15, 23, 42, 0.09);
        font-size: 0.85rem;
    }
    .ok-word {
        padding: 0 2px;
        border-bottom: 2px solid rgba(34, 197, 94, 0.6);
    }
    .missing-word {
        padding: 0 2px;
        background: rgba(239, 68, 68, 0.18);
        border-radius: 5px;
        text-decoration: line-through;
        text-decoration-thickness: 2px;
    }
    .extra-word {
        padding: 0 2px;
        background: rgba(245, 158, 11, 0.25);
        border-radius: 5px;
    }
    .replace-word {
        padding: 0 2px;
        background: rgba(168, 85, 247, 0.18);
        border-radius: 5px;
    }
    .ref-block, .user-block {
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(248, 250, 252, 0.9);
        margin-bottom: 0.75rem;
        line-height: 1.8;
        word-break: keep-all;
    }
    .muted {
        color: #64748b;
        font-size: 0.92rem;
    }
</style>
"""


COMMON_WORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "and",
    "as",
    "at",
    "be",
    "because",
    "before",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "me",
    "more",
    "my",
    "no",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "up",
    "use",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}


REDUCED_PRONUNCIATION_HINTS = [
    ("going to", "going to는 빠른 발음에서 'gonna'처럼 들릴 수 있어요."),
    ("want to", "want to는 빠른 말에서 'wanna'처럼 들릴 수 있어요."),
    ("got to", "got to는 회화체에서 'gotta'처럼 이어질 수 있어요."),
    ("have to", "have to는 'hafta'처럼 들릴 수 있어요."),
    ("has to", "has to는 'hasta'처럼 들릴 수 있어요."),
    ("did you", "did you는 빠르게 붙으면 'didja'처럼 들릴 수 있어요."),
    ("would you", "would you는 빠른 발화에서 'wouldja'처럼 이어질 수 있어요."),
    ("could you", "could you는 'couldja'처럼 들릴 수 있어요."),
    ("to you", "to you는 앞 단어와 연결되어 약하게 들릴 수 있어요."),
    ("and then", "and는 약하게 붙어 들릴 수 있으니 연결 발음을 의식해보세요."),
]


ADVANCED_HINTS = [
    ("because", "because 이하 절이 뒤따르는지 확인하면서 덩어리로 끊어 들으면 좋아요."),
    ("although", "although가 나오면 앞뒤 절 경계를 나눠 듣는 연습이 중요해요."),
    ("which", "which / that 관계절은 선행사와 함께 묶어 듣는 습관이 도움이 됩니다."),
    ("who", "who가 이끄는 관계절은 앞 명사를 먼저 잡아두면 해석이 쉬워져요."),
    ("unless", "unless 같은 조건 접속사는 뒤집어 해석해보면 정확도가 올라가요."),
]


@dataclass
class SentenceItem:
    english: str
    korean: str
    difficulty: str
    length_band: str
    source: str
    score: float
    pronunciation_patterns: Optional[List[str]] = None


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text.lower())


def length_band(word_count: int) -> str:
    if word_count <= 8:
        return "짧음"
    if word_count <= 16:
        return "보통"
    return "긴 문장"


def estimate_difficulty(text: str) -> Tuple[str, float]:
    words = tokenize_words(text)
    if not words:
        return "Beginner", 0.0

    word_count = len(words)
    avg_len = sum(len(w) for w in words) / word_count
    unique_ratio = len(set(words)) / word_count
    long_words = sum(1 for w in words if len(w) >= 7)
    clause_markers = sum(
        1 for w in words if w in {"because", "although", "unless", "while", "since", "which", "who", "that"}
    )
    punctuation_bonus = len(re.findall(r"[,:;()]", text))
    reduced_bonus = sum(1 for phrase, _ in REDUCED_PRONUNCIATION_HINTS if phrase in text.lower())

    score = (
        word_count * 0.55
        + avg_len * 1.4
        + unique_ratio * 4.0
        + long_words * 0.7
        + clause_markers * 1.2
        + punctuation_bonus * 0.4
        + reduced_bonus * 0.8
    )

    if score < 13:
        return "Beginner", score
    if score < 21:
        return "Intermediate", score
    return "Advanced", score


def sentence_length_label(text: str) -> str:
    return length_band(len(tokenize_words(text)))


def normalize_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_sentence_item(english: str, korean: str, source: str) -> Optional[SentenceItem]:
    english = normalize_sentence(english)
    korean = normalize_sentence(korean)
    if not english or not korean:
        return None
    if len(tokenize_words(english)) < 4:
        return None
    difficulty, score = estimate_difficulty(english)
    return SentenceItem(
        english=english,
        korean=korean,
        difficulty=difficulty,
        length_band=sentence_length_label(english),
        source=source,
        score=score,
    )


@st.cache_resource(show_spinner=False)
def load_hf_sentence_pool(pool_target: int = 12000) -> List[SentenceItem]:
    """
    Build a large in-memory pool from the OPUS-100 en-ko training split.
    The dataset itself contains 1,000,000 training rows, so even a sampled pool
    gives plenty of variety for repeated practice.
    """

    if load_dataset is None:
        return []

    try:
        stream = load_dataset(
            "Helsinki-NLP/opus-100",
            "en-ko",
            split="train",
            streaming=True,
        )
        stream = stream.shuffle(seed=42, buffer_size=10_000)
    except Exception:
        return []

    pool: List[SentenceItem] = []
    seen: set = set()

    for row in islice(stream, 25000):
        translation = row.get("translation", {})
        english = translation.get("en", "") if isinstance(translation, dict) else ""
        korean = translation.get("ko", "") if isinstance(translation, dict) else ""
        item = build_sentence_item(english, korean, "Hugging Face OPUS-100 en-ko")
        if not item:
            continue
        signature = (item.english.lower(), item.korean)
        if signature in seen:
            continue
        seen.add(signature)
        pool.append(item)
        if len(pool) >= pool_target:
            break

    if not pool:
        fallback_pairs = [
            ("I usually walk to work.", "나는 보통 걸어서 출근해요."),
            ("Please speak a little more slowly.", "조금 더 천천히 말해 주세요."),
            ("She has already finished her homework.", "그녀는 이미 숙제를 끝냈어요."),
            ("We are going to meet after lunch.", "우리는 점심 후에 만날 거예요."),
            ("Could you repeat that one more time?", "한 번만 더 말씀해 주시겠어요?"),
            ("The movie was more interesting than I expected.", "그 영화는 내가 예상했던 것보다 더 흥미로웠어요."),
            ("Although it was raining, they kept walking.", "비가 왔지만 그들은 계속 걸었어요."),
            ("If you need help, just let me know.", "도움이 필요하면 언제든 알려 주세요."),
        ]
        for eng, kor in fallback_pairs:
            item = build_sentence_item(eng, kor, "Local fallback list")
            if item:
                pool.append(item)

    return pool


def extract_json_payload(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except Exception:
        return None


def openai_generate_sentence(level: str, length_pref: str, api_key: str, model: str) -> Tuple[Optional[SentenceItem], str]:
    if OpenAI is None or not api_key.strip():
        return None, ""

    try:
        client = OpenAI(api_key=api_key.strip())
        prompt = f"""
You are creating a Korean-English dictation practice item.
Return JSON only, with keys: english, korean, tip.

Constraints:
- English must be one natural sentence for dictation.
- Difficulty target: {level}
- Sentence length preference: {length_pref}
- The English sentence should be suitable for speaking practice, not a quote or a fragment.
- Keep the English sentence between 6 and 20 words.
- Provide a natural Korean translation.
- Tip must be a short Korean note about pronunciation, linking, or grammar.

JSON schema:
{{
  "english": "...",
  "korean": "...",
  "tip": "..."
}}
"""
        response = client.responses.create(
            model=model.strip() or "gpt-5",
            input=prompt,
        )
        payload = extract_json_payload(getattr(response, "output_text", ""))
        if not payload:
            return None, ""
        english = payload.get("english", "")
        korean = payload.get("korean", "")
        tip = payload.get("tip", "")
        item = build_sentence_item(english, korean, "OpenAI generated")
        return item, tip
    except Exception:
        return None, ""


def score_match(item: SentenceItem, difficulty: str, length_pref: str) -> Tuple[int, int]:
    difficulty_score = 2 if item.difficulty == difficulty else 1
    length_score = 2 if item.length_band == length_pref else 1
    return difficulty_score, length_score


def select_sentence(
    items: Sequence[SentenceItem],
    difficulty: str,
    length_pref: str,
    avoid: Optional[set] = None,
) -> Tuple[SentenceItem, str]:
    avoid = avoid or set()

    def candidates_for(diff: Optional[str], length: Optional[str]) -> List[SentenceItem]:
        result = []
        for item in items:
            if (item.english, item.korean) in avoid:
                continue
            if diff and item.difficulty != diff:
                continue
            if length and item.length_band != length:
                continue
            result.append(item)
        return result

    exact = candidates_for(difficulty, length_pref)
    if exact:
        return random.choice(exact), "exact"

    by_difficulty = candidates_for(difficulty, None)
    if by_difficulty:
        return random.choice(by_difficulty), "difficulty"

    by_length = candidates_for(None, length_pref)
    if by_length:
        return random.choice(by_length), "length"

    fallback = [item for item in items if (item.english, item.korean) not in avoid]
    if fallback:
        return random.choice(fallback), "fallback"

    raise RuntimeError("No sentence items available.")


@st.cache_data(show_spinner=False)
def make_tts_audio_bytes(text: str, lang: str = "en") -> Optional[bytes]:
    if gTTS is None or not text.strip():
        return None
    try:
        fp = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(fp)
        return fp.getvalue()
    except Exception:
        return None


def english_tokenize_with_spans(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text)


def diff_markup(reference: str, user_text: str) -> Tuple[str, str, Dict[str, List[str]]]:
    ref_tokens = english_tokenize_with_spans(reference)
    user_tokens = english_tokenize_with_spans(user_text)

    matcher = SequenceMatcher(a=[t.lower() for t in ref_tokens], b=[t.lower() for t in user_tokens])
    ref_parts: List[str] = []
    user_parts: List[str] = []
    missing: List[str] = []
    extra: List[str] = []
    replaced: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_chunk = ref_tokens[i1:i2]
        user_chunk = user_tokens[j1:j2]
        if tag == "equal":
            for token in ref_chunk:
                ref_parts.append(f"<span class='ok-word'>{html.escape(token)}</span>")
            for token in user_chunk:
                user_parts.append(f"<span class='ok-word'>{html.escape(token)}</span>")
        elif tag == "delete":
            missing.extend(ref_chunk)
            for token in ref_chunk:
                ref_parts.append(f"<span class='missing-word'>{html.escape(token)}</span>")
        elif tag == "insert":
            extra.extend(user_chunk)
            for token in user_chunk:
                user_parts.append(f"<span class='extra-word'>{html.escape(token)}</span>")
        elif tag == "replace":
            missing.extend(ref_chunk)
            extra.extend(user_chunk)
            replaced.extend(ref_chunk + user_chunk)
            for token in ref_chunk:
                ref_parts.append(f"<span class='missing-word'>{html.escape(token)}</span>")
            for token in user_chunk:
                user_parts.append(f"<span class='replace-word'>{html.escape(token)}</span>")

    summary = {"missing": missing, "extra": extra, "replaced": replaced}
    return " ".join(ref_parts), " ".join(user_parts), summary


def build_tips(sentence: str, difficulty: str) -> List[str]:
    text = sentence.lower()
    tips: List[str] = []

    for phrase, tip in REDUCED_PRONUNCIATION_HINTS:
        if phrase in text:
            tips.append(tip)

    if difficulty == "Beginner":
        tips.append("짧은 문장은 주어-동사-목적어 순서를 먼저 잡고 들으면 훨씬 쉬워요.")
        tips.append("관사(a/the)와 be동사, 전치사 같은 기능어를 놓치지 않도록 주의해보세요.")
    elif difficulty == "Intermediate":
        tips.append("수식어가 길어져도 핵심 동사와 목적어를 먼저 잡으면 문장 구조가 안정됩니다.")
        tips.append("연결어 and, but, because, when 뒤에서 새 덩어리가 시작되는지 확인해보세요.")
    else:
        tips.append("절이 길어질수록 접속사와 관계대명사 뒤의 내용을 한 덩어리로 듣는 습관이 중요해요.")
        tips.append("의미 단위로 끊어서 받아 적고, 마지막에 문장 전체를 다시 점검해보세요.")

    for marker, tip in ADVANCED_HINTS:
        if marker in text:
            tips.append(tip)

    # Keep unique and compact.
    compact = []
    seen = set()
    for tip in tips:
        if tip not in seen:
            compact.append(tip)
            seen.add(tip)
    return compact[:4]


def calculate_accuracy(reference: str, attempt: str) -> Tuple[float, int, int]:
    ref = tokenize_words(reference)
    att = tokenize_words(attempt)
    if not ref:
        return 0.0, 0, 0
    matcher = SequenceMatcher(a=ref, b=att)
    correct = sum(triple.size for triple in matcher.get_matching_blocks())
    return round(correct / len(ref) * 100, 1), correct, len(ref)


def render_sentence_block(title: str, html_body: str, kind: str = "ref") -> None:
    st.markdown(f"**{title}**")
    class_name = "ref-block" if kind == "ref" else "user-block"
    st.markdown(f"<div class='{class_name}'>{html_body}</div>", unsafe_allow_html=True)


def get_history_path() -> str:
    return os.path.join(os.path.dirname(__file__), "learning_history.json")


def load_history() -> List[dict]:
    path = get_history_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(history: List[dict]) -> None:
    with open(get_history_path(), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def upsert_history_entry(history: List[dict], item: SentenceItem, accuracy: float, error_ratio: float) -> List[dict]:
    today = date.today().isoformat()
    entry = {
        "english": item.english,
        "korean": item.korean,
        "difficulty": item.difficulty,
        "length_band": item.length_band,
        "source": item.source,
        "created_at": today,
        "last_reviewed_at": today,
        "review_count": 1,
        "total_attempts": 1,
        "accuracy_sum": accuracy,
        "last_accuracy": accuracy,
        "last_error_ratio": error_ratio,
        "priority_score": round(max(0.0, min(100.0, (error_ratio * 50) + 20)), 1),
        "pronunciation_patterns": item.pronunciation_patterns or [],
    }

    for existing in history:
        if existing.get("english") == item.english:
            existing["total_attempts"] = existing.get("total_attempts", 0) + 1
            existing["accuracy_sum"] = existing.get("accuracy_sum", 0.0) + accuracy
            existing["last_accuracy"] = accuracy
            existing["last_error_ratio"] = error_ratio
            existing["last_reviewed_at"] = today
            existing["review_count"] = existing.get("review_count", 0) + 1
            existing["priority_score"] = round(max(0.0, min(100.0, (existing.get("priority_score", 0.0) + entry["priority_score"]) / 2.0)), 1)
            return history

    history.append(entry)
    return history


def get_due_review_candidates(history: List[dict], reference_date: Optional[date] = None) -> List[dict]:
    reference_date = reference_date or date.today()
    review_cycles = [
        (1, 30),
        (3, 50),
        (7, 60),
        (15, 70),
        (30, 70),
        (60, 70),
        (120, 70),
        (240, 70),
    ]
    candidates = []
    for entry in history:
        created_at = entry.get("created_at")
        if not created_at:
            continue
        try:
            created_date = datetime.strptime(created_at, "%Y-%m-%d").date()
        except Exception:
            continue
        priority_score = float(entry.get("priority_score", 0.0))
        matching_cycle = None
        for cycle_days, threshold in review_cycles:
            due_date = created_date + timedelta(days=cycle_days)
            if reference_date == due_date and priority_score >= threshold:
                matching_cycle = cycle_days
                break
        if matching_cycle is None:
            continue
        candidates.append({**entry, "cycle_days": matching_cycle})

    candidates.sort(key=lambda item: (-item.get("priority_score", 0.0), item.get("english", "")))
    return candidates


def initialize_state() -> None:
    defaults = {
        "current_item": None,
        "current_source_mode": "Hugging Face 데이터셋",
        "current_difficulty": "Intermediate",
        "current_length": "보통",
        "current_mode": "실시간 딕테이션 (신규 학습)",
        "attempt1": "",
        "attempt2": "",
        "translation": "",
        "sentence_generation_count": 0,
        "used_signatures": set(),
        "force_new_sentence": False,
        "openai_tip": "",
        "last_load_reason": "",
        "review_mode_note": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_local_sentence_pool() -> List[SentenceItem]:
    pool = load_hf_sentence_pool()
    return pool


def create_sentence_from_settings(
    source_mode: str,
    difficulty: str,
    length_pref: str,
    api_key: str,
    model: str,
    mode: str,
    reference_date: Optional[date] = None,
) -> Tuple[SentenceItem, str, str]:
    used = st.session_state.get("used_signatures", set())
    if source_mode == "OpenAI 실시간 생성":
        item, tip = openai_generate_sentence(difficulty, length_pref, api_key, model)
        if item:
            signature = (item.english, item.korean)
            used.add(signature)
            st.session_state.used_signatures = used
            st.session_state.openai_tip = tip
            return item, "openai", "OpenAI 모델로 새 문장을 생성했어요."

    if mode == "망각곡선 스마트 복습 (1·3·7·15·30·60·120·240일)":
        history = load_history()
        due_items = get_due_review_candidates(history, reference_date=reference_date)
        if due_items:
            selected = due_items[0]
            item = SentenceItem(
                english=selected["english"],
                korean=selected["korean"],
                difficulty=selected.get("difficulty", "Intermediate"),
                length_band=selected.get("length_band", "보통"),
                source=selected.get("source", "History"),
                score=float(selected.get("priority_score", 0.0)),
                pronunciation_patterns=selected.get("pronunciation_patterns", []),
            )
            return item, "review", f"복습 큐에서 선별된 문장입니다. ({selected['cycle_days']}일차 기준)"

    pool = get_local_sentence_pool()
    if pool:
        item, reason = select_sentence(pool, difficulty, length_pref, avoid=used)
        used.add((item.english, item.korean))
        st.session_state.used_signatures = used
        if reason == "exact":
            note = "선택한 난이도와 길이에 정확히 맞는 문장을 불러왔어요."
        elif reason == "difficulty":
            note = "난이도는 정확히 맞췄고, 길이 조건은 가장 가까운 문장을 사용했어요."
        elif reason == "length":
            note = "길이 조건은 정확히 맞췄고, 난이도는 가장 가까운 문장을 사용했어요."
        else:
            note = "조건에 맞는 문장이 부족해서 가능한 범위에서 가장 적절한 문장을 불러왔어요."
        return item, reason, note

    # Ultimate fallback so the app still works offline or when data access fails.
    fallback = build_sentence_item(
        "Could you help me carry these boxes to the car?",
        "이 상자들을 차까지 옮기는 걸 도와주실 수 있나요?",
        "Local fallback list",
    )
    assert fallback is not None
    return fallback, "fallback", "데이터를 불러오지 못해 기본 문장을 사용했어요."


initialize_state()

st.markdown(APP_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-hero">
        <h1>단계별 영어 딕테이션 & 해석</h1>
        <p>난이도별 문장을 불러와 1차 딕테이션, 2차 수정, 한국어 해석, 정답 비교까지 한 번에 연습합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="chip-row">
        <span class="chip">Hugging Face OPUS-100 en-ko</span>
        <span class="chip">gTTS 오디오</span>
        <span class="chip">Streamlit session_state</span>
        <span class="chip">OpenAI 생성 옵션</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("설정")
    source_mode = st.selectbox(
        "문장 공급원",
        ["Hugging Face 데이터셋", "OpenAI 실시간 생성"],
        index=0 if st.session_state.current_source_mode == "Hugging Face 데이터셋" else 1,
        key="source_mode_select",
    )
    difficulty = st.selectbox(
        "난이도",
        ["Beginner", "Intermediate", "Advanced"],
        index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.current_difficulty),
        key="difficulty_select",
    )
    length_pref = st.selectbox(
        "문장 길이",
        ["짧음", "보통", "긴 문장"],
        index=["짧음", "보통", "긴 문장"].index(st.session_state.current_length),
        key="length_select",
    )
    learning_mode = st.selectbox(
        "학습 모드",
        ["실시간 딕테이션 (신규 학습)", "망각곡선 스마트 복습 (1·3·7·15·30·60·120·240일)"],
        index=0 if st.session_state.current_mode == "실시간 딕테이션 (신규 학습)" else 1,
        key="learning_mode_select",
    )
    review_reference_date = st.date_input(
        "복습 기준 날짜",
        value=date.today(),
        key="review_reference_date",
    )

    st.caption("OpenAI 실시간 생성 모드를 쓰려면 API 키를 입력하세요.")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-...",
    )
    openai_model = st.text_input("OpenAI model", value="gpt-5")

    if st.button("새 문장 불러오기", use_container_width=True):
        st.session_state.force_new_sentence = True

    st.divider()
    history = load_history()
    due_candidates = get_due_review_candidates(history, review_reference_date)
    st.caption(f"저장된 학습 기록: {len(history)}개")
    st.caption(f"현재 복습 큐: {len(due_candidates)}개")
    st.write("선택한 날짜 기준으로 복습 대상 문장을 미리 확인할 수 있습니다.")


if (
    st.session_state.current_item is None
    or st.session_state.force_new_sentence
    or st.session_state.current_source_mode != source_mode
    or st.session_state.current_difficulty != difficulty
    or st.session_state.current_length != length_pref
    or st.session_state.current_mode != learning_mode
):
    with st.spinner("문장을 준비하는 중..."):
        item, source_tag, load_reason = create_sentence_from_settings(
            source_mode=source_mode,
            difficulty=difficulty,
            length_pref=length_pref,
            api_key=openai_api_key,
            model=openai_model,
            mode=learning_mode,
            reference_date=review_reference_date,
        )
    st.session_state.current_item = item
    st.session_state.current_source_mode = source_mode
    st.session_state.current_difficulty = difficulty
    st.session_state.current_length = length_pref
    st.session_state.current_mode = learning_mode
    st.session_state.last_load_reason = load_reason
    st.session_state.force_new_sentence = False
    st.session_state.attempt1 = ""
    st.session_state.attempt2 = ""
    st.session_state.translation = ""
    if source_tag != "openai":
        st.session_state.openai_tip = ""

item: SentenceItem = st.session_state.current_item

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("### 날짜별 복습/기록 확인")
    history = load_history()
    due_candidates = get_due_review_candidates(history, review_reference_date)
    if learning_mode == "망각곡선 스마트 복습 (1·3·7·15·30·60·120·240일)":
        if due_candidates:
            st.write(f"{review_reference_date} 기준 복습 큐")
            for entry in due_candidates[:8]:
                st.write(f"- {entry.get('english', '-')} | 점수 {entry.get('priority_score', 0.0):.1f} | {entry.get('cycle_days')}일차")
        else:
            st.info("이 날짜에는 복습 대상 문장이 없습니다.")
    else:
        st.caption("복습 모드에서만 날짜 기준 조회가 적용됩니다.")

    if history:
        st.markdown("#### 최근 학습 기록")
        recent_history = sorted(history, key=lambda item: item.get("created_at", ""), reverse=True)[:8]
        for entry in recent_history:
            st.write(f"- {entry.get('created_at', '-')} | {entry.get('english', '-')}" )
    else:
        st.info("아직 저장된 학습 기록이 없습니다.")
    st.subheader("현재 문장")
    st.info(st.session_state.last_load_reason)

    meta_cols = st.columns(4)
    meta_cols[0].metric("난이도", item.difficulty)
    meta_cols[1].metric("길이", item.length_band)
    meta_cols[2].metric("출처", "OpenAI" if "OpenAI" in item.source else ("복습" if learning_mode == "망각곡선 스마트 복습 (1·3·7·15·30·60·120·240일)" else "HF"))
    meta_cols[3].metric("점수", f"{item.score:.1f}")

    audio_bytes = make_tts_audio_bytes(item.english, "en")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3", autoplay=False)
    else:
        st.warning("오디오를 만들 수 없어 텍스트만 표시합니다. gTTS 설치 또는 네트워크 연결을 확인하세요.")

    st.markdown("### Step 1: 1차 딕테이션")
    st.text_area(
        "들리는 대로 영어를 받아 적어 보세요.",
        key="attempt1",
        height=110,
        placeholder="예: I usually walk to work.",
    )

    st.markdown("### Step 2: 2차 딕테이션")
    st.caption("같은 문장을 한 번 더 듣고 1차 입력을 수정해 보세요.")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3", autoplay=False)
    st.text_area(
        "1차 입력을 바탕으로 수정한 최종 영어 문장",
        key="attempt2",
        height=110,
        placeholder="예: I usually walk to work.",
    )

    st.markdown("### Step 3: 한국어 해석")
    st.text_area(
        "완성한 영어 문장의 뜻을 한국어로 적어 보세요.",
        key="translation",
        height=110,
        placeholder="예: 나는 보통 걸어서 출근한다.",
    )

    submit = st.button("제출", type="primary", use_container_width=True)

with right:
    st.subheader("학습 보조")

    st.markdown("#### 연음 / 문법 포인트")
    tips = build_tips(item.english, item.difficulty)
    for tip in tips:
        st.write(f"- {tip}")

    st.markdown("#### 문장 정보")
    st.write(f"- 영어 원문 길이: {len(tokenize_words(item.english))} 단어")
    st.write(f"- 한국어 번역 길이: {len(item.korean)} 글자")
    st.write(f"- 데이터 출처: {item.source}")

    if st.session_state.openai_tip:
        st.markdown("#### OpenAI 생성 메모")
        st.write(st.session_state.openai_tip)


if submit:
    st.divider()
    st.subheader("정답 및 분석 결과")

    reference = item.english
    translation = item.korean
    attempt1 = st.session_state.attempt1.strip()
    attempt2 = st.session_state.attempt2.strip()
    user_korean = st.session_state.translation.strip()

    st.markdown("### 영어 원문")
    st.code(reference, language="text")

    st.markdown("### 정확한 한국어 번역")
    st.success(translation)

    if user_korean:
        st.markdown("### 사용자가 적은 한국어 해석")
        st.write(user_korean)

    history = load_history()
    acc1, _, _ = calculate_accuracy(reference, attempt1)
    acc2, _, _ = calculate_accuracy(reference, attempt2)
    best_acc = max(acc1, acc2) if attempt1 or attempt2 else 0.0
    error_ratio = round(max(0.0, 1.0 - (best_acc / 100.0)), 3)
    history = upsert_history_entry(history, item, best_acc, error_ratio)
    save_history(history)

    col_a, col_b = st.columns(2)
    with col_a:
        ref_html_1, user_html_1, summary_1 = diff_markup(reference, attempt1)
        st.markdown("#### 1차 딕테이션 비교")
        render_sentence_block("원문 기준 하이라이트", ref_html_1, kind="ref")
        render_sentence_block("사용자 입력", user_html_1, kind="user")
        acc1, correct1, total1 = calculate_accuracy(reference, attempt1)
        st.write(f"- 정확도: {acc1}% ({correct1}/{total1})")
        if summary_1["missing"]:
            st.write(f"- 누락: {' '.join(summary_1['missing'])}")
        if summary_1["extra"]:
            st.write(f"- 추가: {' '.join(summary_1['extra'])}")

    with col_b:
        ref_html_2, user_html_2, summary_2 = diff_markup(reference, attempt2)
        st.markdown("#### 2차 딕테이션 비교")
        render_sentence_block("원문 기준 하이라이트", ref_html_2, kind="ref")
        render_sentence_block("사용자 입력", user_html_2, kind="user")
        acc2, correct2, total2 = calculate_accuracy(reference, attempt2)
        st.write(f"- 정확도: {acc2}% ({correct2}/{total2})")
        if summary_2["missing"]:
            st.write(f"- 누락: {' '.join(summary_2['missing'])}")
        if summary_2["extra"]:
            st.write(f"- 추가: {' '.join(summary_2['extra'])}")

    if attempt1 or attempt2:
        st.markdown("### 개선 요약")
        if attempt1 and attempt2:
            delta = acc2 - acc1
            st.write(f"- 1차 대비 2차 정확도 변화: {delta:+.1f}점")
        elif attempt2:
            st.write("- 2차 입력만 제출되었어요.")
        else:
            st.write("- 1차 입력만 제출되었어요.")

    st.markdown("### 다음 액션")
    if learning_mode == "망각곡선 스마트 복습 (1·3·7·15·30·60·120·240일)":
        st.write("복습 모드로 다시 새 문장을 불러오면, 저장된 기록 기준으로 다음 복습 대상이 자동으로 선택됩니다.")
    else:
        st.write("사이드바에서 난이도나 길이를 바꾸고 새 문장을 불러오면 계속 연습할 수 있어요.")

