
import re
import torch
import streamlit as st
from difflib import SequenceMatcher
from transformers import AutoTokenizer, T5ForConditionalGeneration

from pathlib import Path

# ============================================================
# CONSTANTS
# ============================================================
MODEL_DIR = str(Path.home() / "Downloads" / "my_ai_project" / "flan_t5_api_model_v3")

# MPS (Apple Silicon) > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
PREFIX = "translate command to API: "
MAX_INPUT_LEN = 128
DEFAULT_MAX_TARGET_LEN = 256

EXECUTORS = {
    "send_email":       lambda p: f"📧 Email sent to **{p.get('to','?')}** — Subject: *{p.get('subject','?')}*",
    "schedule_meeting": lambda p: f"📅 Meeting with **{p.get('person','?')}** on {p.get('date','?')} at {p.get('time','?')}",
    "set_reminder":     lambda p: f"⏰ Reminder: {p.get('task','?')} on {p.get('date','?')} at {p.get('time','?')}",
    "send_message":     lambda p: f"💬 Message to **{p.get('to','?')}**: {p.get('message','?')}",
    "create_task":      lambda p: f"✅ Task created: {p.get('title','?')} — due {p.get('due_date','?')}",
    "search_web":       lambda p: f"🔍 Searching: {p.get('query','?')}",
    "play_music":       lambda p: f"🎵 Playing *{p.get('song','?')}* by {p.get('artist','?')}",
    "set_alarm":        lambda p: f"🔔 Alarm set for {p.get('time','?')} — {p.get('label','?')}",
    "get_weather":      lambda p: f"🌤️ Weather for {p.get('location','?')}",
    "book_cab":         lambda p: f"🚕 Cab booked: {p.get('pickup','?')} → {p.get('destination','?')}",
}

KNOWN_PARAMS = {
    "send_email": ["to", "subject", "body"],
    "schedule_meeting": ["person", "time", "date"],
    "set_reminder": ["task", "time", "date"],
    "send_message": ["to", "message"],
    "create_task": ["title", "due_date", "priority"],
    "search_web": ["query"],
    "play_music": ["song", "artist"],
    "set_alarm": ["time", "label"],
    "get_weather": ["location"],
    "book_cab": ["pickup", "destination", "time"],
}

# Params that are nice-to-have but should NOT trigger slot-filling prompts
OPTIONAL_PARAMS = {"body", "priority", "label"}

# Temporal params that should only be injected as null if the model explicitly
# output them as NULL, not if the model simply omitted the key
TEMPORAL_SCHEMA_PARAMS = {"time", "date", "due_date"}

SKIP_COPY_PARAMS = {"time", "date", "duration", "priority", "repeat", "day", "hour", "minute", "count", "amount", "due_date"}
MATCH_THRESHOLD = 0.55

NAME_PATTERNS = [
    # 'send/drop/fire a message/email to NAME saying/about...'
    r'(?:send|drop|shoot|fire|forward)\s+(?:a\s+|an\s+)?(?:email|mail|message|text|note)\s+to\s+(.+?)\s+(?:about|regarding|saying|that|with)',
    # 'send/drop/fire a message/email to NAME' (end of string)
    r'(?:send|drop|shoot|fire|forward)\s+(?:a\s+|an\s+)?(?:email|mail|message|text|note)\s+to\s+(.+?)\s*$',
    # 'email/mail to NAME about...'
    r'(?:email|mail|write)\s+to\s+(.+?)\s+(?:about|regarding|with|saying)',
    # 'email/mail to NAME' (end of string)
    r'(?:email|mail|write)\s+to\s+(.+?)\s*$',
    # 'Call/Text NAME and/about...'
    r'(?:call|text|ping|contact|ring)\s+(.+?)\s+(?:and|to|about|saying|that|regarding)',
    # 'Message NAME and...' (but NOT 'message to')
    r'(?:message)\s+(?!to\b)(.+?)\s+(?:and|to|about|saying|that|regarding)',
    # 'Email NAME about...' (no 'to')
    r'^(?:email|mail)\s+(.+?)\s+(?:about|regarding|with)',
    # 'Let/Tell/Remind/Notify NAME know/that/about...'
    r'(?:let|tell|remind|notify|inform|alert)\s+(.+?)\s+(?:know|that|about|regarding|of)',
    # 'meeting/call/appointment with NAME on/at/tomorrow...'
    r'(?:meeting|call|appointment|session)\s+with\s+(.+?)\s+(?:on|at|tomorrow|today|next|this|for)',
    # 'meeting/call/appointment with NAME' (end of string)
    r'(?:meeting|call|appointment|session)\s+with\s+(.+?)\s*$',
    # 'cab/ride for NAME to/from/at...'
    r'(?:cab|ride|taxi|uber)\s+for\s+(.+?)\s+(?:to|from|at)',
    # 'for NAME' at end of string ONLY in cab/ride context
    r'(?:cab|ride|taxi|uber|book)\s+.*\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$',
    # 'Ping NAME with...'
    r'(?:ping)\s+(.+?)\s+(?:with)',
    # 'with NAME' at end of string
    r'with\s+(.+?)(?:\s*$|\s*,|\s+and\b)',
]

# Common location keywords — if a value matches these, it IS a valid location
_LOCATION_WORDS = {
    # Generic places
    "home", "office", "work", "airport", "station", "hospital", "college",
    "university", "school", "mall", "market", "park", "hotel", "cafe",
    "restaurant", "gym", "library", "temple", "church", "mosque",
    "bus stop", "metro", "railway", "downtown", "city center",
    "clinic", "pharmacy", "warehouse", "factory", "lab", "studio",
    "dormitory", "hostel", "canteen", "auditorium", "playground",
    # Indian cities
    "mumbai", "delhi", "bangalore", "bengaluru", "chennai", "kolkata",
    "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "kanpur",
    "nagpur", "indore", "bhopal", "patna", "guwahati", "chandigarh",
    "thiruvananthapuram", "kochi", "coimbatore", "vizag", "surat",
    "vadodara", "noida", "gurgaon", "gurugram", "faridabad", "agra",
    "varanasi", "amritsar", "ranchi", "dehradun", "shimla", "manali",
    "mysore", "mysuru", "udaipur", "jodhpur", "bhubaneswar", "raipur",
    # Global cities
    "london", "paris", "tokyo", "new york", "berlin", "sydney",
    "singapore", "dubai", "toronto", "san francisco", "seattle",
    "boston", "chicago", "los angeles", "houston", "bangkok",
    # IIT campuses
    "iit", "iit guwahati", "iitg", "iit bombay", "iit delhi",
    "iit madras", "iit kanpur", "iit kharagpur", "campus",
}


def _looks_like_person_name(value: str) -> bool:
    """Heuristic: capitalized word(s) that aren't common location words."""
    val = value.strip()
    if not val:
        return False
    # If it's a known location word, it's NOT a person name
    if val.lower() in _LOCATION_WORDS:
        return False
    # Check if any word in the value is a known location
    for word in val.lower().split():
        if word in _LOCATION_WORDS:
            return False
    # Single-letter values aren't names
    if len(val) <= 1:
        return False
    # Capitalized single/multi word = likely a person name
    words = val.split()
    if all(w[0].isupper() for w in words if w):
        return True
    return False


# ============================================================
# LINEARIZED FORMAT PARSER
# ============================================================
def linear_to_json(linear_str: str) -> dict | None:
    actions = []
    try:
        for block in linear_str.split("[INTENT]"):
            block = block.strip()
            if not block:
                continue
            intent_part = block.split("[PARAMS]")
            intent = intent_part[0].strip()
            if not intent:
                continue
            params = {}
            if len(intent_part) > 1:
                params_str = intent_part[1].replace("[END]", "").strip()
                if params_str:
                    for pair in params_str.split("|"):
                        pair = pair.strip()
                        if "=" in pair:
                            key, val = pair.split("=", 1)
                            key, val = key.strip(), val.strip()
                            params[key] = None if val == "NULL" else val
            actions.append({"intent": intent, "parameters": params})
    except Exception:
        pass
    return {"actions": actions} if actions else None


# ============================================================
# ENTITY CORRECTION (Positional + Fuzzy)
# ============================================================
def _extract_name(command: str) -> str | None:
    for pattern in NAME_PATTERNS:
        match = re.search(pattern, command, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            name = re.sub(r'\s+(?:him|her|them|his|their)\s*$', '', name, flags=re.IGNORECASE)
            return name
    return None


def _ngrams(text: str, max_n: int = 8) -> list[str]:
    words = text.split()
    out = []
    for n in range(1, min(max_n, len(words)) + 1):
        for i in range(len(words) - n + 1):
            out.append(" ".join(words[i:i + n]))
    return out


def _align(value: str, source: str) -> str:
    if not value or not source:
        return value
    candidates = _ngrams(source)
    best, best_r = value, 0.0
    for c in candidates:
        r = SequenceMatcher(None, value.lower(), c.lower()).ratio()
        if r > best_r:
            best_r, best = r, c
    return best if best_r >= MATCH_THRESHOLD else value


def _enforce_schema(result: dict) -> dict:
    """Inject null for any required params the model entirely omitted.
    Skip temporal params (time/date) — if model omitted them, the command
    likely didn't specify them, and we shouldn't force slot-filling for those."""
    if not result or "actions" not in result:
        return result
    for action in result["actions"]:
        intent = action.get("intent", "")
        params = action.get("parameters", {})
        required = KNOWN_PARAMS.get(intent, [])
        for key in required:
            if key not in params and key not in TEMPORAL_SCHEMA_PARAMS:
                params[key] = None
    return result


def _is_grounded(value: str, command: str) -> bool:
    """Check if value has any grounding in the original command."""
    if not value or not command:
        return False
    val_lower = value.strip().lower()
    cmd_lower = command.lower()
    if val_lower in cmd_lower:
        return True
    for ngram in _ngrams(command, max_n=8):
        if SequenceMatcher(None, val_lower, ngram.lower()).ratio() >= 0.6:
            return True
    return False


# ============================================================
# PER-PARAM TYPE CONSTRAINTS
# ============================================================
# "person"   = must be a person name, NOT a location/time/topic
# "location" = must be a place, NOT a person name
# "temporal" = time/date — already handled by SKIP_COPY_PARAMS
# "freetext" = subject, body, message, query etc — no type constraint, just grounding
#
# This catches cross-slot contamination for ALL 10 intents:
#   send_email:       person name in subject/body, or topic in "to"
#   schedule_meeting: location in "person", or person name in "date"
#   set_reminder:     person name in "task"
#   send_message:     task/location in "to"
#   create_task:      person name in "title"
#   search_web:       no person/location confusion likely
#   play_music:       person name in "song", or song in "artist"
#   set_alarm:        person name in "label"
#   get_weather:      person name in "location"
#   book_cab:         person name in "pickup"/"destination"
# ============================================================

PARAM_TYPE_PERSON = {"to", "person", "name", "recipient", "contact", "attendee", "artist"}
PARAM_TYPE_LOCATION = {"pickup", "destination", "location", "place", "city", "address"}

# Patterns that indicate a value is a time/date, not a name or place
_TEMPORAL_RE = re.compile(
    r'^\d|am$|pm$|morning|afternoon|evening|night|noon|midnight|'
    r'today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|'
    r'friday|saturday|sunday|january|february|march|april|may|june|'
    r'july|august|september|october|november|december|'
    r'next\s|this\s|last\s|\d{1,2}[:/]\d{2}',
    re.IGNORECASE,
)


def _looks_temporal(value: str) -> bool:
    return bool(_TEMPORAL_RE.search(value.strip()))


def apply_entity_correction(result: dict, command: str) -> dict:
    if not result or "actions" not in result:
        return result
    result = _enforce_schema(result)
    name = _extract_name(command)

    for action in result["actions"]:
        params = action.get("parameters", {})
        intent = action.get("intent", "")

        for key, value in list(params.items()):
            if value is None or key.lower() in SKIP_COPY_PARAMS:
                continue

            k = key.lower()

            # === PERSON PARAMS: correct first, then validate ===
            if k in PARAM_TYPE_PERSON:
                # Temporal value in person slot (e.g., person="Saturday")
                if _looks_temporal(value):
                    params[key] = None
                    continue
                # We have a regex-extracted name → replace if needed
                if name:
                    ratio = SequenceMatcher(None, value.lower(), name.lower()).ratio()
                    if ratio < 0.5:
                        # Hallucinated or wrong name → replace with extracted name
                        params[key] = name
                    else:
                        # Close match → fuzzy-align to clean up truncations
                        params[key] = _align(value, command)
                    continue
                # No regex name available → grounding check
                if not _is_grounded(value, command):
                    params[key] = None
                    continue
                # Grounded, no name extracted → fuzzy-align
                params[key] = _align(value, command)
                continue

            # === LOCATION PARAMS: grounding + person-name guard ===
            if k in PARAM_TYPE_LOCATION:
                if not _is_grounded(value, command):
                    params[key] = None
                    continue
                # Person name landed in location slot
                if name and SequenceMatcher(None, value.lower(), name.lower()).ratio() > 0.5:
                    params[key] = None
                    continue
                if _looks_like_person_name(value) and value.lower() not in _LOCATION_WORDS:
                    params[key] = None
                    continue
                # Valid location → fuzzy-align
                params[key] = _align(value, command)
                continue

            # === ALL OTHER PARAMS (freetext: subject, body, message, query, etc.) ===
            if not _is_grounded(value, command):
                params[key] = None
                continue
            if isinstance(value, str) and len(value) > 1:
                params[key] = _align(value, command)

    return result


# ============================================================
# MODEL LOADING (cached singleton)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    # MPS does NOT support bfloat16 or device_map; use float32 + manual .to()
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


# ============================================================
# GENERATION
# ============================================================
def generate(command: str, tokenizer, model, max_target_len: int) -> tuple[str, dict | None]:
    input_text = PREFIX + command
    inputs = tokenizer(
        input_text, return_tensors="pt",
        truncation=True, max_length=MAX_INPUT_LEN,
    ).to(DEVICE)

    configs = [
        dict(num_beams=5, early_stopping=True, repetition_penalty=1.3, length_penalty=1.0),
        dict(num_beams=8, early_stopping=True, repetition_penalty=1.0, length_penalty=1.0),
        dict(num_beams=1, do_sample=False),
    ]

    decoded = ""
    for cfg in configs:
        with torch.no_grad():
            ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_target_len,
                **cfg,
            )
        decoded = tokenizer.decode(ids[0], skip_special_tokens=True)
        result = linear_to_json(decoded)
        if result is not None:
            result = apply_entity_correction(result, command)
            return decoded, result

    return decoded, None


# ============================================================
# SLOT-FILLING HELPERS
# ============================================================
def get_missing_slots(payload: dict) -> list[tuple[int, str, str]]:
    """Returns list of (action_idx, intent, param_key) for null slots, skipping optional ones."""
    missing = []
    for i, action in enumerate(payload.get("actions", [])):
        for k, v in action.get("parameters", {}).items():
            if v is None and k not in OPTIONAL_PARAMS:
                missing.append((i, action["intent"], k))
    return missing


def fill_slots(payload: dict, user_input: str, missing: list[tuple[int, str, str]]) -> dict:
    """Fill missing slots from user input. Supports key=value, key: value, and plain values."""
    parts = [s.strip() for s in re.split(r'[,;]', user_input) if s.strip()]

    # Collect the set of expected missing keys for smarter kv detection
    missing_keys = {k.lower() for _, _, k in missing}

    kv_parsed = {}
    plain_values = []
    for part in parts:
        matched_kv = False
        # '=' is always a kv separator
        if "=" in part:
            k, v = part.split("=", 1)
            kv_parsed[k.strip().lower()] = v.strip()
            matched_kv = True
        # ':' is kv ONLY if the left side is a known missing param key (not a time like "3:30pm")
        elif ":" in part:
            k, v = part.split(":", 1)
            k_clean = k.strip().lower()
            if k_clean in missing_keys and k_clean.isalpha():
                kv_parsed[k_clean] = v.strip()
                matched_kv = True
        if not matched_kv:
            plain_values.append(part)

    filled = 0
    for idx, intent, key in missing:
        if key.lower() in kv_parsed:
            payload["actions"][idx]["parameters"][key] = kv_parsed[key.lower()]
            filled += 1
        elif plain_values:
            payload["actions"][idx]["parameters"][key] = plain_values.pop(0)
            filled += 1

    return payload


def execute_payload(payload: dict) -> list[str]:
    results = []
    for action in payload.get("actions", []):
        intent = action.get("intent", "UNKNOWN")
        params = action.get("parameters", {})
        executor = EXECUTORS.get(intent)
        if executor:
            results.append(executor(params))
        else:
            results.append(f"⚠️ No executor for intent `{intent}`")
    return results


# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(
    page_title="NLP → API Translator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dark theme overrides */
    .stApp {
        background-color: #212121;
    }

    /* Chat input styling */
    .stChatInput > div {
        background-color: #2f2f2f !important;
        border: 1px solid #424242 !important;
        border-radius: 24px !important;
    }

    /* Chat message bubbles */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid #2a2a2a;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ececec;
    }

    /* JSON code blocks */
    .stJson {
        background-color: #1a1a2e !important;
        border-radius: 12px !important;
    }

    /* Slider */
    .stSlider > div > div {
        color: #ececec;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        margin: 4px 0;
    }
    .status-online {
        background: #0d3320;
        color: #4ade80;
        border: 1px solid #166534;
    }
    .status-offline {
        background: #3b1219;
        color: #f87171;
        border: 1px solid #991b1b;
    }

    /* Execution result cards */
    .exec-card {
        background: #1e3a2f;
        border: 1px solid #2d5a45;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #d1fae5;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_payload" not in st.session_state:
    st.session_state.pending_payload = None
if "pending_missing" not in st.session_state:
    st.session_state.pending_missing = []
if "original_command" not in st.session_state:
    st.session_state.original_command = ""

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚡ NLP → API")
    st.markdown("---")

    # Model loading
    try:
        tokenizer, model = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        load_error = str(e)

    if model_loaded:
        device_label = "MPS" if DEVICE.type == "mps" else DEVICE.type.upper()
        st.markdown(f'<span class="status-badge status-online">🟢 Model Loaded ({device_label})</span>', unsafe_allow_html=True)
        st.caption(f"FLAN-T5-base · float32 · {device_label}")
    else:
        st.markdown('<span class="status-badge status-offline">🔴 Model Failed</span>', unsafe_allow_html=True)
        st.error(f"Load error: {load_error}")

    st.markdown("---")
    max_gen_len = st.slider("Max Generation Length", 64, 512, DEFAULT_MAX_TARGET_LEN, step=32)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_payload = None
        st.session_state.pending_missing = []
        st.session_state.original_command = ""
        st.rerun()

    st.markdown("---")
    st.markdown("##### Supported Intents")
    for intent in EXECUTORS:
        st.markdown(f"`{intent}`")

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("json_payload"):
            st.markdown(msg["content"])
            st.json(msg["json_payload"], expanded=True)
            if msg.get("exec_results"):
                for er in msg["exec_results"]:
                    st.markdown(f'<div class="exec-card">{er}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Type a command like: 'Send an email to Priya about the report'")

if user_input and model_loaded:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- SLOT-FILLING MODE ----
    if st.session_state.pending_payload is not None:
        payload = fill_slots(
            st.session_state.pending_payload,
            user_input,
            st.session_state.pending_missing,
        )
        still_missing = get_missing_slots(payload)

        if still_missing:
            # Still incomplete
            st.session_state.pending_payload = payload
            st.session_state.pending_missing = still_missing
            missing_labels = [f"`{key}` ({intent})" for _, intent, key in still_missing]
            reply = f"Thanks — still need: {', '.join(missing_labels)}. Please provide the remaining values."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            # All filled — show final payload and execute
            st.session_state.pending_payload = None
            st.session_state.pending_missing = []
            exec_results = execute_payload(payload)
            reply = "All slots filled. Here is the final API payload:"
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "json_payload": payload,
                "exec_results": exec_results,
            })
            with st.chat_message("assistant"):
                st.markdown(reply)
                st.json(payload, expanded=True)
                for er in exec_results:
                    st.markdown(f'<div class="exec-card">{er}</div>', unsafe_allow_html=True)

    # ---- NORMAL INFERENCE MODE ----
    else:
        with st.spinner("Generating..."):
            raw, result = generate(user_input, tokenizer, model, max_gen_len)

        if result is None:
            reply = f"Could not parse model output into a valid API call.\n\n**Raw output:** `{raw}`"
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            missing = get_missing_slots(result)

            if missing:
                # Incomplete — enter slot-filling mode
                st.session_state.pending_payload = result
                st.session_state.pending_missing = missing
                st.session_state.original_command = user_input

                missing_labels = [f"`{key}` ({intent})" for _, intent, key in missing]
                reply = (
                    f"Command parsed, but some parameters are missing.\n\n"
                    f"Please provide: {', '.join(missing_labels)}\n\n"
                    f"You can reply with values separated by commas, or use `key: value` format."
                )
                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
            else:
                # Complete — show payload and execute
                exec_results = execute_payload(result)
                reply = "Command parsed successfully. Here is the API payload:"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "json_payload": result,
                    "exec_results": exec_results,
                })
                with st.chat_message("assistant"):
                    st.markdown(reply)
                    st.json(result, expanded=True)
                    for er in exec_results:
                        st.markdown(f'<div class="exec-card">{er}</div>', unsafe_allow_html=True)
