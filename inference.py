"""
INFERENCE ENGINE v3 — Local
==============================
Run: py -3.11 inference_v3_local.py
"""
import json
import re
import torch
from difflib import SequenceMatcher
from transformers import AutoTokenizer, T5ForConditionalGeneration

OUTPUT_DIR = "flan_t5_api_model_v3"
PREFIX = "translate command to API: "
MAX_INPUT_LEN = 48
MAX_TARGET_LEN = 64

EXECUTORS = {
    "send_email":        lambda p: f"Email sent to {p.get('to','?')} | Subj: {p.get('subject','?')}",
    "schedule_meeting":  lambda p: f"Meeting with {p.get('person','?')} on {p.get('date','?')} at {p.get('time','?')}",
    "set_reminder":      lambda p: f"Reminder: {p.get('task','?')} on {p.get('date','?')} at {p.get('time','?')}",
    "send_message":      lambda p: f"Message to {p.get('to','?')}: {p.get('message','?')}",
    "create_task":       lambda p: f"Task: {p.get('title','?')} due {p.get('due_date','?')}",
    "search_web":        lambda p: f"Searching: {p.get('query','?')}",
    "play_music":        lambda p: f"Playing {p.get('song','?')} by {p.get('artist','?')}",
    "set_alarm":         lambda p: f"Alarm at {p.get('time','?')} - {p.get('label','?')}",
    "get_weather":       lambda p: f"Weather for {p.get('location','?')}",
    "book_cab":          lambda p: f"Cab: {p.get('pickup','?')} -> {p.get('destination','?')}",
}


def linear_to_json(linear_str):
    actions = []
    try:
        for block in linear_str.split("[INTENT]"):
            block = block.strip()
            if not block: continue
            intent_part = block.split("[PARAMS]")
            intent = intent_part[0].strip()
            if not intent: continue
            params = {}
            if len(intent_part) > 1:
                ps = intent_part[1].replace("[END]", "").strip()
                if ps:
                    for pair in ps.split("|"):
                        pair = pair.strip()
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            k, v = k.strip(), v.strip()
                            params[k] = None if v == "NULL" else v
            actions.append({"intent": intent, "parameters": params})
    except Exception:
        pass
    if actions:
        return {"actions": actions}
    return None


# === ENTITY CORRECTION ===
NAME_PATTERNS = [
    r'(?:send|drop|shoot|fire|forward)\s+(?:a\s+|an\s+)?(?:email|mail|message|text|note)\s+to\s+(.+?)\s+(?:about|regarding|saying|that|with)',
    r'(?:email|mail|write)\s+to\s+(.+?)\s+(?:about|regarding|with|saying)',
    r'^(?:email|mail)\s+(.+?)\s+(?:about|regarding|with)',
    r'(?:send|compose|draft|write|drop)\s+(?:a\s+|an\s+)?(?:email|mail)\s+to\s+(.+?)(?:\s*$|\s+and\b|\.\s)',
    r'(?:email|mail)\s+(.+?)(?:\s*$|\s+and\b|\.\s)',
    r'(?:call|text|ping|contact|ring)\s+(.+?)\s+(?:and|to|about|saying|that|regarding)',
    r'(?:message)\s+(?!to\b)(.+?)\s+(?:and|to|about|saying|that|regarding)',
    r'(?:let|tell|remind|notify|inform|alert)\s+(.+?)\s+(?:know|that|about|regarding|of)',
    r'(?:meeting|call|appointment|session)\s+with\s+(.+?)\s+(?:on|at|tomorrow|today|next|this|for)',
    r'(?:meet)\s+(.+?)\s+(?:on|at|tomorrow|today|next|this|for)',
    r'(?:cab|ride|taxi|uber)\s+for\s+(.+?)\s+(?:to|from|at)',
    r'(?:send|can you send)\s+(.+?)\s+an?\s+(?:email|mail)',
    r'with\s+(.+?)(?:\s*$|\s*,|\s+and\b)',
    r'(?:text)\s+(.+?)\s+(?:that|saying)',
]
PERSON_PARAMS = {"to", "person", "name", "recipient", "contact", "attendee"}
SKIP_COPY = {"time", "date", "duration", "priority", "repeat", "day", "hour", "minute", "count", "amount", "due_date"}


def extract_name(command):
    for pattern in NAME_PATTERNS:
        m = re.search(pattern, command, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'\s+(?:him|her|them|his|their)\s*$', '', name, flags=re.IGNORECASE)
            if len(name.split()) <= 4:
                return name
    return None


def _ngrams(text, max_n=8):
    words = text.split()
    out = []
    for n in range(1, min(max_n, len(words)) + 1):
        for i in range(len(words) - n + 1):
            out.append(" ".join(words[i:i+n]))
    return out


def _align(gen_val, input_text):
    if not gen_val or not input_text: return gen_val
    gl = gen_val.strip().lower()
    best, best_r = gen_val, 0.0
    for c in _ngrams(input_text):
        r = SequenceMatcher(None, gl, c.lower()).ratio()
        if r > best_r:
            best_r = r
            best = c
    return best if best_r >= 0.55 else gen_val


def apply_entity_correction(result, command):
    if not result or "actions" not in result: return result
    name = extract_name(command)
    for action in result["actions"]:
        params = action.get("parameters", {})
        for key, value in params.items():
            if value is None or key.lower() in SKIP_COPY: continue
            if key.lower() in PERSON_PARAMS and name is not None:
                r = SequenceMatcher(None, value.lower(), name.lower()).ratio()
                params[key] = name if r < 0.5 else _align(value, command)
            elif isinstance(value, str) and len(value) > 1:
                params[key] = _align(value, command)
    return result


# === GENERATION ===
def generate(model, tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors="pt",
                       truncation=True, max_length=MAX_INPUT_LEN).to(device)
    configs = [
        dict(num_beams=5, early_stopping=True, repetition_penalty=1.3, length_penalty=1.0),
        dict(num_beams=8, early_stopping=True, repetition_penalty=1.0, length_penalty=1.0),
        dict(num_beams=1, do_sample=False),
    ]
    for i, cfg in enumerate(configs):
        with torch.no_grad():
            gen_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                                     max_length=MAX_TARGET_LEN, **cfg)
        decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        result = linear_to_json(decoded)
        if result is not None:
            return decoded, result, i
    return decoded, None, -1


def run_inference(command, model=None, tokenizer=None):
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        model = T5ForConditionalGeneration.from_pretrained(OUTPUT_DIR, torch_dtype=dtype).to(device)
        model.eval()

    device = next(model.parameters()).device
    raw, result, idx = generate(model, tokenizer, PREFIX + command, device)
    names = ["beam+penalty", "wide-beam", "greedy", "FAILED"]
    used = names[idx] if idx >= 0 else "ALL FAILED"

    print(f"\n{'='*70}")
    print(f"  Command  : {command}")
    print(f"  Raw Out  : {raw}")
    print(f"  Strategy : {used}")

    if result is None:
        print(f"  [FAILED] Could not parse.")
        print(f"{'='*70}")
        return None

    result = apply_entity_correction(result, command)
    print(f"  [PARSED] {json.dumps(result)}")
    print(f"{'='*70}")

    for i, action in enumerate(result.get("actions", [])):
        intent = action.get("intent", "UNKNOWN")
        params = action.get("parameters", {})
        missing = [k for k, v in params.items() if v is None]
        if missing:
            print(f"  [{i+1}] {intent}: INCOMPLETE — slot filling needed")
            for m in missing:
                print(f"      -> Please provide: '{m}'")
        else:
            ex = EXECUTORS.get(intent)
            if ex:
                print(f"  [{i+1}] {intent}: EXECUTED ✓")
                print(f"      {ex(params)}")
            else:
                print(f"  [{i+1}] {intent}: No executor registered")
    return result


def run_tests():
    print("\n" + "="*70)
    print(" INFERENCE TEST SUITE v3")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    model = T5ForConditionalGeneration.from_pretrained(OUTPUT_DIR, torch_dtype=dtype).to(device)
    model.eval()
    print(f"Loaded ({'BF16' if dtype == torch.bfloat16 else 'FP32'})")

    if device == "cuda":
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"VRAM: {vram:.2f} GB")

    tests = [
        "Schedule a meeting with Deepak tomorrow at 3pm",
        "Send an email to Bob about the report and set a reminder for Friday at noon",
        "Send an email to John",
        "Book a meeting on Saturday",
        "Email Priya about the project and schedule a call with Raj",
        "Fire off a mail to the professor regarding the thesis",
        "Call Krishnamurthy and remind him about the deadline",
        "Send an email to Raghunath about the quarterly review",
        "Call Eklavya and remind him about the deadline",
        "Text Jean-Pierre that I will be late",
        "Let Chen know that the meeting is rescheduled",
        "Remind Paarth about the submission deadline",
        "Notify Dr. Raghav Krishnamurthy about the quarterly review",
    ]

    passed = 0
    for cmd in tests:
        result = run_inference(cmd, model=model, tokenizer=tokenizer)
        if result and isinstance(result.get("actions"), list):
            passed += 1

    print(f"\n{'='*70}")
    print(f" Results: {passed}/{len(tests)} commands parsed successfully")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_tests()
