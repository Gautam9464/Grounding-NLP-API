"""
TRAINING SCRIPT v3 — Local RTX 4060 (8GB VRAM)
=================================================
Setup:
  py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  py -3.11 -m pip install transformers pandas

Run:
  py -3.11 train_v3_local.py
"""
import os
import sys
import json
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "google/flan-t5-base"
PREFIX = "translate command to API: "

TRAIN_PATH = "train_dataset_5k.csv"
TEST_PATH = "test_dataset_1k.csv"
OUTPUT_DIR = "flan_t5_api_model_v3"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "training_state.pt")

EPOCHS = 12
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4         # effective batch = 16
LR = 3e-4
MAX_INPUT_LEN = 48
MAX_TARGET_LEN = 64
WARMUP_RATIO = 0.06
VAL_SPLIT = 0.1
FULL_EVAL_EVERY = 4
PRINT_EVERY = 100


# ============================================================
# STARTUP CHECKS
# ============================================================
def startup_checks():
    # CUDA check
    if not torch.cuda.is_available():
        print("=" * 70)
        print("ERROR: PyTorch cannot see your GPU!")
        print()
        print("Install CUDA PyTorch:")
        print("  py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("Verify: py -3.11 -c \"import torch; print(torch.cuda.is_available())\"")
        print("=" * 70)
        sys.exit(1)

    # Dataset check
    for path, label in [(TRAIN_PATH, "Train"), (TEST_PATH, "Test")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} dataset not found at '{path}'")
            print(f"Place '{os.path.basename(path)}' in the same folder as this script.")
            sys.exit(1)

    # Dataset validation
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    dupes = train_df.duplicated(subset=["command"]).sum()
    overlap = len(set(train_df["command"]).intersection(set(test_df["command"])))

    body_hall = 0
    for _, row in train_df.iterrows():
        obj = json.loads(row["api_call"])
        for a in obj["actions"]:
            if a["intent"] == "send_email":
                b = a["parameters"].get("body")
                if b is not None and b.lower() not in row["command"].lower():
                    body_hall += 1

    issues = []
    if dupes > 0: issues.append(f"Train has {dupes} duplicate commands")
    if overlap > 0: issues.append(f"Train-test overlap: {overlap} commands")
    if body_hall > 0: issues.append(f"Email body hallucinated: {body_hall} rows")

    if issues:
        print("WARNING: Dataset issues found:")
        for i in issues: print(f"  - {i}")
        print("You may be using old datasets. Use the new clean ones.")
        resp = input("Continue anyway? (y/n): ").strip().lower()
        if resp != "y": sys.exit(0)

    print(f"Datasets OK: train={len(train_df)}, test={len(test_df)}, "
          f"dupes={dupes}, overlap={overlap}, body_hall={body_hall}")


# ============================================================
# LINEARIZATION
# ============================================================
def json_to_linear(api_call_str):
    obj = json.loads(api_call_str)
    parts = []
    for action in obj["actions"]:
        intent = action["intent"]
        params = action.get("parameters", {})
        pp = []
        for k, v in params.items():
            val = "NULL" if v is None else str(v)
            pp.append(f"{k} = {val}")
        parts.append(f"[INTENT] {intent} [PARAMS] {' | '.join(pp)} [END]")
    return " ".join(parts)


def linear_to_json(linear_str):
    actions = []
    try:
        for block in linear_str.split("[INTENT]"):
            block = block.strip()
            if not block: continue
            intent_part = block.split("[PARAMS]")
            intent = intent_part[0].strip()
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
    return {"error": "parse failed", "raw": linear_str}


# ============================================================
# DATASET
# ============================================================
class APICallDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len, max_target_len, prefix):
        self.data = df.reset_index(drop=True)
        self.tok = tokenizer
        self.mil = max_input_len
        self.mtl = max_target_len
        self.prefix = prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src = self.tok(self.prefix + str(row["command"]),
                       max_length=self.mil, padding="max_length",
                       truncation=True, return_tensors="pt")
        tgt = self.tok(str(row["api_call_linear"]),
                       max_length=self.mtl, padding="max_length",
                       truncation=True, return_tensors="pt")
        labels = tgt["input_ids"].squeeze().clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": labels,
        }


# ============================================================
# METRICS
# ============================================================
def validate_linear(s):
    r = linear_to_json(s)
    return "actions" in r and len(r.get("actions", [])) > 0

def compute_intent_accuracy(pred_str, ref_str):
    pred = linear_to_json(pred_str)
    ref = linear_to_json(ref_str)
    if "error" in pred or "error" in ref: return False
    return sorted([a["intent"] for a in pred.get("actions", [])]) == \
           sorted([a["intent"] for a in ref.get("actions", [])])

@torch.no_grad()
def evaluate_loss_only(model, loader, device, use_bf16):
    model.eval()
    total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        else:
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        total += loss.item()
    return total / max(len(loader), 1)

@torch.no_grad()
def evaluate_full(model, tokenizer, loader, device, use_bf16, use_beam=False):
    model.eval()
    total_loss = 0
    valid_parse = exact_match = intent_match = total = 0
    gen_kw = dict(max_length=MAX_TARGET_LEN)
    if use_beam: gen_kw.update(num_beams=4, early_stopping=True)

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        else:
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        total_loss += loss.item()

        gen = model.generate(input_ids=ids, attention_mask=mask, **gen_kw)
        preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
        rl = labels.clone()
        rl[rl == -100] = tokenizer.pad_token_id
        refs = tokenizer.batch_decode(rl, skip_special_tokens=True)

        for p, r in zip(preds, refs):
            total += 1
            if validate_linear(p): valid_parse += 1
            if p.strip() == r.strip(): exact_match += 1
            if compute_intent_accuracy(p, r): intent_match += 1

    t = max(total, 1)
    return {
        "loss": total_loss / max(len(loader), 1),
        "parse_acc": valid_parse / t,
        "em_acc": exact_match / t,
        "intent_acc": intent_match / t,
    }


# ============================================================
# CHECKPOINT
# ============================================================
def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }, CHECKPOINT_PATH)

def load_checkpoint(model, optimizer, scheduler, device):
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[RESUME] Found checkpoint at {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start = ckpt["epoch"] + 1
        best = ckpt["best_val_loss"]
        print(f"[RESUME] Resuming from epoch {start + 1}, best_val_loss={best:.4f}")
        return start, best
    return 0, float("inf")


# ============================================================
# TRAINING
# ============================================================
def train():
    startup_checks()
    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU    : {gpu_name} ({vram_gb:.1f} GB)")
    print(f"BF16   : {use_bf16}")
    print(f"Model  : {MODEL_NAME}")
    print(f"Input  : max {MAX_INPUT_LEN} tokens")
    print(f"Target : max {MAX_TARGET_LEN} tokens")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Gradient checkpointing — saves ~60% activation VRAM
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print("Gradient checkpointing: ENABLED")

    model = model.to(device)
    alloc = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after model load: {alloc:.2f} GB / {vram_gb:.1f} GB")

    train_full = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {len(train_full)} rows, Test: {len(test_df)} rows")

    print("Linearizing...")
    train_full["api_call_linear"] = train_full["api_call"].apply(json_to_linear)
    test_df["api_call_linear"] = test_df["api_call"].apply(json_to_linear)

    trunc = sum(1 for i in range(len(train_full))
                if len(tokenizer.encode(train_full.iloc[i]["api_call_linear"])) > MAX_TARGET_LEN)
    if trunc:
        print(f"[WARN] {trunc} targets exceed {MAX_TARGET_LEN} tokens — WRONG DATASET?")
    else:
        print(f"All targets fit in {MAX_TARGET_LEN} tokens ✓")

    failures = sum(1 for i in range(len(train_full))
                   if json.dumps(linear_to_json(train_full.iloc[i]["api_call_linear"]), sort_keys=True)
                   != json.dumps(json.loads(train_full.iloc[i]["api_call"]), sort_keys=True))
    print(f"Round-trip: {len(train_full)-failures}/{len(train_full)} passed")

    split_idx = int((1 - VAL_SPLIT) * len(train_full))
    train_df = train_full.iloc[:split_idx]
    val_df = train_full.iloc[split_idx:]
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_ds = APICallDataset(train_df, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN, PREFIX)
    val_ds = APICallDataset(val_df, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN, PREFIX)
    test_ds = APICallDataset(test_df, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN, PREFIX)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    eff = BATCH_SIZE * GRAD_ACCUM_STEPS
    print(f"\nSteps/epoch: {len(train_loader)}, Total opt steps: {total_steps}")
    print(f"Batch: {BATCH_SIZE}, Grad accum: {GRAD_ACCUM_STEPS}, Effective: {eff}")
    print(f"Full eval every: {FULL_EVAL_EVERY} epochs")
    print(f"Saving to: {OUTPUT_DIR}/")
    if start_epoch > 0:
        print(f"Skipping epochs 1-{start_epoch} (already done)")
    print(f"{'='*80}\n")

    t_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                    loss = outputs.loss / GRAD_ACCUM_STEPS
            else:
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            if epoch == start_epoch and step == 0:
                rl = outputs.loss.item()
                print(f"[DEBUG] First batch loss: {rl:.4f}")
                if rl == 0.0 or torch.isnan(outputs.loss):
                    print("[FATAL] Loss is 0 or NaN. Aborting.")
                    return
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"[DEBUG] Peak VRAM: {peak:.2f} GB / {vram_gb:.1f} GB")

            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += outputs.loss.item()

            if (step + 1) % PRINT_EVERY == 0:
                avg = epoch_loss / (step + 1)
                el = time.time() - t0
                eta = el / (step + 1) * (len(train_loader) - step - 1)
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | "
                      f"Loss: {avg:.4f} | ETA: {eta:.0f}s")

        avg_train = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        do_full = ((epoch + 1) % FULL_EVAL_EVERY == 0) or (epoch + 1 == EPOCHS)

        if do_full:
            vm = evaluate_full(model, tokenizer, val_loader, device, use_bf16)
            print(f"Epoch {epoch+1:02d}/{EPOCHS} [{elapsed:.0f}s] | "
                  f"Train: {avg_train:.4f} | Val: {vm['loss']:.4f} | "
                  f"Parse: {vm['parse_acc']:.1%} | Intent: {vm['intent_acc']:.1%} | "
                  f"EM: {vm['em_acc']:.1%}")
            val_loss = vm["loss"]
        else:
            val_loss = evaluate_loss_only(model, val_loader, device, use_bf16)
            print(f"Epoch {epoch+1:02d}/{EPOCHS} [{elapsed:.0f}s] | "
                  f"Train: {avg_train:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  -> Best model saved (val={val_loss:.4f})")

        save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss)
        print(f"  -> Checkpoint saved (epoch {epoch+1})")

    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time/60:.1f} minutes")

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint removed (training complete).")

    # === FINAL TEST ===
    print(f"\n{'='*80}")
    print("FINAL TEST SET EVALUATION (1000 unseen commands, beam search)")
    print(f"{'='*80}")

    model = T5ForConditionalGeneration.from_pretrained(
        OUTPUT_DIR, torch_dtype=torch.bfloat16 if use_bf16 else torch.float32
    ).to(device)
    tm = evaluate_full(model, tokenizer, test_loader, device, use_bf16, use_beam=True)

    print(f"  Test Loss    : {tm['loss']:.4f}")
    print(f"  Parse Acc    : {tm['parse_acc']:.1%}")
    print(f"  Intent Acc   : {tm['intent_acc']:.1%}")
    print(f"  Exact Match  : {tm['em_acc']:.1%}")
    print(f"\n  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Model at     : {OUTPUT_DIR}/")


if __name__ == "__main__":
    train()
