"""
Attack-detection RF + LoRA fine-tuning script (SSH/server version).

Setup before running:
    pip uninstall -y transformers trl peft accelerate
    pip install -q transformers==4.46.3 trl==0.11.4 peft==0.13.2 accelerate==1.0.1 bitsandbytes datasets

    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   # must have accepted the Llama-3.2-3B-Instruct license

Run (recommended inside tmux/screen so it survives SSH disconnects):
    tmux new -s training
    python3 train_script.py 2>&1 | tee training.log
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────
# TODO: update these to the actual location of your dataset/output on this server
INPUT_DIR = "/path/to/your/dataset"        # placeholder — was /kaggle/input/...
WORK_DIR = "/path/to/your/working/dir"     # placeholder — was /kaggle/working

RF_DIR = f"{WORK_DIR}/rf"
JSONL_DIR = f"{WORK_DIR}/jsonl"
MODEL_DIR = f"{WORK_DIR}/model_out"
FINAL_DIR = f"{MODEL_DIR}/final"

TRAIN_CSV = f"{INPUT_DIR}/train_augmented.csv"   # for RF training
TEST_CSV = f"{INPUT_DIR}/test_clean.csv"          # for honest evaluation
TRAIN_JSONL = f"{INPUT_DIR}/train.jsonl"
VAL_JSONL = f"{INPUT_DIR}/val.jsonl"

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Create output directories
for d in [RF_DIR, JSONL_DIR, MODEL_DIR, FINAL_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

print("✓ Imports done")
print(f"  CUDA available : {torch.cuda.is_available()}")
print(f"  GPU count      : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}          : {torch.cuda.get_device_name(i)}")

# ── HF token (set via `export HF_TOKEN=...` before running) ──
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable not set. "
        "Run: export HF_TOKEN=hf_xxxxxxxx before launching this script."
    )
print("✓ HuggingFace token loaded")

# ── Load data ──────────────────────────────────
df_train = pd.read_csv(f"{INPUT_DIR}/train_augmented.csv")
df_test = pd.read_csv(f"{INPUT_DIR}/test_clean.csv")
df_val = pd.read_csv(f"{INPUT_DIR}/val_clean.csv")

print(f"Train shape : {df_train.shape}")
print(f"Val shape   : {df_val.shape}")
print(f"Test shape  : {df_test.shape}")
print(f"\nTrain label distribution:")
print(df_train["label"].value_counts().to_string())
print(f"\nTest label distribution:")
print(df_test["label"].value_counts().to_string())
print(f"\nMissing values (train) : {df_train.isnull().sum().sum()}")

# ── Prepare features ───────────────────────────
META_COLS = ["label", "run", "filename", "aug_method"]

drop_cols = [c for c in META_COLS if c in df_train.columns]
X_train = df_train.drop(columns=drop_cols).select_dtypes(include=[np.number])
y_train = df_train["label"]
feature_names = X_train.columns.tolist()

X_test = df_test.drop(columns=[c for c in META_COLS if c in df_test.columns])
X_test = X_test.select_dtypes(include=[np.number])[feature_names]
y_test = df_test["label"]

le = LabelEncoder()
le.fit(y_train)
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

print(f"Features     : {len(feature_names)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples : {len(X_test)}")
print(f"Classes      : {list(le.classes_)}")

# ── Cross-validation ───────────────────────────
print("\nRunning 5-fold cross-validation on train set ...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
cv_scores = cross_val_score(rf, X_train, y_train_enc, cv=5, scoring="f1_macro", n_jobs=-1)
print(f"CV F1-macro : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\nFitting final model on train set ...")
rf.fit(X_train, y_train_enc)

y_pred_test = rf.predict(X_test)
print("\n" + "=" * 50)
print("REAL Classification Report (unseen test set):")
print("=" * 50)
print(classification_report(y_test_enc, y_pred_test, target_names=le.classes_))

y_pred_train = rf.predict(X_train)
print("Train set report (expect near 100% — just for comparison):")
print(classification_report(y_train_enc, y_pred_train, target_names=le.classes_))

cm = confusion_matrix(y_test_enc, y_pred_test)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_,
            yticklabels=le.classes_, ax=ax, cmap="Blues")
ax.set_title("Random Forest — Confusion Matrix (UNSEEN TEST SET)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{RF_DIR}/confusion_matrix.png")
plt.close(fig)  # no display in a headless script

importances = pd.Series(rf.feature_importances_, index=feature_names).nlargest(15)
print("\nTop 15 features:")
print(importances.to_string())

joblib.dump(rf, f"{RF_DIR}/random_forest.pkl")
joblib.dump(le, f"{RF_DIR}/label_encoder.pkl")
with open(f"{RF_DIR}/feature_names.json", "w") as f:
    json.dump(feature_names, f)

print(f"\n✓ RF model saved → {RF_DIR}/")

# ── Sanity check ───────────────────────────────
rf_test = joblib.load(f"{RF_DIR}/random_forest.pkl")
le_test = joblib.load(f"{RF_DIR}/label_encoder.pkl")
with open(f"{RF_DIR}/feature_names.json") as f:
    fn_test = json.load(f)

sample_pred = rf_test.predict(X_train.iloc[:5])
for i, p in enumerate(sample_pred):
    print(f"  Row {i}: predicted={le_test.inverse_transform([p])[0]}  actual={y_train.iloc[i]}")
print(f"\n✓ RF artifacts verified  ({len(fn_test)} features)")

# ── Build SFT JSONL records ────────────────────
ATTACK_EXPLANATIONS = {
    "brute_force": "Repeated login attempts — many small TCP flows to same port, high RST/FIN counts.",
    "slowloris": "Slow HTTP DoS — many source IPs holding partial connections open, tiny bytes/flow.",
    "dos_http_flood": "HTTP flood — massive GET/POST volume to port 80/8080, high PSH+ACK ratio.",
    "dos_syn_flood": "SYN flood — near-100% SYN-only packets, no ACK completion, half-open connections.",
    "dos_udp_flood": "UDP flood — high UDP ratio, random destination ports, high port entropy.",
    "mixed_attack": "Multi-vector — simultaneous SYN, UDP, and HTTP anomalies across many ports.",
    "port_scan": "Port scan — very high unique destination ports, high RST count, tiny flows.",
    "replay_attack": "Replay attack — high repeated payload fingerprints, duplicate packet sequences.",
    "rtsp_brute_force": "RTSP brute force — many connection attempts to port 554/8554, high rtsp_pkts.",
    "normal": "Normal traffic — low packet rate, balanced protocols, expected port distribution.",
}

SYSTEM_PROMPT = """You are a cybersecurity expert specializing in IoT and CCTV network security. \
Given a network traffic feature matrix and a Random Forest classifier prediction, confirm or \
correct the classification and explain the key indicators. \
Valid classes: brute_force, slowloris, dos_http_flood, dos_syn_flood, dos_udp_flood, \
mixed_attack, port_scan, replay_attack, rtsp_brute_force, normal."""

FEATURE_GROUPS = {
    "Volume & Timing": ["total_packets", "duration_sec", "packets_per_sec", "bytes_total", "bytes_per_sec"],
    "Inter-Arrival Time": ["iat_mean", "iat_std", "iat_min", "iat_max", "iat_median"],
    "Packet Size": ["pkt_size_mean", "pkt_size_std", "pkt_size_min", "pkt_size_max"],
    "Protocol Counts": ["count_tcp", "count_udp", "count_icmp", "count_arp", "ratio_tcp", "ratio_udp"],
    "TCP Flags": ["flag_syn", "flag_ack", "flag_fin", "flag_rst", "syn_only_count", "ratio_syn_only", "ratio_syn_ack"],
    "Ports": ["unique_src_ports", "unique_dst_ports", "port_entropy_dst", "http_pkts", "rtsp_pkts", "ssh_pkts"],
    "IP Diversity": ["unique_src_ips", "unique_dst_ips", "src_ip_entropy"],
    "Payload": ["payload_mean", "n_has_payload", "max_payload_repeat", "n_http_get", "n_http_post"],
    "Flows": ["n_flows", "pkts_per_flow_mean", "pkts_per_flow_max", "bytes_per_flow_mean"],
    "ARP": ["arp_who_has", "arp_is_at"],
}


def build_matrix_text(row):
    lines = []
    for group, feats in FEATURE_GROUPS.items():
        present = [f for f in feats if f in row.index]
        if not present:
            continue
        lines.append(f"[{group}]")
        for f in present:
            val = row.get(f, "N/A")
            if isinstance(val, float):
                val = round(val, 4)
            lines.append(f"  {f:<28} = {val}")
        lines.append("")
    return "\n".join(lines)


def build_records(source_df, source_X):
    """Build JSONL records for a given dataframe split."""
    probs = rf.predict_proba(source_X)
    pred_labels = le.inverse_transform(np.argmax(probs, axis=1))
    confidences = probs.max(axis=1)
    top5_idx = np.argsort(rf.feature_importances_)[::-1][:5]
    records = []

    for i, (_, row) in enumerate(source_df.iterrows()):
        true_label = str(row.get("label", "unknown")).strip()
        rf_label = pred_labels[i]
        rf_conf = round(float(confidences[i]) * 100, 1)
        match = true_label == rf_label

        top5_str = "\n".join([
            f"  {feature_names[j]:<28} = {round(float(source_X.iloc[i][feature_names[j]]), 4)}"
            for j in top5_idx
        ])

        def g(col):
            v = row.get(col, None)
            return round(float(v), 4) if v is not None and str(v) != "nan" else "N/A"

        indicator_map = {
            "brute_force": f"n_flows={g('n_flows')}, top_flow_pkts={g('top_flow_pkts')}, flag_rst={g('flag_rst')}",
            "slowloris": f"n_http_conn_ips={g('n_http_conn_ips')}, bytes_per_flow_mean={g('bytes_per_flow_mean')}",
            "dos_http_flood": f"n_http_get={g('n_http_get')}, packets_per_sec={g('packets_per_sec')}",
            "dos_syn_flood": f"syn_only_count={g('syn_only_count')}, ratio_syn_only={g('ratio_syn_only')}, ratio_syn_ack={g('ratio_syn_ack')}",
            "dos_udp_flood": f"count_udp={g('count_udp')}, port_entropy_dst={g('port_entropy_dst')}",
            "mixed_attack": f"flag_syn={g('flag_syn')}, count_udp={g('count_udp')}, unique_dst_ports={g('unique_dst_ports')}",
            "port_scan": f"unique_dst_ports={g('unique_dst_ports')}, port_entropy_dst={g('port_entropy_dst')}, flag_rst={g('flag_rst')}",
            "replay_attack": f"max_payload_repeat={g('max_payload_repeat')}, n_has_payload={g('n_has_payload')}",
            "rtsp_brute_force": f"rtsp_pkts={g('rtsp_pkts')}, ratio_rtsp={g('ratio_rtsp')}, n_flows={g('n_flows')}",
            "normal": f"packets_per_sec={g('packets_per_sec')}, port_entropy_dst={g('port_entropy_dst')}",
        }

        user_msg = (
            f"Network capture feature matrix (SPARSH CCTV, 10-second window):\n\n"
            f"{build_matrix_text(row)}\n"
            f"Random Forest prediction:\n"
            f"  Predicted class : {rf_label}\n"
            f"  Confidence      : {rf_conf}%\n\n"
            f"Top 5 contributing features:\n{top5_str}\n\n"
            f"Confirm the classification and explain the key indicators."
        )

        answer = (
            f"**Classification {'CONFIRMED' if match else 'CORRECTED'}: {true_label.upper()}**\n\n"
            f"{'RF prediction is correct.' if match else f'Correct label is {true_label}, not {rf_label}.'}\n\n"
            f"{ATTACK_EXPLANATIONS.get(true_label, '')}\n\n"
            f"Key indicators: {indicator_map.get(true_label, '')}"
        )

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": answer},
            ]
        })

    return records


drop_cols_inf = [c for c in META_COLS if c in df_val.columns]
X_val = df_val.drop(columns=drop_cols_inf).select_dtypes(include=[np.number])[feature_names]
X_test_jsonl = df_test.drop(columns=drop_cols_inf).select_dtypes(include=[np.number])[feature_names]

print("Building train records ...")
train_records = build_records(df_train, X_train)

print("Building val records ...")
val_records = build_records(df_val, X_val)

print("Building test records ...")
test_records = build_records(df_test, X_test_jsonl)

Path(JSONL_DIR).mkdir(parents=True, exist_ok=True)

for split_name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
    out_path = f"{JSONL_DIR}/{split_name}.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"✓ {split_name}.jsonl → {len(records)} records → {out_path}")

with open(f"{JSONL_DIR}/train.jsonl") as f:
    sample = json.loads(f.readline())

print("Sample JSONL record:\n")
for msg in sample["messages"]:
    print(f"[{msg['role'].upper()}]")
    print(msg["content"][:300])
    print()

for split in ["train", "val", "test"]:
    path = f"{JSONL_DIR}/{split}.jsonl"
    with open(path) as f:
        count = sum(1 for _ in f)
    print(f"{split}.jsonl : {count} records")

# ── Tokenizer ───────────────────────────────────
print(f"Loading tokenizer for {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"✓ Tokenizer loaded")
print(f"  Vocab size     : {tokenizer.vocab_size}")
print(f"  Pad token      : {tokenizer.pad_token}")

test_text = "This is a test sentence for the CCTV attack detector."
tokens = tokenizer(test_text, return_tensors="pt")
print(f"  Test encode    : {tokens['input_ids'].shape[1]} tokens for '{test_text[:40]}...'")


def format_chat(example):
    text = "<|begin_of_text|>"
    for msg in example["messages"]:
        text += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>\n"
    text += "<|end_of_text|>"
    return {"text": text}


print("Loading and formatting JSONL datasets ...")
dataset = load_dataset(
    "json",
    data_files={
        "train": f"{JSONL_DIR}/train.jsonl",
        "validation": f"{JSONL_DIR}/val.jsonl",
    }
)
dataset = dataset.map(format_chat, remove_columns=["messages"])

print(f"✓ Dataset loaded")
print(f"  Train      : {len(dataset['train'])} samples")
print(f"  Validation : {len(dataset['validation'])} samples")
print(f"\nSample formatted text (first 500 chars):")
print(dataset["train"][0]["text"][:500])

# ── Load model in 4-bit ─────────────────────────
print(f"Loading {MODEL_ID} in 4-bit ...")
print("(This may take 5-10 minutes to download)\n")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
    torch_dtype=torch.float16,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

print(f"\n✓ Model loaded")
print(f"  dtype  : {next(model.parameters()).dtype}")
print(f"  device : {next(model.parameters()).device}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem_used = torch.cuda.memory_allocated(i) / 1e9
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i} memory : {mem_used:.1f} GB used / {mem_total:.1f} GB total")

# ── LoRA ─────────────────────────────────────────
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: trainable params ~20M out of 3.2B (~0.63%)

print("\n✓ LoRA adapters applied")

import transformers
import trl
import peft

print("transformers:", transformers.__version__)
print("trl:", trl.__version__)
print("peft:", peft.__version__)

# ── Training ──────────────────────────────────────
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,      # effective batch = 16
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    fp16=False,
    bf16=False,                         # set True instead if your GPU supports bf16 (e.g. A100/4090, not T4)
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    group_by_length=True,
    dataloader_num_workers=2,
    report_to="none",                   # set to "wandb" if you want experiment tracking
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
    args=training_args,
)

print("✓ Trainer configured")
print(f"  Effective batch size : {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Total epochs         : {training_args.num_train_epochs}")
print(f"  Steps per epoch      : {len(dataset['train']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")

print("Starting training ...")
train_result = trainer.train()

print(f"\n✓ Training complete")
print(f"  Total steps    : {train_result.global_step}")
print(f"  Training loss  : {train_result.training_loss:.4f}")

print(f"Saving model to {FINAL_DIR} ...")
trainer.model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print("\n✓ Saved files:")
for f in sorted(Path(FINAL_DIR).iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:<45} {size_mb:.1f} MB")

# ── Quick inference test ────────────────────────
TEST_PROMPT = """Network capture feature matrix (SPARSH CCTV, 10-second window):

[Volume & Timing]
  total_packets              = 9821
  packets_per_sec            = 982.1
  bytes_per_sec               = 58860.0

[TCP Flags]
  flag_syn                   = 9750
  flag_ack                   = 42
  syn_only_count             = 9748
  ratio_syn_only             = 0.9926
  ratio_syn_ack              = 232.14

[Ports]
  unique_dst_ports           = 2
  rtsp_pkts                  = 9821

Random Forest prediction:
  Predicted class : dos_syn_flood
  Confidence      : 97.3%

Top 5 contributing features:
  syn_only_count             = 9748
  ratio_syn_only             = 0.9926
  ratio_syn_ack              = 232.14
  flag_syn                   = 9750
  packets_per_sec            = 982.1

Confirm the classification and explain the key indicators."""

prompt = (
    "<|begin_of_text|>"
    f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>\n"
    f"<|start_header_id|>user<|end_header_id|>\n\n{TEST_PROMPT}<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
model_answer = full_text.split("assistant")[-1].strip()

print("Model response:")
print("=" * 50)
print(model_answer)
print("=" * 50)
print("\n✓ Model is working correctly — script finished")
