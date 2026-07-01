"""
split_then_augment.py
=====================
CORRECT pipeline — splits by run folder FIRST, then augments only
the training set, keeping test set completely untouched.

Input  : features.csv  (raw output of extract_features.py, NO augmentation)
Outputs:
    splits/train_augmented.csv   ← augmented train set  (for RF + LLM training)
    splits/test_clean.csv        ← untouched test set   (for honest evaluation)
    splits/val_clean.csv         ← untouched val set    (for LLM validation)

Usage:
    python split_then_augment.py --input features.csv --output_dir ./splits
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

META_COLS    = ["label", "run", "filename"]
TARGET_ROWS  = 9500      # how many augmented train rows to generate
NOISE_STD    = 0.02      # gaussian noise std (2% of feature value)
RANDOM_SEED  = 42

# ─────────────────────────────────────────────
# STEP 1 — Split by run folder (no leakage)
# ─────────────────────────────────────────────

def split_by_run(df: pd.DataFrame):
    """
    Split run folders into train/val/test.
    All rows from a given run go into the SAME split —
    this prevents sliding-window leakage.
    """
    all_runs = sorted(df["run"].unique())
    print(f"  Total run folders : {len(all_runs)}")
    print(f"  Runs              : {all_runs}")

    # 70% train / 15% val / 15% test  (by run count)
    train_runs, temp_runs = train_test_split(
        all_runs, test_size=0.30, random_state=RANDOM_SEED
    )
    val_runs, test_runs = train_test_split(
        temp_runs, test_size=0.50, random_state=RANDOM_SEED
    )

    train_df = df[df["run"].isin(train_runs)].copy()
    val_df   = df[df["run"].isin(val_runs)].copy()
    test_df  = df[df["run"].isin(test_runs)].copy()

    print(f"\n  Train runs ({len(train_runs)}) : {sorted(train_runs)}")
    print(f"  Val runs   ({len(val_runs)})   : {sorted(val_runs)}")
    print(f"  Test runs  ({len(test_runs)})  : {sorted(test_runs)}")
    print(f"\n  Train rows : {len(train_df)}")
    print(f"  Val rows   : {len(val_df)}")
    print(f"  Test rows  : {len(test_df)}")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# STEP 2 — Augment ONLY the train set
# ─────────────────────────────────────────────

def augment(df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    """
    Expand train set to target_rows using:
    1. Gaussian noise on numeric features
    2. Feature scaling variation
    Both are label-preserving — only numeric values are perturbed.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_meta    = [c for c in META_COLS if c in df.columns]

    current_rows = len(df)
    needed       = max(0, target_rows - current_rows)

    if needed == 0:
        print(f"  Already have {current_rows} rows — no augmentation needed")
        return df

    print(f"  Original rows : {current_rows}")
    print(f"  Target rows   : {target_rows}")
    print(f"  Need to add   : {needed} synthetic rows")

    synthetic_batches = []
    generated = 0

    while generated < needed:
        # Sample a random batch from existing data
        batch_size = min(needed - generated, current_rows)
        sample = df.sample(n=batch_size, replace=True, random_state=generated)

        # ── Augmentation method 1: Gaussian noise ──
        noisy = sample.copy()
        for col in numeric_cols:
            col_std = df[col].std()
            if col_std > 0:
                noise = np.random.normal(
                    loc=0,
                    scale=NOISE_STD * col_std,
                    size=len(noisy)
                )
                noisy[col] = (noisy[col] + noise).clip(lower=0)

        noisy["aug_method"] = "gaussian_noise"
        noisy["run"]        = noisy["run"].astype(str) + "_aug"
        noisy["filename"]   = noisy["filename"].astype(str) + "_aug"
        synthetic_batches.append(noisy)
        generated += len(noisy)

    synthetic_df = pd.concat(synthetic_batches, ignore_index=True).head(needed)
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

    # Mark original rows
    augmented_df["aug_method"] = augmented_df.get("aug_method", pd.Series(["original"] * len(df)))
    augmented_df.loc[:current_rows-1, "aug_method"] = "original"

    print(f"  Final rows    : {len(augmented_df)}")
    print(f"\n  Label distribution after augmentation:")
    print(augmented_df["label"].value_counts().to_string())

    return augmented_df


# ─────────────────────────────────────────────
# STEP 3 — Build JSONL from a dataframe
# ─────────────────────────────────────────────

ATTACK_EXPLANATIONS = {
    "brute_force":      "Repeated login attempts — many small TCP flows to same port, high RST/FIN counts.",
    "slowloris":        "Slow HTTP DoS — many source IPs holding partial connections open, tiny bytes/flow.",
    "dos_http_flood":   "HTTP flood — massive GET/POST volume to port 80/8080, high PSH+ACK ratio.",
    "dos_syn_flood":    "SYN flood — near-100% SYN-only packets, no ACK completion, half-open connections.",
    "dos_udp_flood":    "UDP flood — high UDP ratio, random destination ports, high port entropy.",
    "mixed_attack":     "Multi-vector — simultaneous SYN, UDP, and HTTP anomalies across many ports.",
    "port_scan":        "Port scan — very high unique destination ports, high RST count, tiny flows.",
    "replay_attack":    "Replay attack — high repeated payload fingerprints, duplicate packet sequences.",
    "rtsp_brute_force": "RTSP brute force — many connection attempts to port 554/8554, high rtsp_pkts.",
    "normal":           "Normal traffic — low packet rate, balanced protocols, expected port distribution.",
}

SYSTEM_PROMPT = """You are a cybersecurity expert specializing in IoT and CCTV network security. \
Given a network traffic feature matrix from a SPARSH CCTV camera, classify the traffic and \
explain the key indicators. \
Valid classes: brute_force, slowloris, dos_http_flood, dos_syn_flood, dos_udp_flood, \
mixed_attack, port_scan, replay_attack, rtsp_brute_force, normal."""

FEATURE_GROUPS = {
    "Volume & Timing":    ["total_packets","duration_sec","packets_per_sec","bytes_total","bytes_per_sec"],
    "Inter-Arrival Time": ["iat_mean","iat_std","iat_min","iat_max","iat_median"],
    "Packet Size":        ["pkt_size_mean","pkt_size_std","pkt_size_min","pkt_size_max"],
    "Protocol Counts":    ["count_tcp","count_udp","count_icmp","count_arp",
                           "ratio_tcp","ratio_udp","ratio_icmp","ratio_arp"],
    "TCP Flags":          ["flag_syn","flag_ack","flag_fin","flag_rst","flag_psh",
                           "syn_only_count","ratio_syn_only","ratio_syn_ack"],
    "TCP Window":         ["tcp_win_mean","tcp_win_std","tcp_win_min","tcp_win_max"],
    "Ports":              ["unique_src_ports","unique_dst_ports","port_entropy_dst",
                           "http_pkts","https_pkts","rtsp_pkts","ssh_pkts","telnet_pkts"],
    "IP Diversity":       ["unique_src_ips","unique_dst_ips","src_ip_entropy","dst_ip_entropy"],
    "Payload":            ["payload_mean","payload_std","n_has_payload",
                           "max_payload_repeat","n_http_get","n_http_post"],
    "Flows":              ["n_flows","pkts_per_flow_mean","pkts_per_flow_max",
                           "bytes_per_flow_mean","bytes_per_flow_max","top_flow_pkts"],
    "ARP":                ["arp_who_has","arp_is_at"],
    "Attack Indicators":  ["n_http_conn_ips","ratio_rtsp"],
}

def build_matrix_text(row, all_cols):
    lines = []
    for group, feats in FEATURE_GROUPS.items():
        present = [f for f in feats if f in all_cols and f in row.index]
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

def build_jsonl(df: pd.DataFrame, output_path: str):
    all_cols = df.columns.tolist()
    records  = []

    for _, row in df.iterrows():
        label = str(row.get("label", "unknown")).strip()
        if label == "unknown":
            continue

        matrix = build_matrix_text(row, all_cols)
        explanation = ATTACK_EXPLANATIONS.get(label, f"Traffic pattern: {label}")

        user_msg = (
            f"Analyze this network capture feature matrix from a SPARSH CCTV camera "
            f"(10-second window) and classify the attack type:\n\n{matrix}"
        )
        answer = (
            f"**Classification: {label.upper()}**\n\n"
            f"{explanation}"
        )

        records.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": answer},
            ]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Saved {len(records)} records → {output_path}")
    return len(records)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="features.csv",  help="Raw features CSV (no augmentation)")
    parser.add_argument("--output_dir", default="./json_splits",      help="Output directory")
    parser.add_argument("--target_rows",type=int, default=TARGET_ROWS, help="Target train rows after augmentation")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load raw features ─────────────────────
    print("=" * 60)
    print("  Loading features.csv ...")
    print("=" * 60)
    df = pd.read_csv(args.input)
    print(f"  Rows    : {len(df)}")
    print(f"  Columns : {len(df.columns)}")
    print(f"  Labels  : {sorted(df['label'].unique())}")

    # ── STEP 1: Split by run ──────────────────
    print("\n[STEP 1] Splitting by run folder ...")
    train_df, val_df, test_df = split_by_run(df)

    # ── STEP 2: Augment train only ────────────
    print("\n[STEP 2] Augmenting train set only ...")
    train_aug = augment(train_df, target_rows=args.target_rows)

    # ── STEP 3: Save CSVs ─────────────────────
    print("\n[STEP 3] Saving CSV splits ...")
    train_path = os.path.join(args.output_dir, "train_augmented.csv")
    val_path   = os.path.join(args.output_dir, "val_clean.csv")
    test_path  = os.path.join(args.output_dir, "test_clean.csv")

    train_aug.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"  train_augmented.csv → {len(train_aug)} rows")
    print(f"  val_clean.csv       → {len(val_df)} rows")
    print(f"  test_clean.csv      → {len(test_df)} rows")

    # ── STEP 4: Build JSONL ───────────────────
    print("\n[STEP 4] Building JSONL files ...")
    build_jsonl(train_aug, os.path.join(args.output_dir, "train.jsonl"))
    build_jsonl(val_df,    os.path.join(args.output_dir, "val.jsonl"))
    build_jsonl(test_df,   os.path.join(args.output_dir, "test.jsonl"))

    # ── Summary ───────────────────────────────
    print("\n" + "=" * 60)
    print("  Done! Output files:")
    print("=" * 60)
    for f in sorted(Path(args.output_dir).iterdir()):
        size_kb = f.stat().st_size / 1000
        print(f"  {f.name:<30} {size_kb:.1f} KB")

    print("""
Next steps:
  1. Use train_augmented.csv  → RF training  (kaggle_notebook.py Cell 5)
  2. Use test_clean.csv       → honest RF evaluation
  3. Use train.jsonl + val.jsonl → LLM fine-tuning (Cell 10 onwards)
  4. Use test.jsonl           → honest LLM evaluation
""")


if __name__ == "__main__":
    main()