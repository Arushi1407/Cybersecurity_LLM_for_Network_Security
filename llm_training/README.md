# SPARSH CCTV Attack Detection Pipeline

End-to-end pipeline for detecting network attacks on SPARSH CCTV cameras — from raw packet captures all the way to a fine-tuned LLM that classifies and explains attack traffic.

The pipeline has three stages, meant to be run in order:

```
extract_features.py  →  split_then_augment.py  →  train_script.py
     (PCAPs)               (features.csv)           (CSV + JSONL)
```

---

## Supported Attack Classes

| Label | Description |
|---|---|
| `brute_force` | Repeated TCP login attempts to the same port |
| `slowloris` | Slow HTTP DoS holding partial connections open |
| `dos_http_flood` | Massive GET/POST flood to port 80/8080 |
| `dos_syn_flood` | SYN-only packets, no ACK completion (half-open) |
| `dos_udp_flood` | High-volume UDP to random destination ports |
| `mixed_attack` | Simultaneous SYN, UDP, and HTTP anomalies |
| `port_scan` | High unique destination ports, high RST count |
| `replay_attack` | Repeated payload fingerprints / duplicate sequences |
| `rtsp_brute_force` | Brute force to RTSP ports 554/8554 |
| `normal` | Benign traffic — balanced protocols, expected ports |

---

## Stage 1 — `extract_features.py`

### What it does
Walks through a folder of PCAP capture files organised by run folder and extracts ~80 statistical network features from each file into a single `features.csv`. It auto-detects the attack label from the filename, and crucially **saves the CSV after every single file** so no data is lost if it crashes mid-run.

Features extracted cover:
- **Volume & timing** — total packets, duration, packets/sec, bytes/sec
- **Inter-arrival time** — mean, std, min, max, median of gap between packets
- **Packet size** — mean, std, min, max, median
- **Protocol counts & ratios** — TCP, UDP, ICMP, ARP, DNS
- **TCP flags** — SYN, ACK, FIN, RST, PSH, URG counts; SYN-only ratio; SYN/ACK ratio
- **TCP window** — mean, std, min, max window size
- **Ports** — unique source/destination ports, destination port entropy, HTTP/HTTPS/RTSP/SSH/Telnet packet counts
- **IP diversity** — unique source/destination IPs, IP entropy
- **TTL** — mean, std, min, max
- **Payload** — mean size, repeat count (replay indicator), HTTP GET/POST counts
- **Flows** — number of flows, packets/flow, bytes/flow, top flow packet count
- **ARP** — who-has and is-at counts
- **Attack-specific** — `n_http_conn_ips` (slowloris indicator), `ratio_rtsp`

### Expected dataset structure
```
laptop_dataset/
    run_1/
        brute_force_ssl.pcap
        slowloris.pcap
        dos_http_flood.pcap
        dos_syn_flood.pcap
        dos_udp_flood.pcap
        mixed_attack.pcap
        port_scan.pcap
        replay_attack.pcap
        rtsp_brute_force.pcap
        normal.pcap
    run_2/
        ...
    run_31/
        ...
```

Run folders must be named `run_N` (e.g. `run_1`, `run_2`). PCAP filenames must contain keywords from the label map — e.g. a file containing `dos_syn` in its name will be labelled `dos_syn_flood`. Files with unrecognised names are skipped with a warning.

### Installation
```bash
pip install scapy numpy pandas
```

### Usage
```bash
python extract_features.py --dataset ./laptop_dataset --output features.csv
```

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `./laptop_attack` | Path to the root folder containing run subfolders |
| `--output` | `./features.csv` | Output CSV path |

### Output
A single `features.csv` where each row is one PCAP file with columns: `label`, `run`, `filename`, then all ~80 numeric features.

---

## Stage 2 — `split_then_augment.py`

### What it does
Takes the raw `features.csv` from Stage 1 and produces clean train/val/test splits plus JSONL files ready for LLM fine-tuning. The key design decision: **splits by run folder first, then augments only the training set** — this prevents sliding-window data leakage where packets from the same capture could end up in both train and test.

The four steps it runs internally:

1. **Split by run** — assigns entire run folders (not individual rows) to train / val / test at a 70/15/15 ratio by run count, ensuring no leakage between splits
2. **Augment train only** — expands the training set to a target row count using Gaussian noise on numeric features (2% of each feature's std dev), keeping labels intact; val and test are never touched
3. **Save CSVs** — `train_augmented.csv`, `val_clean.csv`, `test_clean.csv`
4. **Build JSONL** — converts each split into chat-format JSONL for LLM fine-tuning, where each record is a system prompt + feature matrix as user message + classification + explanation as assistant reply

### Installation
```bash
pip install numpy pandas scikit-learn
```

### Usage
```bash
python split_then_augment.py --input features.csv --output_dir ./splits
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `features.csv` | Raw features CSV from Stage 1 |
| `--output_dir` | `./json_splits` | Directory where all output files are written |
| `--target_rows` | `9500` | Target number of training rows after augmentation |

### Outputs
```
splits/
    train_augmented.csv   ← augmented train set  (RF + LLM training)
    val_clean.csv         ← untouched val set     (LLM validation)
    test_clean.csv        ← untouched test set    (honest evaluation)
    train.jsonl           ← chat-format records for LLM fine-tuning
    val.jsonl
    test.jsonl
```

---

## Stage 3 — `train_script.py`

### What it does
Trains two models on the splits produced by Stage 2:

**Random Forest baseline**
- 500-tree Random Forest with `class_weight="balanced"` and `max_features="sqrt"`
- 5-fold stratified cross-validation reported on the training set
- Full classification report and confusion matrix on the unseen test set
- Saves model, label encoder, and feature names to disk for reuse

**Llama-3.2-3B-Instruct fine-tuned with LoRA**
- Loads `meta-llama/Llama-3.2-3B-Instruct` in 4-bit (NF4) quantisation via `bitsandbytes`
- Applies LoRA adapters (rank 16) to all attention and MLP projection layers (~0.6% of parameters trainable)
- Fine-tunes with `SFTTrainer` on the JSONL files from Stage 2
- After training, runs a quick inference test to verify the model is working

The LLM is trained to do more than classify — its output is a natural language explanation of *why* the traffic matches a given attack type, referencing specific feature values.

### Requirements

**Python packages**
```bash
pip install transformers==4.46.3 trl==0.11.4 peft==0.13.2 accelerate==1.0.1 \
            bitsandbytes datasets scikit-learn joblib pandas numpy matplotlib seaborn torch
```

**Hardware** — a CUDA GPU with at least 16GB VRAM. The 4-bit quantised 3B model fits comfortably on a single 16GB GPU (T4, A10, RTX 3090/4090, etc).

**Hugging Face access** — `meta-llama/Llama-3.2-3B-Instruct` is a gated model. Accept the license at [huggingface.co/meta-llama](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) before running. Set your token as an environment variable:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### Configuration
Edit the two path variables at the top of `train_script.py`:
```python
INPUT_DIR = "/path/to/your/splits"   # folder containing the CSVs and JSONLs from Stage 2
WORK_DIR  = "/path/to/your/output"   # where checkpoints, plots, and the final model are saved
```

### Usage
```bash
python train_script.py
```

### Outputs
```
output/
    rf/
        random_forest.pkl       ← trained RF model
        label_encoder.pkl       ← sklearn LabelEncoder
        feature_names.json      ← ordered feature list
        confusion_matrix.png    ← test-set confusion matrix
    model_out/
        final/                  ← LoRA adapter weights + tokenizer
        checkpoint-*/           ← intermediate checkpoints
```

---

## Running the full pipeline

```bash
# Stage 1 — extract features from your PCAPs
python extract_features.py --dataset ./laptop_dataset --output features.csv

# Stage 2 — split, augment, and build JSONL
python split_then_augment.py --input features.csv --output_dir ./splits

# Stage 3 — train RF + fine-tune LLM (edit INPUT_DIR/WORK_DIR in the script first)
python train_script.py
```

---

