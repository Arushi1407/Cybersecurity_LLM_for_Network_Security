"""
extract_features.py — SPARSH CCTV Attack Dataset Feature Extraction
=====================================================================
Folder structure expected:
    laptop_dataset/
        run1/
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
        run2/  ...
        run31/ ...

Usage:
    python extract_features.py --dataset ./laptop_dataset --output features.csv

Fixes in this version:
    - Strict ONE level deep traversal (no recursion into subfolders)
    - Saves CSV after EVERY file (no data loss if it crashes)
    - Clear progress printed for every single file
    - Skips non-.pcap/.pcapng files completely
    - Auto-detects label from filename
"""

import os
import re
import sys
import math
import argparse
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Scapy import with clear error message ────────────────────
try:
    from scapy.all import rdpcap, IP, TCP, UDP, ICMP, ARP, Raw  # type: ignore
except ImportError:
    print("[ERROR] Scapy not installed. Run: pip install scapy")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# CONFIG — edit DATASET_ROOT to match your path
# ─────────────────────────────────────────────────────────────

DATASET_ROOT = "./laptop_attack"
OUTPUT_CSV   = "./features.csv"

# Maps substring in filename → label
# e.g. "dos_syn_flood.pcap" contains "dos_syn" → "dos_syn_flood"
LABEL_MAP = {
    "brute_force_ssl":    "brute_force_ssl",
    "slowloris":      "slowloris",
    "dos_http":       "dos_http_flood",
    "dos_syn":        "dos_syn_flood",
    "dos_udp":        "dos_udp_flood",
    "mixed":          "mixed_attack",
    "port_scan":      "port_scan",
    "replay":         "replay_attack",
    "rtsp_brute_force":"rtsp_brute_force",
    "normal":         "normal",
}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def safe_div(a, b, default=0.0):
    return a / b if b else default

def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)

def label_from_filename(stem: str) -> str:
    """Match filename stem to label using LABEL_MAP keywords."""
    s = stem.lower()
    for keyword, label in LABEL_MAP.items():
        if keyword in s:
            return label
    return "unknown"

def safe_stats(arr):
    """Return mean, std, min, max, median of a list. Returns zeros if empty."""
    if not arr:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    a = np.array(arr, dtype=float)
    return (float(a.mean()), float(a.std()),
            float(a.min()),  float(a.max()), float(np.median(a)))

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_features(pcap_path: str) -> dict:
    """Extract ~80 network features from a single PCAP file."""

    # ── Read packets ──────────────────────────
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"      [ERROR] Cannot read file: {e}")
        return None

    n_packets = len(packets)
    if n_packets == 0:
        print(f"      [WARN] File is empty — skipping")
        return None

    # ── Timestamps & IAT ─────────────────────
    timestamps = [float(p.time) for p in packets]
    duration   = max(timestamps[-1] - timestamps[0], 1e-6)
    iats       = [timestamps[i+1] - timestamps[i]
                  for i in range(n_packets - 1)]

    # ── Packet sizes ──────────────────────────
    pkt_sizes = [len(p) for p in packets]

    # ── Protocol counts ───────────────────────
    n_ip   = sum(1 for p in packets if IP   in p)
    n_tcp  = sum(1 for p in packets if TCP  in p)
    n_udp  = sum(1 for p in packets if UDP  in p)
    n_icmp = sum(1 for p in packets if ICMP in p)
    n_arp  = sum(1 for p in packets if ARP  in p)

    # ── TCP flags ─────────────────────────────
    flag_counts      = Counter()
    tcp_window_sizes = []
    for p in packets:
        if TCP in p:
            f = p[TCP].flags
            flag_counts["SYN"] += int(bool(f & 0x02))
            flag_counts["ACK"] += int(bool(f & 0x10))
            flag_counts["FIN"] += int(bool(f & 0x01))
            flag_counts["RST"] += int(bool(f & 0x04))
            flag_counts["PSH"] += int(bool(f & 0x08))
            flag_counts["URG"] += int(bool(f & 0x20))
            tcp_window_sizes.append(p[TCP].window)

    # SYN with no ACK → half-open (SYN flood indicator)
    syn_only = sum(
        1 for p in packets
        if TCP in p and (p[TCP].flags & 0x12) == 0x02
    )

    # ── Ports ─────────────────────────────────
    src_ports, dst_ports = [], []
    for p in packets:
        if TCP in p:
            src_ports.append(p[TCP].sport)
            dst_ports.append(p[TCP].dport)
        elif UDP in p:
            src_ports.append(p[UDP].sport)
            dst_ports.append(p[UDP].dport)

    dst_port_ctr = Counter(dst_ports)

    http_pkts    = sum(1 for d in dst_ports if d in (80, 8080, 8000))
    https_pkts   = sum(1 for d in dst_ports if d in (443, 8443))
    rtsp_pkts    = sum(1 for d in dst_ports if d in (554, 8554))
    telnet_pkts  = sum(1 for d in dst_ports if d == 23)
    ssh_pkts     = sum(1 for d in dst_ports if d == 22)

    # ── IPs & TTL ─────────────────────────────
    src_ips, dst_ips, ttl_vals = [], [], []
    for p in packets:
        if IP in p:
            src_ips.append(p[IP].src)
            dst_ips.append(p[IP].dst)
            ttl_vals.append(p[IP].ttl)

    # ── Payload ───────────────────────────────
    payload_sizes     = []
    repeated_payloads = Counter()
    n_has_payload     = 0
    n_http_get        = 0
    n_http_post       = 0

    for p in packets:
        if Raw in p:
            raw = bytes(p[Raw])
            payload_sizes.append(len(raw))
            n_has_payload += 1
            repeated_payloads[raw[:64]] += 1
            try:
                head = raw[:8].decode("utf-8", errors="ignore")
                if head.startswith("GET "):  n_http_get  += 1
                if head.startswith("POST "): n_http_post += 1
            except Exception:
                pass

    max_payload_repeat = (max(repeated_payloads.values())
                          if repeated_payloads else 0)

    # ── Flows ─────────────────────────────────
    flows = defaultdict(list)
    for p in packets:
        if IP in p:
            proto = (6  if TCP in p else
                     17 if UDP in p else 0)
            dport = (p[TCP].dport if TCP in p else
                     p[UDP].dport if UDP in p else 0)
            flows[(p[IP].src, p[IP].dst, dport, proto)].append(len(p))

    n_flows        = len(flows)
    pkts_per_flow  = [len(v)   for v in flows.values()]
    bytes_per_flow = [sum(v)   for v in flows.values()]

    # ── DNS & ARP ─────────────────────────────
    n_dns = sum(
        1 for p in packets
        if UDP in p and (p[UDP].dport == 53 or p[UDP].sport == 53)
    )
    arp_who_has = sum(1 for p in packets if ARP in p and p[ARP].op == 1)
    arp_is_at   = sum(1 for p in packets if ARP in p and p[ARP].op == 2)

    # Unique IPs on HTTP port (Slowloris indicator)
    http_conn_ips = set(
        p[IP].src for p in packets
        if TCP in p and p[TCP].dport in (80, 8080) and IP in p
    )

    # ── Aggregate stats ───────────────────────
    pm,  ps,  pmin,  pmax,  pmed  = safe_stats(pkt_sizes)
    im,  is_, imin,  imax,  imed  = safe_stats(iats)
    aym, ays, aymin, aymax, aymed = safe_stats(payload_sizes)
    wm,  ws,  wmin,  wmax,  wmed  = safe_stats(tcp_window_sizes)
    tm,  ts,  tmin,  tmax,  tmed  = safe_stats(ttl_vals)
    pfm, pfs, pfmin, pfmax, pfmed = safe_stats(pkts_per_flow)
    bfm, bfs, bfmin, bfmax, bfmed = safe_stats(bytes_per_flow)

    # ── Build flat feature dict ───────────────
    feat = {
        # Volume
        "total_packets":       n_packets,
        "duration_sec":        round(duration, 6),
        "packets_per_sec":     round(safe_div(n_packets, duration), 4),
        "bytes_total":         sum(pkt_sizes),
        "bytes_per_sec":       round(safe_div(sum(pkt_sizes), duration), 4),

        # Packet size
        "pkt_size_mean":       round(pm,   4),
        "pkt_size_std":        round(ps,   4),
        "pkt_size_min":        pmin,
        "pkt_size_max":        pmax,
        "pkt_size_median":     pmed,

        # Inter-arrival time
        "iat_mean":            round(im,   6),
        "iat_std":             round(is_,  6),
        "iat_min":             round(imin, 6),
        "iat_max":             round(imax, 6),
        "iat_median":          round(imed, 6),

        # Protocol ratios
        "ratio_ip":            round(safe_div(n_ip,   n_packets), 4),
        "ratio_tcp":           round(safe_div(n_tcp,  n_packets), 4),
        "ratio_udp":           round(safe_div(n_udp,  n_packets), 4),
        "ratio_icmp":          round(safe_div(n_icmp, n_packets), 4),
        "ratio_arp":           round(safe_div(n_arp,  n_packets), 4),
        "count_ip":            n_ip,
        "count_tcp":           n_tcp,
        "count_udp":           n_udp,
        "count_icmp":          n_icmp,
        "count_arp":           n_arp,
        "count_dns":           n_dns,

        # TCP flags
        "flag_syn":            flag_counts["SYN"],
        "flag_ack":            flag_counts["ACK"],
        "flag_fin":            flag_counts["FIN"],
        "flag_rst":            flag_counts["RST"],
        "flag_psh":            flag_counts["PSH"],
        "flag_urg":            flag_counts["URG"],
        "syn_only_count":      syn_only,
        "ratio_syn_only":      round(safe_div(syn_only, n_tcp), 4),
        "ratio_syn_ack":       round(safe_div(
                                   flag_counts["SYN"],
                                   max(flag_counts["ACK"], 1)), 4),

        # TCP window
        "tcp_win_mean":        round(wm,  4),
        "tcp_win_std":         round(ws,  4),
        "tcp_win_min":         wmin,
        "tcp_win_max":         wmax,

        # Ports
        "unique_src_ports":    len(set(src_ports)),
        "unique_dst_ports":    len(set(dst_ports)),
        "port_entropy_dst":    round(entropy(list(dst_port_ctr.values())), 4),
        "top_dst_port_count":  max(dst_port_ctr.values()) if dst_port_ctr else 0,
        "http_pkts":           http_pkts,
        "https_pkts":          https_pkts,
        "rtsp_pkts":           rtsp_pkts,
        "telnet_pkts":         telnet_pkts,
        "ssh_pkts":            ssh_pkts,

        # IP diversity
        "unique_src_ips":      len(set(src_ips)),
        "unique_dst_ips":      len(set(dst_ips)),
        "ip_pair_count":       len(Counter(zip(src_ips, dst_ips))),
        "src_ip_entropy":      round(entropy(list(Counter(src_ips).values())), 4),
        "dst_ip_entropy":      round(entropy(list(Counter(dst_ips).values())), 4),

        # TTL
        "ttl_mean":            round(tm,  4),
        "ttl_std":             round(ts,  4),
        "ttl_min":             tmin,
        "ttl_max":             tmax,

        # Payload
        "payload_mean":        round(aym, 4),
        "payload_std":         round(ays, 4),
        "payload_max":         aymax,
        "n_has_payload":       n_has_payload,
        "ratio_has_payload":   round(safe_div(n_has_payload, n_packets), 4),
        "max_payload_repeat":  max_payload_repeat,
        "n_http_get":          n_http_get,
        "n_http_post":         n_http_post,

        # Flows
        "n_flows":             n_flows,
        "pkts_per_flow_mean":  round(pfm, 4),
        "pkts_per_flow_std":   round(pfs, 4),
        "pkts_per_flow_max":   pfmax,
        "bytes_per_flow_mean": round(bfm, 4),
        "bytes_per_flow_std":  round(bfs, 4),
        "bytes_per_flow_max":  bfmax,
        "top_flow_pkts":       max(pkts_per_flow) if pkts_per_flow else 0,

        # ARP
        "arp_who_has":         arp_who_has,
        "arp_is_at":           arp_is_at,

        # Attack-specific
        "n_http_conn_ips":     len(http_conn_ips),
        "ratio_rtsp":          round(safe_div(rtsp_pkts, n_packets), 4),
    }

    return feat


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract features from SPARSH CCTV attack PCAPs"
    )
    parser.add_argument("--dataset", default=DATASET_ROOT,
                        help="Path to laptop_dataset folder")
    parser.add_argument("--output",  default=OUTPUT_CSV,
                        help="Output CSV file path")
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        print(f"[ERROR] Dataset folder not found: {root.resolve()}")
        print("        Check --dataset path")
        sys.exit(1)

    # ── Find run folders (ONE level deep only) ────────────────
    run_dirs = sorted(
        [d for d in root.iterdir()
         if d.is_dir() and re.match(r"run_\d+", d.name.lower())],
        key=lambda d: int(re.findall(r"\d+", d.name)[0])
    )

    if not run_dirs:
        print(f"[ERROR] No run folders found inside: {root.resolve()}")
        print("        Expected folders named run1, run2, ... run31")
        sys.exit(1)

    print("=" * 60)
    print("  SPARSH CCTV — Feature Extraction")
    print("=" * 60)
    print(f"  Dataset : {root.resolve()}")
    print(f"  Output  : {Path(args.output).resolve()}")
    print(f"  Runs    : {len(run_dirs)} folders found")
    print("=" * 60)

    all_rows   = []
    file_count = 0
    skip_count = 0
    first_save = True

    for run_dir in run_dirs:

        # Collect ONLY .pcap / .pcapng directly inside run folder
        # (no glob("**/*") — strictly one level)
        pcap_files = sorted(
            [f for f in run_dir.iterdir()
             if f.is_file() and f.suffix.lower() in (".pcap", ".pcapng")]
        )

        print(f"\n  [{run_dir.name}]  {len(pcap_files)} files")

        for pcap_file in pcap_files:
            file_count += 1
            label = label_from_filename(pcap_file.stem)

            print(f"    [{file_count:>3}] {pcap_file.name:<35} label={label}", end="", flush=True)

            if label == "unknown":
                print("  → SKIPPED (filename not recognised)")
                skip_count += 1
                continue

            feat = extract_features(str(pcap_file))

            if feat is None:
                print("  → SKIPPED (read error or empty)")
                skip_count += 1
                continue

            feat["label"]    = label
            feat["run"]      = run_dir.name
            feat["filename"] = pcap_file.name
            all_rows.append(feat)

            # ── Save CSV after every single file ──────────────
            df = pd.DataFrame(all_rows)
            meta  = ["label", "run", "filename"]
            feats = [c for c in df.columns if c not in meta]
            df    = df[meta + feats]

            if first_save:
                df.to_csv(args.output, index=False)
                first_save = False
            else:
                df.to_csv(args.output, index=False)

            print(f"  → OK  (saved {len(all_rows)} rows so far)")

    # ── Final summary ─────────────────────────
    print("\n" + "=" * 60)
    if all_rows:
        df_final = pd.DataFrame(all_rows)
        print(f"  Done!")
        print(f"  Total files processed : {file_count}")
        print(f"  Skipped               : {skip_count}")
        print(f"  Rows saved            : {len(df_final)}")
        print(f"  Features per row      : {len(df_final.columns) - 3}")
        print(f"\n  Label distribution:")
        for lbl, cnt in df_final["label"].value_counts().items():
            print(f"    {lbl:<25} {cnt} samples")
        print(f"\n  Output saved → {Path(args.output).resolve()}")
    else:
        print("  [ERROR] No features were extracted.")
        print("  Check that your filenames contain keywords like:")
        print("  brute_force, slowloris, dos_http, dos_syn, dos_udp,")
        print("  mixed, port_scan, replay, rtsp, normal")
    print("=" * 60)


if __name__ == "__main__":
    main()