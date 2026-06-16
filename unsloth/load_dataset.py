#!/usr/bin/env python3
"""
ServiceNow ITSM Synthetic Dataset Loader
=======================================
Loads chunked JSON files and prepares them for Unsloth fine-tuning.

Usage:
    python load_dataset.py --format conversations  # ChatML/ShareGPT
    python load_dataset.py --format alpaca          # instruction/input/output
"""
import json
import argparse
import glob
from pathlib import Path

def load_chunks(chunk_dir=".", pattern="servicenow_itsm_synthetic_chunk_*.json"):
    """Load all chunk files and merge into single dataset."""
    chunks = sorted(glob.glob(f"{chunk_dir}/{pattern}"))
    dataset = []
    for chunk_path in chunks:
        with open(chunk_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            dataset.extend(data)
        print(f"Loaded {len(data):,} records from {Path(chunk_path).name}")
    print(f"Total loaded: {len(dataset):,} records")
    return dataset

def convert_to_unsloth_format(dataset, format_type="conversations"):
    """Convert to Unsloth-compatible format."""
    if format_type == "conversations":
        # ShareGPT/ChatML format
        return [{"conversations": d["conversations"]} for d in dataset]
    elif format_type == "alpaca":
        # Alpaca instruction format
        return [
            {
                "instruction": d["instruction"],
                "input": d["input"],
                "output": d["output"]
            }
            for d in dataset
        ]
    elif format_type == "raw":
        return dataset
    else:
        raise ValueError(f"Unknown format: {format_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", default="/mnt/agents/output")
    parser.add_argument("--format", default="conversations", choices=["conversations", "alpaca", "raw"])
    parser.add_argument("--output", default="/mnt/agents/output/unsloth_train_ready.json")
    parser.add_argument("--sample", type=int, default=0, help="If >0, only use N random samples for quick testing")
    args = parser.parse_args()

    dataset = load_chunks(args.chunk_dir)

    if args.sample > 0:
        import random
        random.seed(42)
        dataset = random.sample(dataset, min(args.sample, len(dataset)))
        print(f"Sampled down to {len(dataset):,} records")

    converted = convert_to_unsloth_format(dataset, args.format)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False)

    print(f"Saved Unsloth-ready dataset to {args.output}")
    print(f"Format: {args.format}")
    print(f"Records: {len(converted):,}")

if __name__ == "__main__":
    main()
