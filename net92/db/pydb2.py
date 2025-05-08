
import json
import os
import hashlib
from py_solidity_vuln_db import get_vulnerability

VULNERABILITIES = [
    "Reentrancy",
    "Gas griefing",
    "Bad randomness",
    "Forced reception",
    "Unguarded function",
    "Signature replay",
]

HASH_FILE = "hashes.txt"
OUTPUT_FILE = "db.json"
CYCLE_SIZE = 1000
DUPLICATE_THRESHOLD = 0.99  # 60%

def load_seen_hashes():
    seen = set()
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            for line in f:
                seen.add(line.strip())
    return seen

def save_hashes_batch(new_hashes):
    if new_hashes:
        with open(HASH_FILE, "a") as f:
            for h in new_hashes:
                f.write(h + "\n")

def append_dataset_batch(batch):
    if batch:
        with open(OUTPUT_FILE, "a") as f:
            for item in batch:
                f.write(json.dumps(item) + "\n")

def hash_code(code):
    return hashlib.sha256(code.encode()).hexdigest()

def build_dataset():
    seen_hashes = load_seen_hashes()
    print(f"Loaded {len(seen_hashes)} existing hashes.")

    while True:
        unique_count = 0
        duplicate_count = 0
        total_checked = 0
        message_list = []
        new_hashes = set()

        for i in range(1, CYCLE_SIZE + 1):
            vuln = get_vulnerability()
            if not vuln or not vuln.code:
                continue

            code = vuln.code.strip()
            code_hash = hash_code(code)

            total_checked += 1
            is_duplicate = code_hash in seen_hashes

            if is_duplicate:
                duplicate_count += 1
            else:
                seen_hashes.add(code_hash)
                new_hashes.add(code_hash)
                unique_count += 1

                message_list.append({
                    "type": vuln.name,
                    "code": code,
                    "hash": code_hash[:10]
                })

            if i % 100 == 0:
                current_total = unique_count + duplicate_count
                current_rate = (duplicate_count / current_total) if current_total > 0 else 0
                print(f"  Processed {i}/{CYCLE_SIZE} items... Duplicate rate: {current_rate:.2%}")

        total = unique_count + duplicate_count
        duplicate_rate = (duplicate_count / total) if total > 0 else 0

        append_dataset_batch(message_list)
        save_hashes_batch(new_hashes)

        print(f"Cycle finished: {unique_count} unique, {duplicate_count} duplicates, duplicate rate = {duplicate_rate:.2%}")

        if duplicate_rate > DUPLICATE_THRESHOLD:
            print("Duplicate rate too high. Stopping dataset generation.")
            break

build_dataset()
