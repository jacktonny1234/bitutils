import json
from py_solidity_vuln_db import get_vulnerability

VULNERABILITIES = [
    "Reentrancy",
    "Gas griefing",
    "Bad randomness",
    "Forced reception",
    "Unguarded function",
    "Signature replay",
]

def build_dataset(output_path="db.json"):
    dataset = []
    seen_hashes = set()

    Count = 1000

    while True:
        vuln = get_vulnerability()
        if not vuln or not vuln.code:
            continue

        code = vuln.code.strip()
        code_hash = hash(code)
        if code_hash in seen_hashes:
            continue  # Skip duplicates
        seen_hashes.add(code_hash)
        Count = Count - 1
        if Count == 0:
            break

        messages = [
            {
                "role": "system",
                "content": "You are a Solidity smart contract auditor. Given code with line numbers, return an audit report in JSON format."
            },
            {
                "role": "user",
                "content": code
            },
            {
                "role": "assistant",
                "content": json.dumps([{
                    "fromLine": 1,
                    "toLine": len(code.splitlines()),
                    "vulnerabilityClass": vuln.name,
                    "description": f"This code contains {vuln.name} vulnerability.",
                }])
            }
        ]
        dataset.append({"messages": messages})

    with open(output_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset saved to {output_path} with {len(dataset)} unique samples.")

build_dataset()
