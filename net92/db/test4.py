import json
import re

def find_record(contract_code_str, jsonl_file_path):
    """
    Given a contract code string and a .jsonl file path,
    return all JSON records whose 'code' is found in the contract code.
    """
    
    contract_code_clean = normalize_code(contract_code_str)
    print(contract_code_clean)
    
    matches = []
    
    # Load JSONL records
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            json_code_clean = normalize_code(record['code'])
#            print(json_code_clean)

            if json_code_clean in contract_code_clean:
                matches.append(record)
    
    return matches

def normalize_code(code):
    # Remove all whitespace including newlines, tabs, spaces
    return re.sub(r'\s+', '', code)

# Load your contract string (e.g., from a file or string literal)
with open('MyContract.sol', 'r') as f:
    contract_code = f.read()

# Search for matches
matched_records = find_record(contract_code, 'db1.json')

# Print results
for match in matched_records:
    print(f"Matched Hash: {match['hash']}, Type: {match['type']}")
    print('-' * 50)
