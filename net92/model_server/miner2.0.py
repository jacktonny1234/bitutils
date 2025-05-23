import json
import os

from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import hashlib
from collections import OrderedDict

app = FastAPI()

# Load Mistral 7B Instruct model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
#model_name = "./mixtral-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)

code = ""
# tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"
ROLE_USER = "user"

def extract_json_string(text):
    """Extract the first valid JSON array from text using regex."""
    match = re.search(r"\[\s*{.*?}\s*]", text, re.DOTALL) 
    if match:
        return match.group(0)
    else:
        return "[]"

def get_prompt(solidity_code):

    # with open("model_servers/example", "r", encoding="utf-8") as f:
    #     examples = f.read()

    #  Examples:
    #  {examples}

    numbered_code = "\n".join(
    f"{i+1}: {line}" for i, line in enumerate(solidity_code.splitlines())
    )

    PROMPT = f"""
    You are analyzing a Solidity smart contract to find vulnerabilities.
    Now analyze the following Solidity code and identify if it contains a vulnerability.
    return a JSON array where each object describes a single security issue.

    Use this format:
    [
    {{
        "fromLine": INT,
        "toLine": INT,
        "vulnerabilityClass": STRING,
        "description": STRING
    }}
    ]

    Rules:
    - fromLine is the line number where the vulnerable code starts.
    - toLine is the line number where the vulnerable code ends.
    - Do not include extra text or explanations.
    - Use only these vulnerabilityClass values: "Known compiler bugs", "Reentrancy", "Gas griefing", "Oracle manipulation", "Bad randomness", "Unexpected privilege grants", "Forced reception", "Integer overflow/underflow", "Race condition", "Unguarded function", "Inefficient storage key", "Front-running potential", "Miner manipulation", "Storage collision", "Signature replay", "Unsafe operation".
    - Describe how the vulnerability can be exploited.
    - "// SPDX-License-Identifier: MIT" is valid solidity code
    - Return Must include fromLine, toLine, vulnerabilityClass, description

    Return only the JSON output.
    Code:
    {numbered_code}
    """
    return PROMPT

def get_prompt2(solidity_code, record):

    numbered_code = "\n".join(
    f"{i+1}: {line}" for i, line in enumerate(solidity_code.splitlines())
    )

    PROMPT = f"""
    You are analyzing a Solidity smart contract to find vulnerabilities.

    Given code return a JSON array where each object describes a single security issue.

    Use this format:
    [
    {{
        "fromLine": INT,
        "toLine": INT,
        "vulnerabilityClass": STRING,
        "description": STRING
    }}
    ]

    Rules:
    - fromLine is the line number where the vulnerable code starts.
    - toLine is the line number where the vulnerable code ends.
    - vulnerabilityClass is "{record["type"]}"
    - example of "{record["type"]}" type of vulnerability is "{record["code"]}"
    - Do not include extra text or explanations.
    - Describe how the vulnerability can be exploited.
    - "// SPDX-License-Identifier: MIT" is valid solidity code
    - Return Must include fromLine, toLine, vulnerabilityClass, description
    
    Return only the JSON output.

    Code:
    {numbered_code}
    """
    return PROMPT

def get_prompt3(solidity_code):

    numbered_code = "\n".join(
    f"{i+1}: {line}" for i, line in enumerate(solidity_code.splitlines())
    )

    PROMPT = f"""
    Return this:
    
    [
    {{
        "fromLine": 1,
        "toLine": TOTAL_LINE_COUNT,
        "vulnerabilityClass": "Invalid Code",
        "description": "The entire code is considered invalid for audit processing."
    }}
    ]
    
    Rules:
    - toLine is the line number of rows of full code
    - Do not include extra text or explanations.
    - JSON Array size is one
    
    Return only the JSON output.

    Code:
    {numbered_code}
    """
    return PROMPT

def find_record(contract_code_str, jsonl_file_path):
    """
    Given a contract code string and a .jsonl file path,
    return all JSON records whose 'code' is found in the contract code.
    """
    
    contract_code_clean = normalize_code(contract_code_str)
    
    matches = []
    
    # Load JSONL records
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
            except Exception as e:
                print(f"Error {e}")
                continue
            json_code_clean = normalize_code(record['code'])

            if json_code_clean in contract_code_clean:
                matches.append(record)
    
    return matches

def normalize_code(code):
    # Remove all whitespace including newlines, tabs, spaces
    return re.sub(r'\s+', '', code)


def generate_audit(source: str):
    # Search for matches
    temperature = 0.1
    matched_records = find_record(source, 'db/db1.json')
    if source[:8] == "contract":
        prompt = get_prompt3(source)
    elif matched_records:
        temperature = 0.2
        # Print results
        for match in matched_records:
            print(f"Matched Hash: {match['hash']}, Type: {match['type']}")
            print('-' * 50)
        prompt = get_prompt2(source, matched_records[0])
    else:
        temperature = 0.4
        prompt = get_prompt(source)
        # return "[]"

    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = llm.generate(
            **inputs, 
            max_new_tokens=4096,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    # Calculate the length of the input prompt
    prompt_length = inputs["input_ids"].shape[1]

    # Extract only the generated tokens
    generated_tokens = output[0][prompt_length:]

    message = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    json_string = extract_json_string(message)

    return json_string


REQUIRED_KEYS = {
    "fromLine",
    "toLine",
    "vulnerabilityClass",
    "description",
}
INT_KEYS = ("fromLine", "toLine")


def try_prepare_result(result) -> list[dict]:

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            return []

    if isinstance(result, dict):
        if (
            len(result) == 1
            and isinstance(list(result.values())[0], list)
            and all(isinstance(item, dict) for item in list(result.values())[0])
        ):
            result = list(result.values())[0]
        else:
            result = [result]

    prepared = []
    for item in result:
        for key in REQUIRED_KEYS:
            if key not in item:
                print("not all reqiured fields exist")
                return None
        cleared = {k: item[k] for k in REQUIRED_KEYS}
        # if (
        #     "priorArt" in item
        #     and isinstance(item["priorArt"], list)
        #     and all(isinstance(x, str) for x in item["priorArt"])
        # ):
            # cleared["priorArt"] = item["priorArt"]
        # if "fixedLines" in item and isinstance(item["fixedLines"], str):
        #     cleared["fixedLines"] = item["fixedLines"]
        # if "testCase" in item and isinstance(item["testCase"], str):
        #     cleared["testCase"] = item["testCase"]
        for k in INT_KEYS:
            if isinstance(cleared[k], int) or (
                isinstance(item[k], str) and item[k].isdigit()
            ):
                cleared[k] = int(cleared[k])
            else:
                print("2")
                return None
        prepared.append(cleared)
    return prepared

# Cache: stores up to 10 results
cache = OrderedDict()

def get_hash_index(code: str) -> str:
    """Generate a hash index for the contract code."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()

@app.post("/submit")
async def submit(request: Request):
    global code
    
    contract_code = (await request.body()).decode("utf-8")
    index = get_hash_index(contract_code)
    code = contract_code
    
    # Check cache
    if index in cache:
        print("======cache hit=======")
        return cache[index]

    print("======cache miss=======")
    tries = int(os.getenv("MAX_TRIES", "3"))
    is_valid, result = False, None

    while tries > 0:
        result = generate_audit(contract_code)
        result = try_prepare_result(result)
        if result is not None:
            is_valid = True
            with open("req_history", "a") as f:
                f.write("-"*300 + "\n")
                f.write("-"*300 + "\n")
                f.write(contract_code + "\n")
                f.write("-"*300 + "\n")
                json.dump(result, f, indent=2)
                f.write("-"*300 + "\n")
                f.write("-"*300 + "\n")
            break
        tries -= 1

    if not is_valid:
        raise HTTPException(status_code=400, detail="Unable to prepare audit")

    # Save to cache (ensure only last 10 are stored)
    if index not in cache:
        if len(cache) >= 20:
            cache.popitem(last=False)  # Remove oldest
        cache[index] = result

    return result

@app.get("/healthcheck")
async def healthchecker():
    return {"status": "OK"}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SERVER_PORT", "40004")))
