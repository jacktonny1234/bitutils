import json
import os

from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

app = FastAPI()

# Load Mistral 7B Instruct model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
#model_name = "./mixtral-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        raise ValueError("No valid JSON array found in model output.")

def get_prompt(solidity_code):
    PROMPT = f"""
    You are analyzing a Solidity smart contract to find vulnerabilities.
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

    when code don't start with "pragma solidity or // SPDX" is Invalid code
    If then return  JSON Array of size one:
    Use this format:
    [
    {{
        "fromLine": 1,
        "toLine": TOTAL_LINE_COUNT,
        "vulnerabilityClass": "Invalid Code",
        "description": "The entire code is considered invalid for audit processing."
    }}
    ]

    Rules:
    - Do not include extra text or explanations.
    - Use only these vulnerabilityClass values: "Known compiler bugs", "Reentrancy", "Gas griefing", "Oracle manipulation", "Bad randomness", "Unexpected privilege grants", "Forced reception", "Integer overflow/underflow", "Race condition", "Unguarded function", "Inefficient storage key", "Front-running potential", "Miner manipulation", "Storage collision", "Signature replay", "Unsafe operation".
    - Explain how the vulnerability can be exploited.
    - "// SPDX-License-Identifier: MIT" is valid solidity code
    - if 

    Return only the JSON output.

    Code:
    {solidity_code}
    """
    return PROMPT

def get_prompt2(solidity_code, record):
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

    - vulnerabilityClass is "{record["type"]}"
    - example of "{record["type"]}" type of vulnerability is "{record["code"]}"
    - Do not include extra text or explanations.
    - Explain how the vulnerability can be exploited.
    - "// SPDX-License-Identifier: MIT" is valid solidity code

    Return only the JSON output.

    Code:
    {solidity_code}
    """
    return PROMPT

def get_prompt3(solidity_code):
    PROMPT = f"""
    You are analyzing a Solidity smart contract to find vulnerabilities.

    If the entire code is invalid or unprocessable, return this:
    [
    {{
        "fromLine": 1,
        "toLine": TOTAL_LINE_COUNT,
        "vulnerabilityClass": "Invalid Code",
        "description": "The entire code is considered invalid for audit processing."
    }}
    ]
    
    If the entire code is not invalid or processable, return None:
        
    Rules:
    - Do not include extra text or explanations.

    Return only the JSON output or None.

    Code:
    {solidity_code}
    """
    return PROMPT

def find_record(contract_code_str, jsonl_file_path):
    """
    Given a contract code string and a .jsonl file path,
    return all JSON records whose 'code' is found in the contract code.
    """
    
    contract_code_clean = normalize_code(contract_code_str)
    print(f"contract_code_clean:{contract_code_clean}")
    
    matches = []
    
    # Load JSONL records
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
            except Exception as e:
                continue
            json_code_clean = normalize_code(record['code'])

            if json_code_clean in contract_code_clean:
                matches.append(record)
    
    return matches

def normalize_code(code):
    # Remove all whitespace including newlines, tabs, spaces
    return re.sub(r'\s+', '', code)


def generate_audit(source: str):

    #preprocessing invalid code
    if source[:7] == "contract":
        json_string = """
        [
            {
                "fromLine": 1,
                "toLine": TOTAL_LINE_COUNT,
                "vulnerabilityClass": "Invalid Code",
                "description": "The entire code is considered invalid for audit processing."
            }
        ]
        """
        return json_string

            

    # Search for matches
    matched_records = find_record(source, 'db/db1.json')

    if matched_records:
        # Print results
        for match in matched_records:
            print(f"Matched Hash: {match['hash']}, Type: {match['type']}")
            print('-' * 50)
        
        prompt = get_prompt2(source, matched_records[0])
    else:
        prompt = get_prompt(source)

    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = llm.generate(
            **inputs, 
            max_new_tokens=8096,
            temperature=0.4,
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


def try_prepare_result(result) -> list[dict] | None:
    print(f"result:{result}")
    with open("./output.txt", "w", encoding="utf-8") as file:
        file.write(result)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            return None

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
                print("3")
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


@app.post("/submit")
async def submit(request: Request):
    tries = int(os.getenv("MAX_TRIES", "3"))
    is_valid, result = False, None
    contract_code = (await request.body()).decode("utf-8")
    while tries > 0:
        result = generate_audit(contract_code)

        result = try_prepare_result(result)
        print("======after_prepare=======")
        print(result)   
        if result is not None:
            is_valid = True
            break
        tries -= 1
    if not is_valid:
        raise HTTPException(status_code=400, detail="Unable to prepare audit")
    return result

@app.get("/healthcheck")
async def healthchecker():
    return {"status": "OK"}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SERVER_PORT", "40004")))
