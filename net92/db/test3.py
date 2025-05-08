from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
# Load Mistral 7B Instruct model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
#model_name = "./mixtral-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Solidity code with line numbers
solidity_code = """
pragma solidity ^0.8.17;

interface IERC20 {
    function totalSupply() external view returns (uint256);

    function balanceOf(address account) external view returns (uint256);

    function transfer(address recipient, uint256 amount) external returns (bool);

    function allowance(address owner, address spender) external view returns (uint256);

    function approve(address spender, uint256 amount) external returns (bool);

    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

}

contract DynamicFinanceProtocol {
    address public governance;
    address public teamMultisig;
    uint256 public lastUpdateTimestamp;
    uint256 public totalValueLocked;
    struct CollateralSettings {
        uint256 collateralFactor;
        uint256 liquidationThreshold;
        uint256 liquidationPenalty;
        bool isActive;
        uint256 supplyCap;
    }
    struct PoolSettings {
        uint256 interestRate;
        uint256 utilizationTarget;
        uint256 reserveFactor;
        uint256 liquidityMiningRate;
        bool isPaused;
    }
    struct LiquidationParams {
        uint256 maxDiscountRate;
        uint256 liquidationDelay;
        uint256 gracePeriod;
        uint256 minLiquidationAmount;
    }
    struct Protocol {
        mapping(uint32 => Pool) pools;
        LiquidationParams liquidationParams;
        uint256 removeLimitOrderFee;
        uint256 minimumOrderNotional;
        uint256 minRequiredMargin;
    }
    struct Pool {
        PoolSettings settings;
        uint256 totalDeposits;
        uint256 totalBorrows;
        mapping(address => uint256) userDeposits;
    }
    Protocol public protocol;
    mapping(IERC20 => CollateralSettings) public collateralConfigs;
    event CollateralSettingsUpdated(address indexed cToken, CollateralSettings settings);
    event PoolSettingsUpdated(uint32 indexed poolId, PoolSettings settings);
    event ProtocolSettingsUpdated(LiquidationParams liquidationParams, uint256 removeLimitOrderFee, uint256 minimumOrderNotional, uint256 minRequiredMargin);
    event PoolCreated(uint32 indexed poolId, PoolSettings settings);
    event UserPositionLiquidated(address indexed user, uint256 amount, uint256 penalty);
    modifier onlyGovernanceOrTeamMultisig() {
        require(msg.sender == governance || msg.sender == teamMultisig, 'Unauthorized: not governance or team multisig');
        _;
    }
    constructor(address _governance, address _teamMultisig) {
        governance = _governance;
        teamMultisig = _teamMultisig;
        lastUpdateTimestamp = block.timestamp;
        protocol.liquidationParams = LiquidationParams({maxDiscountRate: 10, liquidationDelay: 1 hours, gracePeriod: 6 hours, minLiquidationAmount: 100 ether});
        protocol.removeLimitOrderFee = 0.001 ether;
        protocol.minimumOrderNotional = 0.1 ether;
        protocol.minRequiredMargin = 0.05 ether;
    }

    function _updateCollateralSettings(IERC20 cToken, CollateralSettings memory collateralSettings) internal {
        collateralConfigs[cToken] = collateralSettings;
        emit CollateralSettingsUpdated(address(cToken), collateralSettings);
    }

    function modifyCollateralConfig(IERC20 cToken, CollateralSettings memory collateralSettings) external onlyGovernanceOrTeamMultisig {
        _updateCollateralSettings(cToken, collateralSettings);
    }

    function adjustPoolConfig(uint32 poolId, PoolSettings calldata newSettings) public onlyGovernanceOrTeamMultisig {
        protocol.pools[poolId].settings = newSettings;
        emit PoolSettingsUpdated(poolId, newSettings);
    }

    function modifyProtocolSettings(LiquidationParams calldata _liquidationParams, uint256 _removeLimitOrderFee, uint256 _minimumOrderNotional, uint256 _minRequiredMargin) external onlyGovernanceOrTeamMultisig {
        protocol.liquidationParams = _liquidationParams;
        protocol.removeLimitOrderFee = _removeLimitOrderFee;
        protocol.minimumOrderNotional = _minimumOrderNotional;
        protocol.minRequiredMargin = _minRequiredMargin;
        emit ProtocolSettingsUpdated(_liquidationParams, _removeLimitOrderFee, _minimumOrderNotional, _minRequiredMargin);
    }

    function createNewPool(uint32 poolId, PoolSettings calldata initialSettings) external onlyGovernanceOrTeamMultisig {
        require(protocol.pools[poolId].settings.interestRate == 0, 'Pool already exists');
        protocol.pools[poolId].settings = initialSettings;
        totalValueLocked += protocol.pools[poolId].totalDeposits;
        emit PoolCreated(poolId, initialSettings);
    }

    function executeEmergencyShutdown(uint32 poolId) external onlyGovernanceOrTeamMultisig {
        PoolSettings storage settings = protocol.pools[poolId].settings;
        settings.isPaused = true;
        settings.interestRate = 0;
        settings.liquidityMiningRate = 0;
        emit PoolSettingsUpdated(poolId, settings);
    }

    function calculateHealthFactor(address user, IERC20 collateralToken, uint256 borrowAmount) public view returns (uint256) {
        uint256 collateralValue = collateralToken.balanceOf(user) * getOraclePrice(address(collateralToken));
        uint256 borrowValue = borrowAmount;
        CollateralSettings memory settings = collateralConfigs[collateralToken];
        if (borrowValue == 0) {
            return type(uint256).max;
        }
        return (collateralValue * settings.collateralFactor) / (borrowValue * 100);
    }

    function getOraclePrice(address token) internal view returns (uint256) {
        return 1000;
    }

}
"""

# solidity_code="pragma solidity ^0.8.17;\n\ninterface IERC20 {\n    function totalSupply() external view returns (uint256);\n\n    function balanceOf(address account) external view returns (uint256);\n\n    function transfer(address recipient, uint256 amount) external returns (bool);\n\n    function allowance(address owner, address spender) external view returns (uint256);\n\n    function approve(address spender, uint256 amount) external returns (bool);\n\n    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);\n\n}\n\ncontract DynamicFinanceProtocol {\n    address public governance;\n    address public teamMultisig;\n    uint256 public lastUpdateTimestamp;\n    uint256 public totalValueLocked;\n    struct CollateralSettings {\n        uint256 collateralFactor;\n        uint256 liquidationThreshold;\n        uint256 liquidationPenalty;\n        bool isActive;\n        uint256 supplyCap;\n    }\n    struct PoolSettings {\n        uint256 interestRate;\n        uint256 utilizationTarget;\n        uint256 reserveFactor;\n        uint256 liquidityMiningRate;\n        bool isPaused;\n    }\n    struct LiquidationParams {\n        uint256 maxDiscountRate;\n        uint256 liquidationDelay;\n        uint256 gracePeriod;\n        uint256 minLiquidationAmount;\n    }\n    struct Protocol {\n        mapping(uint32 => Pool) pools;\n        LiquidationParams liquidationParams;\n        uint256 removeLimitOrderFee;\n        uint256 minimumOrderNotional;\n        uint256 minRequiredMargin;\n    }\n    struct Pool {\n        PoolSettings settings;\n        uint256 totalDeposits;\n        uint256 totalBorrows;\n        mapping(address => uint256) userDeposits;\n    }\n    Protocol public protocol;\n    mapping(IERC20 => CollateralSettings) public collateralConfigs;\n    event CollateralSettingsUpdated(address indexed cToken, CollateralSettings settings);\n    event PoolSettingsUpdated(uint32 indexed poolId, PoolSettings settings);\n    event ProtocolSettingsUpdated(LiquidationParams liquidationParams, uint256 removeLimitOrderFee, uint256 minimumOrderNotional, uint256 minRequiredMargin);\n    event PoolCreated(uint32 indexed poolId, PoolSettings settings);\n    event UserPositionLiquidated(address indexed user, uint256 amount, uint256 penalty);\n    modifier onlyGovernanceOrTeamMultisig() {\n        require(msg.sender == governance || msg.sender == teamMultisig, 'Unauthorized: not governance or team multisig');\n        _;\n    }\n    constructor(address _governance, address _teamMultisig) {\n        governance = _governance;\n        teamMultisig = _teamMultisig;\n        lastUpdateTimestamp = block.timestamp;\n        protocol.liquidationParams = LiquidationParams({maxDiscountRate: 10, liquidationDelay: 1 hours, gracePeriod: 6 hours, minLiquidationAmount: 100 ether});\n        protocol.removeLimitOrderFee = 0.001 ether;\n        protocol.minimumOrderNotional = 0.1 ether;\n        protocol.minRequiredMargin = 0.05 ether;\n    }\n\n    function _updateCollateralSettings(IERC20 cToken, CollateralSettings memory collateralSettings) internal {\n        collateralConfigs[cToken] = collateralSettings;\n        emit CollateralSettingsUpdated(address(cToken), collateralSettings);\n    }\n\n    function modifyCollateralConfig(IERC20 cToken, CollateralSettings memory collateralSettings) external onlyGovernanceOrTeamMultisig {\n        _updateCollateralSettings(cToken, collateralSettings);\n    }\n\n    function adjustPoolConfig(uint32 poolId, PoolSettings calldata newSettings) public onlyGovernanceOrTeamMultisig {\n        protocol.pools[poolId].settings = newSettings;\n        emit PoolSettingsUpdated(poolId, newSettings);\n    }\n\n    function modifyProtocolSettings(LiquidationParams calldata _liquidationParams, uint256 _removeLimitOrderFee, uint256 _minimumOrderNotional, uint256 _minRequiredMargin) external onlyGovernanceOrTeamMultisig {\n        protocol.liquidationParams = _liquidationParams;\n        protocol.removeLimitOrderFee = _removeLimitOrderFee;\n        protocol.minimumOrderNotional = _minimumOrderNotional;\n        protocol.minRequiredMargin = _minRequiredMargin;\n        emit ProtocolSettingsUpdated(_liquidationParams, _removeLimitOrderFee, _minimumOrderNotional, _minRequiredMargin);\n    }\n\n    function createNewPool(uint32 poolId, PoolSettings calldata initialSettings) external onlyGovernanceOrTeamMultisig {\n        require(protocol.pools[poolId].settings.interestRate == 0, 'Pool already exists');\n        protocol.pools[poolId].settings = initialSettings;\n        totalValueLocked += protocol.pools[poolId].totalDeposits;\n        emit PoolCreated(poolId, initialSettings);\n    }\n\n    function executeEmergencyShutdown(uint32 poolId) external onlyGovernanceOrTeamMultisig {\n        PoolSettings storage settings = protocol.pools[poolId].settings;\n        settings.isPaused = true;\n        settings.interestRate = 0;\n        settings.liquidityMiningRate = 0;\n        emit PoolSettingsUpdated(poolId, settings);\n    }\n\n    function calculateHealthFactor(address user, IERC20 collateralToken, uint256 borrowAmount) public view returns (uint256) {\n        uint256 collateralValue = collateralToken.balanceOf(user) * getOraclePrice(address(collateralToken));\n        uint256 borrowValue = borrowAmount;\n        CollateralSettings memory settings = collateralConfigs[collateralToken];\n        if (borrowValue == 0) {\n            return type(uint256).max;\n        }\n        return (collateralValue * settings.collateralFactor) / (borrowValue * 100);\n    }\n\n    function getOraclePrice(address token) internal view returns (uint256) {\n        return 1000;\n    }\n\n}\n\n"

# Mistral-formatted prompt
prompt = f"""
You are analyzing a Solidity smart contract to find vulnerabilities.

Given code with line numbers, return a JSON array where each object describes a single security issue.

Use this format:
[
  {{
    "fromLine": INT,
    "toLine": INT,
    "vulnerabilityClass": STRING,
    "testCase": STRING,
    "description": STRING,
    "priorArt": ARRAY,
    "fixedLines": STRING
  }}
]

If the entire code is invalid or unprocessable, return this:
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
- Each issue must include a realistic testCase example and a clear fix in fixedLines.

Return only the JSON output.

Code:
{solidity_code}
"""

def search_jsonl_code_in_contract(contract_code_str, jsonl_file_path):
    """
    Given a contract code string and a .jsonl file path,
    return all JSON records whose 'code' is found in the contract code.
    """
    matches = []
    
    # Load JSONL records
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if record['code'] in contract_code_str:
                matches.append(record)
    
    return matches

# Search for matches
matched_records = search_jsonl_code_in_contract(solidity_code, 'db1.json')

# Print results
for match in matched_records:
    print(f"Matched Hash: {match['hash']}, Type: {match['type']}")
    print('-' * 50)

exit(0)


# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=8096, temperature=0.2, do_sample=False)
# Calculate the length of the input prompt
prompt_length = inputs["input_ids"].shape[1]

# Extract only the generated tokens
generated_tokens = output[0][prompt_length:]
# Decode and print
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print(tokenizer.decode(generated_tokens, skip_special_tokens=True)[8:])
