from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
)

#tokenizer = AutoTokenizer.from_pretrained("./mixtral-finetuned")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=bnb_config, device_map="auto", trust_remote_code=True)

audit_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_audit(source: str):
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": source},
    ]
    input_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print(input_prompt)
#    output = audit_pipe(input_prompt, max_new_tokens=1024, do_sample=False)[0]["generated_text"]
    output = audit_pipe(input_prompt, max_new_tokens=1024, do_sample=False)
    return output


PROMPT = """
You are analyzing a Solidity smart contract to find vulnerabilities.

Given code with line numbers, return a JSON array where each object describes a single security issue.

Use this format:
[
  {
    "fromLine": INT,
    "toLine": INT,
    "vulnerabilityClass": STRING,
    "testCase": STRING,
    "description": STRING,
    "priorArt": ARRAY,
    "fixedLines": STRING
  }
]

If the entire code is invalid or unprocessable, return this:
[
  {
    "fromLine": 1,
    "toLine": TOTAL_LINE_COUNT,
    "vulnerabilityClass": "Invalid Code",
    "description": "The entire code is considered invalid for audit processing."
  }
]

Rules:
- Do not include extra text or explanations.
- Use only these vulnerabilityClass values: "Known compiler bugs", "Reentrancy", "Gas griefing", "Oracle manipulation", "Bad randomness", "Unexpected privilege grants", "Forced reception", "Integer overflow/underflow", "Race condition", "Unguarded function", "Inefficient storage key", "Front-running potential", "Miner manipulation", "Storage collision", "Signature replay", "Unsafe operation".
- Each issue must include a realistic testCase example and a clear fix in fixedLines.

Return only the JSON output.
""".strip()

code="pragma solidity ^0.8.17;\n\ninterface IERC20 {\n    function totalSupply() external view returns (uint256);\n\n    function balanceOf(address account) external view returns (uint256);\n\n    function transfer(address recipient, uint256 amount) external returns (bool);\n\n    function allowance(address owner, address spender) external view returns (uint256);\n\n    function approve(address spender, uint256 amount) external returns (bool);\n\n    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);\n\n}\n\ncontract DynamicFinanceProtocol {\n    address public governance;\n    address public teamMultisig;\n    uint256 public lastUpdateTimestamp;\n    uint256 public totalValueLocked;\n    struct CollateralSettings {\n        uint256 collateralFactor;\n        uint256 liquidationThreshold;\n        uint256 liquidationPenalty;\n        bool isActive;\n        uint256 supplyCap;\n    }\n    struct PoolSettings {\n        uint256 interestRate;\n        uint256 utilizationTarget;\n        uint256 reserveFactor;\n        uint256 liquidityMiningRate;\n        bool isPaused;\n    }\n    struct LiquidationParams {\n        uint256 maxDiscountRate;\n        uint256 liquidationDelay;\n        uint256 gracePeriod;\n        uint256 minLiquidationAmount;\n    }\n    struct Protocol {\n        mapping(uint32 => Pool) pools;\n        LiquidationParams liquidationParams;\n        uint256 removeLimitOrderFee;\n        uint256 minimumOrderNotional;\n        uint256 minRequiredMargin;\n    }\n    struct Pool {\n        PoolSettings settings;\n        uint256 totalDeposits;\n        uint256 totalBorrows;\n        mapping(address => uint256) userDeposits;\n    }\n    Protocol public protocol;\n    mapping(IERC20 => CollateralSettings) public collateralConfigs;\n    event CollateralSettingsUpdated(address indexed cToken, CollateralSettings settings);\n    event PoolSettingsUpdated(uint32 indexed poolId, PoolSettings settings);\n    event ProtocolSettingsUpdated(LiquidationParams liquidationParams, uint256 removeLimitOrderFee, uint256 minimumOrderNotional, uint256 minRequiredMargin);\n    event PoolCreated(uint32 indexed poolId, PoolSettings settings);\n    event UserPositionLiquidated(address indexed user, uint256 amount, uint256 penalty);\n    modifier onlyGovernanceOrTeamMultisig() {\n        require(msg.sender == governance || msg.sender == teamMultisig, 'Unauthorized: not governance or team multisig');\n        _;\n    }\n    constructor(address _governance, address _teamMultisig) {\n        governance = _governance;\n        teamMultisig = _teamMultisig;\n        lastUpdateTimestamp = block.timestamp;\n        protocol.liquidationParams = LiquidationParams({maxDiscountRate: 10, liquidationDelay: 1 hours, gracePeriod: 6 hours, minLiquidationAmount: 100 ether});\n        protocol.removeLimitOrderFee = 0.001 ether;\n        protocol.minimumOrderNotional = 0.1 ether;\n        protocol.minRequiredMargin = 0.05 ether;\n    }\n\n    function _updateCollateralSettings(IERC20 cToken, CollateralSettings memory collateralSettings) internal {\n        collateralConfigs[cToken] = collateralSettings;\n        emit CollateralSettingsUpdated(address(cToken), collateralSettings);\n    }\n\n    function modifyCollateralConfig(IERC20 cToken, CollateralSettings memory collateralSettings) external onlyGovernanceOrTeamMultisig {\n        _updateCollateralSettings(cToken, collateralSettings);\n    }\n\n    function adjustPoolConfig(uint32 poolId, PoolSettings calldata newSettings) public onlyGovernanceOrTeamMultisig {\n        protocol.pools[poolId].settings = newSettings;\n        emit PoolSettingsUpdated(poolId, newSettings);\n    }\n\n    function modifyProtocolSettings(LiquidationParams calldata _liquidationParams, uint256 _removeLimitOrderFee, uint256 _minimumOrderNotional, uint256 _minRequiredMargin) external onlyGovernanceOrTeamMultisig {\n        protocol.liquidationParams = _liquidationParams;\n        protocol.removeLimitOrderFee = _removeLimitOrderFee;\n        protocol.minimumOrderNotional = _minimumOrderNotional;\n        protocol.minRequiredMargin = _minRequiredMargin;\n        emit ProtocolSettingsUpdated(_liquidationParams, _removeLimitOrderFee, _minimumOrderNotional, _minRequiredMargin);\n    }\n\n    function createNewPool(uint32 poolId, PoolSettings calldata initialSettings) external onlyGovernanceOrTeamMultisig {\n        require(protocol.pools[poolId].settings.interestRate == 0, 'Pool already exists');\n        protocol.pools[poolId].settings = initialSettings;\n        totalValueLocked += protocol.pools[poolId].totalDeposits;\n        emit PoolCreated(poolId, initialSettings);\n    }\n\n    function executeEmergencyShutdown(uint32 poolId) external onlyGovernanceOrTeamMultisig {\n        PoolSettings storage settings = protocol.pools[poolId].settings;\n        settings.isPaused = true;\n        settings.interestRate = 0;\n        settings.liquidityMiningRate = 0;\n        emit PoolSettingsUpdated(poolId, settings);\n    }\n\n    function calculateHealthFactor(address user, IERC20 collateralToken, uint256 borrowAmount) public view returns (uint256) {\n        uint256 collateralValue = collateralToken.balanceOf(user) * getOraclePrice(address(collateralToken));\n        uint256 borrowValue = borrowAmount;\n        CollateralSettings memory settings = collateralConfigs[collateralToken];\n        if (borrowValue == 0) {\n            return type(uint256).max;\n        }\n        return (collateralValue * settings.collateralFactor) / (borrowValue * 100);\n    }\n\n    function getOraclePrice(address token) internal view returns (uint256) {\n        return 1000;\n    }\n\n}\n\n"
#code="function _allocatePerformanceFee(uint256 _performanceFee, address _performanceFeeReceiver, uint256 _performanceFeeDebt, WrappedPiErc20Interface _piToken, IERC20 _underlying, uint256 _totalReward) internal returns (uint256 performance, uint256 remainder, uint256 resultPerformanceFeeDebt) {\n    performance = 0;\n    remainder = 0;\n    resultPerformanceFeeDebt = _performanceFeeDebt;\n    if (_performanceFee > 0) {\n        performance = _totalReward.mul(_performanceFee).div(HUNDRED_PCT);\n        remainder = _totalReward.sub(performance);\n        uint256 performanceFeeDebtBefore = _performanceFeeDebt;\n        uint256 underlyingBalance = _underlying.balanceOf(address(_piToken));\n        uint256 totalFeeToPayOut = performance.add(performanceFeeDebtBefore);\n        if (underlyingBalance >= totalFeeToPayOut) {\n            _safeTransfer(_piToken, _underlying, _performanceFeeReceiver, totalFeeToPayOut);\n        } else {\n            uint256 diff = totalFeeToPayOut.sub(underlyingBalance);\n            resultPerformanceFeeDebt = totalFeeToPayOut.sub(diff);\n            _safeTransfer(_piToken, _underlying, _performanceFeeReceiver, underlyingBalance);\n        }\n        emit DistributePerformanceFee(performanceFeeDebtBefore, resultPerformanceFeeDebt, underlyingBalance, performance);\n    } else {\n        remainder = _totalReward;\n    }\n}"
code="""
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
result = generate_audit(code)

print("+++++++++++++++++++\n")
print(result)