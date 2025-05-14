git clone https://github.com/ReinforcedAIAudits/solidity-audit.git
cd solidity-audit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

pip install transformers
pip install torch
pip install huggingface-hub
pip install accelerate
huggingface-cli login
key input

python model_servers/miner2.py
