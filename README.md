# Decentralised Ledger Architecture for Trustworthy Multi-Agent Memory
**A Research Project for RM8501 - Research Planning**

This repository contains the code and documentation for the research project, "Decentralised Ledger Architecture for Trustworthy Multi-Agent Memory." The project focuses on designing, implementing, and evaluating a blockchain-inspired system to ensure data provenance and integrity in multi-agent systems (MAS).

## Abstract

Long-term memory is a cornerstone for autonomous Large Language Model (LLM) agents, yet extending memory capabilities to multi-agent systems (MAS) introduces significant challenges, particularly around trust, provenance, and data integrity in a decentralised environment. Existing work largely overlooks robust, quantifiable mechanisms for shared memory, thereby limiting effective collaboration and coordination among autonomous agents. This research proposal addresses that gap by outlining and planning to quantitatively evaluate a blockchain-inspired decentralised ledger architecture for LLM-based agents. This project serves as a foundational, data-driven proof-of-concept to fill this important literature gap and provide a solid foundation for future, scaled-up research in collaborative AI systems.

## Core Research Question

> How can core components of a blockchain-inspired decentralised ledger architecture be designed to ensure verifiable provenance and tamper-evidence for shared memory entries among potentially non-cooperative LLM-based agents?

## Project Goals

1.  **Architecture Blueprint**: Design a detailed, practical architecture for core DLT-based system components tailored for trustworthy shared memory in MAS.
2.  **Foundational Evaluation Framework**: Define and implement a quantitative framework with statistical analysis to rigorously assess the effectiveness of the proposed architecture.
3.  **Proof-of-Concept Simulation**: Develop a minimal Python-based simulation to demonstrate the core trustworthiness actions (signing, appending, verifying) and collect performance data.

## Architecture Overview

The system operates on a simple yet robust principle:

1.  **Signing**: An agent creates a memory entry (e.g., a simple factual statement) and cryptographically signs it using its unique private key.
2.  **Appending**: The agent submits this signed entry as a transaction to a shared, append-only ledger running on a local EVM-compatible blockchain (Anvil).
3.  **Querying & Verifying**: Any other agent can query the ledger to retrieve memory entries. It can then verify the entry's **provenance** (origin) by checking the cryptographic signature against the claimed agent's public key and its **integrity** (that it hasn't been altered) by checking its hash.

## Technology Stack

| Layer          | Choice                                                     | Rationale                                                     |
| -------------- | ---------------------------------------------------------- | ------------------------------------------------------------- |
| **Ledger**     | Anvil (Foundry toolkit, fork of OP Stack)                  | Fast, local EVM node for development and simulation.          |
| **Agent Runtime**| Python 3.10+ (with `asyncio`)                              | Lightweight, ideal for I/O operations and wallet interactions.|
| **Blockchain Lib** | `web3.py`                                                  | Standard Python library for interacting with Ethereum nodes.  |
| **Cryptography** | `ecdsa`, `hashlib`                                         | Standard libraries for digital signatures and hashing.        |
| **Smart Contracts**| Solidity 0.8.x (Optional)                                | Standard for EVM logic if a contract-based approach is used.  |
| **Analysis**   | `pandas`, `numpy`, `scipy`, `jupyter`                      | For data collection, statistical analysis, and reporting.     |

## Setup and Installation

Follow these steps to set up the development environment.

**1. Clone the Repository:**
```bash
git clone https://github.com/your-username/trustworthy-mas-memory.git
cd trustworthy-mas-memory

**2. Install Foundry (for Anvil):**
Foundry is used to run the local blockchain node. Install it by following the official instructions:
curl -L https://foundry.paradigm.xyz | bash
foundryup

3. Set up Python Environment:
It is highly recommended to use a Python virtual environment.
code
Bash
python3 -m venv venv
source venv/bin/activate
# On Windows, use: venv\Scripts\activate

4. Install Python Dependencies:
code
Bash
pip install -r requirements.txt

#Running the Simulation
1. Start the Local Blockchain:
In a separate terminal, start the Anvil development node. It will print a list of available accounts and private keys.
code
Bash
anvil
Keep this terminal window running.

2. Configure the Environment:
If a smart contract is used, it must be deployed first.
code
Bash
# Example deployment command using Forge
forge create --rpc-url http://127.0.0.1:8545 --private-key <YOUR_ANVIL_PRIVATE_KEY> src/MemoryLedger.sol:MemoryLedger
Update the configuration file (config.py or similar) with the deployed contract address and the RPC URL (http://127.0.0.1:8545).

3. Execute the Simulation:
Run the main simulation script to collect data.
code
Bash
python src/simulation.py
The script will run T=10 independent simulations, automatically calculate the metrics for each run, and store the aggregated results in the data/ directory as a CSV file.

4. Analyze the Results:
A Jupyter notebook is provided to parse the output CSV, perform statistical validation, and generate plots.
jupyter notebook notebooks/analysis.ipynb
```

## Evaluation and Success Metrics

The success of this project is quantified by six key metrics, evaluated over T=10 repeated simulation runs. The goal is to meet the following predefined thresholds:

| Metric                          | Abbreviation | Target Value  |
| ------------------------------- | ------------ | ------------- |
| Provenance Verification Accuracy| PVA          | ≥ 99%         |   
| Verification Latency            | VL           | ≤ 150 ms      |
| Transaction Throughput          | TT           | ≥ 100 txn/sec |
| Average Gas Cost	              | AvgGas       | nan           |

## Project Structure
```
.
├── contracts/
│ └── MemoryLedger.sol # Optional Solidity smart contract
├── data/
│ └── simulation_results.csv # Output data from the simulation
├── src/
│ ├── agent.py # Agent class with signing logic
│ ├── blockchain_simulation.py # Main script to run the experiment 
├── .gitignore
├── README.md
└── requirements.txt # Python dependencies
```

## Limitations
Narrow Scope: The project does not design novel consensus mechanisms, integrate complex LLM reasoning, or perform formal security proofs.
Simulated Environment: Results are from a local, controlled environment and may not reflect performance on a public or high-latency network.
Infrastructure Assumption: The project assumes a pre-existing, reliable DLT infrastructure for consensus and basic ledger operations.

## Author and Supervisor
Author: James Ross
Primary Advisor: Pro. Ickjai Lee

## License
This project is licensed under the MIT License. See the LICENSE file for details.
