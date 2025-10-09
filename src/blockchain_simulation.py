# ==============================================================================
# blockchain_simulation.py
# ------------------------------------------------------------------------------
# This script serves as the central orchestrator for the entire research project.
# It integrates all other components to execute a live, end-to-end simulation
# of the proposed decentralised ledger architecture.
#
# Its primary responsibilities include:
#   1. Connecting to a live Anvil blockchain node via a Web3 provider.
#   2. Instantiating the deployed `MemoryLedger` smart contract using its
#      address and ABI.
#   3. Running a series of `T` independent simulation loops to ensure
#      statistical validity.
#   4. In each loop, simulating `N` agents that create, sign, and submit
#      memory entries as real transactions to the blockchain.
#   5. Collecting raw performance data on transaction throughput (TT),
#      verification latency (VL), and provenance verification accuracy (PVA).
#   6. Calculating the final project metrics based on the collected raw data.
#   7. Performing a comprehensive statistical analysis on the aggregated metrics
#      from all runs, including descriptive statistics and hypothesis tests.
#   8. Generating and saving visualizations (e.g., boxplots, histograms)
#      to support the findings in the final research report.
# ==============================================================================

import time
import random
import json
import binascii # Added to handle potential hex decoding errors
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List

# --- Import from your other project files ---
# Ensure you have agent.py in the same src/ folder
from agent import Agent, verify_signature

# --- Import the REAL web3 library ---
from web3 import Web3

# --- Blockchain Connection Setup ---
# This assumes Anvil is running on the default local RPC endpoint
anvil_url = "http://127.0.0.1:8545"
w3 = Web3(Web3.HTTPProvider(anvil_url))

# Check the connection
if not w3.is_connected():
    print("FATAL: Could not connect to Anvil.")
    print(f"Please make sure Anvil is running and accessible at {anvil_url}")
    exit()

# --- Smart Contract Details ---
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"

CONTRACT_ABI_JSON = """
[
  {
    "type": "function",
    "name": "addEntry",
    "inputs": [
      {
        "name": "_agentAddress",
        "type": "string",
        "internalType": "string"
      },
      {
        "name": "_content",
        "type": "string",
        "internalType": "string"
      },
      {
        "name": "_signature",
        "type": "string",
        "internalType": "string"
      }
    ],
    "outputs": [],
    "stateMutability": "nonpayable"
  },
  {
    "type": "event",
    "name": "EntryAdded",
    "inputs": [
      {
        "name": "timestamp",
        "type": "uint256",
        "indexed": true,
        "internalType": "uint256"
      },
      {
        "name": "agentAddress",
        "type": "string",
        "indexed": false,
        "internalType": "string"
      },
      {
        "name": "content",
        "type": "string",
        "indexed": false,
        "internalType": "string"
      },
      {
        "name": "signature",
        "type": "string",
        "indexed": false,
        "internalType": "string"
      }
    ],
    "anonymous": false
  }
]
"""
CONTRACT_ABI = json.loads(CONTRACT_ABI_JSON)

# Using one of Anvil's pre-funded accounts to pay for gas fees
w3.eth.default_account = w3.eth.accounts[0]
ledger_contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)


# --- Core Simulation Function ---
def run_live_simulation(num_entries: int) -> Dict:
    """
    Runs a simulation that interacts with a live Anvil blockchain, focusing on
    PVA, VL, and TT metrics.
    """
    agents = [Agent() for _ in range(3)]
    all_submitted_entries = []

    print(f"  -> Submitting {num_entries} entries to the blockchain...")
    start_time = time.time()
    for i in range(num_entries):
        agent = random.choice(agents)
        message_content = {"data": f"live_entry_{i}", "timestamp": time.time()}
        signature = agent.sign_message(message_content)
        entry = {
            "public_key": agent.public_key.to_string().hex(),
            "message": message_content,
            "signature": signature
        }
        
        try:
            tx_hash = ledger_contract.functions.addEntry(
                agent.address,
                json.dumps(entry["message"]),
                entry["signature"]
            ).transact()
            w3.eth.wait_for_transaction_receipt(tx_hash)
            all_submitted_entries.append(entry)
        except Exception as e:
            print(f"Warning: Transaction failed: {e}")

    end_time = time.time()
    simulation_duration_seconds = end_time - start_time
    total_successful_transactions = len(all_submitted_entries)
    
    print("  -> Verifying all submitted entries...")
    legitimate_verification_results = []
    verification_latencies_ms = []
    for entry in all_submitted_entries:
        latency_start = time.time()
        is_valid = verify_signature(entry['public_key'], entry['signature'], entry['message'])
        latency_end = time.time()
        verification_latencies_ms.append((latency_end - start_time) * 1000)
        legitimate_verification_results.append(is_valid)

    return {
        'legitimate_verification_results': legitimate_verification_results,
        'verification_latencies_ms': verification_latencies_ms,
        'total_successful_transactions': total_successful_transactions,
        'simulation_duration_seconds': simulation_duration_seconds,
    }

# --- Metric Calculation and Analysis ---
def calculate_live_metrics(simulation_data: Dict) -> Dict[str, float]:
    """Calculates metrics relevant to the live simulation."""
    metrics = {}
    legit_results = simulation_data.get('legitimate_verification_results', [])
    latencies = simulation_data.get('verification_latencies_ms', [])
    total_txs = simulation_data.get('total_successful_transactions', 0)
    duration = simulation_data.get('simulation_duration_seconds', 0)
    
    metrics['PVA'] = sum(legit_results) / len(legit_results) if legit_results else 1.0
    metrics['VL'] = np.mean(latencies) if latencies else 0.0
    metrics['TT'] = total_txs / duration if duration > 0 else 0.0
    metrics['TDR'] = 1.0  # Default value, not measured in live run
    metrics['FPR'] = 0.0  # Default value
    metrics['ACR'] = 1.0  # Default value
    return metrics

def analyze_and_visualize_results(all_run_metrics: List[Dict], latencies_from_last_run: List[float]):
    """Performs full statistical analysis and generates plots."""
    print("\n--- Full Statistical Analysis ---")
    df = pd.DataFrame(all_run_metrics)
    
    # 1. Calculate Descriptive Statistics and 95% Confidence Intervals
    summary = df.describe(percentiles=[.025, .975]).transpose()
    # Due to small sample sizes (T<30), t-distribution has been selected for accuracy for C.I.
    # However, direct percentiles are a robust non-parametric estimate. Both are provided.
    conf_intervals = {}
    for col in df.columns:
        if len(df[col]) > 1:
            mean, std, count = df[col].mean(), df[col].std(), len(df[col])
            ci = stats.t.interval(0.95, df=count-1, loc=mean, scale=std/np.sqrt(count))
            conf_intervals[col] = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        else:
            conf_intervals[col] = "[N/A]"
    summary['95% C.I. (t-dist)'] = pd.Series(conf_intervals)
    print("Descriptive Statistics Across All Runs:")
    print(summary[['mean', 'std', '95% C.I. (t-dist)']])
    
    # 2. Perform Hypothesis Testing
    print("\nHypothesis Test Results (alpha = 0.05):")
    # H0: PVA < 0.99, H1: PVA >= 0.99
    pva_target = 0.99
    if len(df['PVA']) > 1:
        t_stat_pva, p_value_pva = stats.ttest_1samp(df['PVA'], pva_target, alternative='greater')
        print(f"PVA >= {pva_target}: t-statistic={t_stat_pva:.4f}, p-value={p_value_pva:.4f}")
        if p_value_pva < 0.05 and df['PVA'].mean() >= pva_target:
            print("  -> RESULT: Reject H0. The system's PVA meets the target.")
        else:
            print("  -> RESULT: Fail to reject H0. The system's PVA does not meet the target.")
    else:
        print("PVA: Not enough data for t-test.")

    # TDR, FPR, ACR were not measured, so no hypothesis tests are run for them.

    # 3. Generate and Save Visualizations
    print("\nGenerating and saving plots...")
    # Boxplot of all metrics
    df.plot(kind='box', subplots=True, layout=(2, 3), figsize=(15, 8), title="Distribution of Metrics Across All Simulation Runs")
    plt.tight_layout()
    plt.savefig("live_metrics_boxplot.png")
    print("  -> Saved 'live_metrics_boxplot.png'")
    
    # Histogram of Verification Latency from the last run
    if latencies_from_last_run:
        plt.figure(figsize=(10, 5))
        plt.hist(latencies_from_last_run, bins=30, edgecolor='black')
        plt.title("Distribution of Verification Latency (Last Run)")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.5)
        plt.savefig("live_latency_histogram.png")
        print("  -> Saved 'live_latency_histogram.png'")
    plt.close('all')

# --- Main Execution Block ---
if __name__ == "__main__":
    T = 3    # Number of simulation runs. Testing: 3 Final: 30
    N = 50   # Number of transactions per run. Testing: 50 Final: 1000
    
    all_run_metrics = []
    latencies_from_last_run = []

    print(f"Starting LIVE simulation series: {T} runs, {N} entries each.")
    
    for i in range(T):
        print(f"\n--- Running LIVE Simulation {i + 1}/{T} ---")
        simulation_data = run_live_simulation(num_entries=N)
        metrics_for_this_run = calculate_live_metrics(simulation_data)
        all_run_metrics.append(metrics_for_this_run)
        
        if i == T - 1:
            latencies_from_last_run = simulation_data['verification_latencies_ms']
            
        print(f"  -> Run {i + 1} Complete. Metrics:")
        print(pd.Series(metrics_for_this_run).to_string())

    analyze_and_visualize_results(all_run_metrics, latencies_from_last_run)
    print("\nLive simulation and analysis complete.")