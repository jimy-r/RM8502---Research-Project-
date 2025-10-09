// SPDX-License-Identifier: MIT
// =============================================================================
// MemoryLedger.sol
// -----------------------------------------------------------------------------
// This smart contract provides a simple, append-only ledger for storing signed
// memory entries from multiple agents. Its primary purpose is to create a
// trustworthy, on-chain log of events. It prioritises gas efficiency by using
// events for data storage, making the data accessible to off-chain services
// (like this project's Python simulation script) without consuming expensive contract storage.
// =============================================================================

// Specifies the version of the Solidity compiler this code is written for.
// The caret (^) means it will compile with version 0.8.0 and any later minor
// or patch versions (e.g., 0.8.1, 0.8.20) but not with 0.9.0 or higher.
pragma solidity ^0.8.0;

/**
 * @title MemoryLedger
 * @dev A simple ledger contract for logging memory entries from agents.
 */
contract MemoryLedger {

    // --- 1. Struct Definition (Data Blueprint) ---
    // -------------------------------------------------------------------------
    // This struct defines a data structure for a memory entry. While it's not
    // directly used to store data on-chain in this contract (to save gas), it
    // serves as a clear blueprint for the data that the `EntryAdded` event will log.
    struct MemoryEntry {
        uint timestamp;       // The time the entry was recorded.
        string agentAddress;  // The unique identifier of the agent.
        string content;       // The actual data/content of the memory entry.
        string signature;     // The agent's cryptographic signature for the content.
    }

    // --- 2. Event Definition (The Logging Mechanism) ---
    // -------------------------------------------------------------------------
    // Events are a core feature of the Ethereum Virtual Machine (EVM) for logging.
    // Emitting an event is significantly cheaper than writing to contract storage.
    // Data from events is stored in a special data structure on the blockchain
    // (transaction logs) which is easily queryable from the outside (e.g., via web3.py).
    event EntryAdded(
        // The `indexed` keyword allows for efficient searching/filtering of logs
        // based on this parameter. We index the timestamp to make it easy to
        // query for entries within a certain time range.
        uint indexed timestamp,

        // The public key or address of the agent who submitted the entry.
        string agentAddress,

        // The JSON string representing the content of the memory entry.
        string content,

        // The hexadecimal string of the signature, for off-chain verification.
        string signature
    );

    // --- 3. Public Function (The Entry Point) ---
    // -------------------------------------------------------------------------
    /**
     * @dev Public function that allows any agent to add a new signed entry to the ledger.
     * @param _agentAddress The public address of the signing agent.
     * @param _content The JSON content of the memory entry.
     * @param _signature The cryptographic signature of the content.
     */
    function addEntry(string memory _agentAddress, string memory _content, string memory _signature) public {
        // The core action of this function is to `emit` the `EntryAdded` event.
        // This takes the data provided by the caller and records it in the
        // transaction logs of the block where this transaction is included.
        //
        // `block.timestamp` is a global variable in Solidity that provides the
        // timestamp of the current block, ensuring a secure and tamper-proof
        // record of when the entry was officially added to the ledger.
        emit EntryAdded(block.timestamp, _agentAddress, _content, _signature);
    }
}