
use alloy::{
    consensus::{
        BlobTransactionSidecar, SignableTransaction, TxEip4844, TxEip4844Variant,
        TxEip4844WithSidecar, TxEnvelope,
    },
    eips::{eip2718::Encodable2718, eip2930::AccessList, eip4844::BYTES_PER_BLOB},
    network::{Ethereum, TxSigner},
    primitives::{bytes, FixedBytes, U256},
    providers::{Provider, ProviderBuilder},
    rpc::client::RpcClient,
    signers::{k256::elliptic_curve::rand_core::block, wallet::LocalWallet},
};
use c_kzg::{Blob, KzgCommitment, KzgProof, KzgSettings};
use dotenv::dotenv;
use starknet::core::types::{
    BlockId, FieldElement, MaybePendingStateUpdate, StateUpdate, StorageEntry,
};
use starknet::providers::jsonrpc::HttpTransport;
// use starknet::providers::Provider;
use color_eyre::eyre::eyre;
use color_eyre::Result;
use num::BigInt;
use serde_json::Value;
use starknet::providers::{JsonRpcClient, Provider as skProvider, Url};
use std::fs;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::io::{Error, Write};
use std::str::FromStr;
use std::{collections::HashMap, os::macos::raw::stat};
use std::{env, path::Path};

use lazy_static::lazy_static;
use majin_blob_core::blob;
use majin_blob_types::serde;
use num_bigint::{BigUint, ToBigUint};
use num_traits::Num;
use num_traits::{One, Zero};
// use eyre::Result;

pub const BLOB_LEN: usize = 4096;
lazy_static! {
    /// EIP-4844 BLS12-381 modulus.
    ///
    /// As defined in https://eips.ethereum.org/EIPS/eip-4844
    pub static ref BLS_MODULUS: BigUint = BigUint::from_str(
        "52435875175126190479447740508185965837690552500527637822603658699938581184513",
    )
    .unwrap();
    /// Generator of the group of evaluation points (EIP-4844 parameter).
    pub static ref GENERATOR: BigUint = BigUint::from_str(
        "39033254847818212395286706435128746857159659164139250548781411570340225835782",
    )
    .unwrap();
    pub static ref TWO: BigUint = 2u32.to_biguint().unwrap();
}


async fn send_Tx_4844(block_no: u64) -> Result<()> {
    dotenv().ok();

    let provider = JsonRpcClient::new(HttpTransport::new(
        Url::parse(get_env_var_or_panic("RPC_STARKNET_INFURA_MAIN").as_str())
            .expect("Failed to parse URL"),
    ));
    let state_update = provider.get_state_update(BlockId::Number(block_no)).await?;

    let state_update = match state_update {
        MaybePendingStateUpdate::PendingUpdate(_) => {
            println!("failed for block no {:?}", block_no);
            return Err(eyre!(
                "Cannot process block {} for job id  as it's still in pending state",
                block_no
            ));
        }
        MaybePendingStateUpdate::Update(state_update) => state_update,
    };
    let state_update_value = state_update_to_blob_data(block_no, state_update);

    let biguint_vec = convert_to_biguint(state_update_value);

    let self_state_diffs = serde::parse_state_diffs(biguint_vec.as_slice());
    let self_state_diffs_json = serde::to_json(&self_state_diffs.as_slice());
    fs::write("self_json_creation.txt", self_state_diffs_json)?;

    
    
    let blob_data = serde::parse_file_to_blob_data("./test_blob.txt");

    // Recover the original data
    let original_data = blob::recover(blob_data);
    let state_diffs = serde::parse_state_diffs(original_data.as_slice());
    let state_diffs_json = serde::to_json(state_diffs.as_slice());
    fs::write("from_recover_json.txt", state_diffs_json)?;


    Ok(()) 
}
pub fn get_env_var(key: &str) -> Result<String> {
    dotenv().ok();
    let variable_here = env::var(key).expect("PK must be set");
    Ok(variable_here)
    // std::env::var(key).map_err(|e| e.into())
}

pub fn get_env_var_or_panic(key: &str) -> String {
    get_env_var(key).unwrap_or_else(|e| panic!("Failed to get env var {}: {}", key, e))
}

fn state_update_to_blob_data(block_no: u64, state_update: StateUpdate) -> Vec<FieldElement> {
    let state_diff = state_update.state_diff;
    let mut blob_data: Vec<FieldElement> = vec![
        // TODO: confirm first three fields
        FieldElement::from(state_diff.storage_diffs.len()),
        FieldElement::ONE,
        FieldElement::ONE,
        FieldElement::from(block_no),
        state_update.block_hash,
    ];

    let storage_diffs: HashMap<FieldElement, &Vec<StorageEntry>> = state_diff
        .storage_diffs
        .iter()
        .map(|item| (item.address, &item.storage_entries))
        .collect();
    let declared_classes: HashMap<FieldElement, FieldElement> = state_diff
        .declared_classes
        .iter()
        .map(|item| (item.class_hash, item.compiled_class_hash))
        .collect();
    let deployed_contracts: HashMap<FieldElement, FieldElement> = state_diff
        .deployed_contracts
        .iter()
        .map(|item| (item.address, item.class_hash))
        .collect();
    let replaced_classes: HashMap<FieldElement, FieldElement> = state_diff
        .replaced_classes
        .iter()
        .map(|item| (item.contract_address, item.class_hash))
        .collect();
    let mut nonces: HashMap<FieldElement, FieldElement> = state_diff
        .nonces
        .iter()
        .map(|item| (item.contract_address, item.nonce))
        .collect();

    // Loop over storage diffs
    for (addr, writes) in storage_diffs {
        blob_data.push(addr);

        let class_flag = deployed_contracts
            .get(&addr)
            .or_else(|| replaced_classes.get(&addr));

        let nonce = nonces.remove(&addr);
        blob_data.push(da_word(class_flag.is_some(), nonce, writes.len() as u64));

        if let Some(class_hash) = class_flag {
            blob_data.push(*class_hash);
        }

        for entry in writes {
            blob_data.push(entry.key);
            blob_data.push(entry.value);
        }
    }

    // Handle nonces
    for (addr, nonce) in nonces {
        blob_data.push(addr);

        let class_flag = deployed_contracts
            .get(&addr)
            .or_else(|| replaced_classes.get(&addr));

        blob_data.push(da_word(class_flag.is_some(), Some(nonce), 0_u64));
        if let Some(class_hash) = class_flag {
            blob_data.push(*class_hash);
        }
    }

    // Handle deployed contracts
    for (addr, class_hash) in deployed_contracts {
        blob_data.push(addr);

        blob_data.push(da_word(true, None, 0_u64));
        blob_data.push(class_hash);
    }

    // Handle declared classes
    blob_data.push(FieldElement::from(declared_classes.len()));

    for (class_hash, compiled_class_hash) in &declared_classes {
        blob_data.push(*class_hash);
        blob_data.push(*compiled_class_hash);
    }

    blob_data
}

fn da_word(class_flag: bool, nonce_change: Option<FieldElement>, num_changes: u64) -> FieldElement {
    const CLASS_FLAG_TRUE: &str = "0x100000000000000000000000000000001"; // 2 ^ 128 + 1
    const NONCE_BASE: &str = "0xFFFFFFFFFFFFFFFF"; // 2 ^ 64 - 1

    let mut word = FieldElement::ZERO;

    if class_flag {
        word += FieldElement::from_hex_be(CLASS_FLAG_TRUE).unwrap();
    }
    if let Some(new_nonce) = nonce_change {
        word += new_nonce + FieldElement::from_hex_be(NONCE_BASE).unwrap();
    }

    word += FieldElement::from(num_changes);

    word
}

fn convert_to_biguint(elements: Vec<FieldElement>) -> Vec<BigUint> {
    let mut biguint_vec = Vec::with_capacity(4096);

    // Iterate over the first 4096 elements of the input vector or until we reach 4096 elements
    for i in 0..4096 {
        if let Some(element) = elements.get(i) {
            // Convert FieldElement to [u8; 32]
            let bytes: [u8; 32] = element.to_bytes_be();

            // Convert [u8; 32] to BigUint
            let biguint = BigUint::from_bytes_be(&bytes);

            biguint_vec.push(biguint);
        } else {
            // If we run out of elements, push a zero BigUint
            biguint_vec.push(BigUint::zero());
        }
    }

    biguint_vec
}
#[tokio::main]
async fn main() -> Result<()> {
    send_Tx_4844(630872).await?;
    Ok(())
}