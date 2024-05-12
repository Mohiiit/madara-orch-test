use crate::local_state_diff::{ClassDeclaration, ContractUpdate, DataJson, StorageUpdate};
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
use std::collections::HashSet;

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
// use majin_blob_types::serde;
use num_bigint::{BigUint, ToBigUint};
use num_traits::Num;
use num_traits::{One, Zero};

mod local_serde;
mod local_state_diff;

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

    let write_to_file = write_biguint_to_file(&biguint_vec, "state_file.txt")
        .expect("issue while writing it to the file");

    let self_state_diffs = local_serde::parse_state_diffs(biguint_vec.as_slice());
    let self_state_diffs_json = local_serde::to_json(self_state_diffs);
    fs::write("self_json_creation.txt", self_state_diffs_json)?;

    let blob_data = local_serde::parse_file_to_blob_data("./test_blob_v5.txt");

    // Recover the original data
    let original_data = blob::recover(blob_data);

    let write_to_file = write_biguint_to_file(&original_data, "state_file_recover_fn.txt")
        .expect("issue while writing it to the file");
    let state_diffs = local_serde::parse_state_diffs(original_data.as_slice());
    let state_diffs_json = local_serde::to_json(state_diffs);
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

fn write_biguint_to_file(numbers: &Vec<BigUint>, file_path: &str) -> io::Result<()> {
    let mut file = File::create(file_path)?;
    for number in numbers {
        writeln!(file, "{}", number)?;
    }
    Ok(())
}

fn state_update_to_blob_data(block_no: u64, state_update: StateUpdate) -> Vec<FieldElement> {
    let state_diff = state_update.state_diff;
    let mut blob_data: Vec<FieldElement> = vec![
        // TODO: confirm first three fields
        // TODO: should be number of unique addresses over here
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
        // compare_the_field_element_with_da_word(addr, "address was this".to_string());

        let class_flag = deployed_contracts
            .get(&addr)
            .or_else(|| replaced_classes.get(&addr));

        let nonce = nonces.remove(&addr);
        let da_word_here = da_word(class_flag.is_some(), nonce, writes.len() as u64);
        let (class_flag_here, nonce_here, change_here) = decode_da_word(da_word_here);
        // println!(
        //     "all the data here is {:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?}, ",
        //     addr,
        //     class_flag,
        //     nonce,
        //     writes.len(),
        //     da_word_here,
        //     class_flag_here,
        //     nonce_here,
        //     change_here
        // );
        //all the data here is None,None,3,FieldElement { inner: 0x0000000000000000000000000000000000000000000000000000000000000003 },false,0,3,
        //all the data here is None,Some(FieldElement { inner: 0x0000000000000000000000000000000000000000000000000000000000000200 }),0,FieldElement { inner: 0x00000000000000000000000000000000000000000000000100000000000001ff },false,1,511,
        // all the data here is None,Some(FieldElement { inner: 0x00000000000000000000000000000000000000000000000000000000000004e2 }),0,FieldElement { inner: 0x00000000000000000000000000000000000000000000000100000000000004e1 },false,1,1249,

        // if (addr == FieldElement::ONE && da_word_here == FieldElement::ONE) {
        //     continue;
        // }
        blob_data.push(addr);
        blob_data.push(da_word_here);
        // compare_the_field_element_with_da_word(
        //     da_word_here,
        //     "da word below address was this".to_string(),
        // );

        if let Some(class_hash) = class_flag {
            blob_data.push(*class_hash);
            // compare_the_field_element_with_da_word(
            //     *class_flag.expect("issue in class flag"),
            //     "class hash was this".to_string(),
            // );
        }

        for entry in writes {
            blob_data.push(entry.key);
            blob_data.push(entry.value);
            // compare_the_field_element_with_da_word(entry.key, "key was this".to_string());
            // compare_the_field_element_with_da_word(entry.value, "value was this".to_string());
        }
    }

    // Handle nonces
    // for (addr, nonce) in nonces {
    //     blob_data.push(addr);

    //     let class_flag = deployed_contracts
    //         .get(&addr)
    //         .or_else(|| replaced_classes.get(&addr));

    //     blob_data.push(da_word(class_flag.is_some(), Some(nonce), 0_u64));
    //     if let Some(class_hash) = class_flag {
    //         blob_data.push(*class_hash);
    //     }
    // }

    // // Handle deployed contracts
    // for (addr, class_hash) in deployed_contracts {
    //     blob_data.push(addr);

    //     blob_data.push(da_word(true, None, 0_u64));
    //     blob_data.push(class_hash);
    // }

    // Handle declared classes
    blob_data.push(FieldElement::from(declared_classes.len()));

    for (class_hash, compiled_class_hash) in &declared_classes {
        blob_data.push(*class_hash);
        blob_data.push(*compiled_class_hash);
    }

    blob_data
}

fn da_word(class_flag: bool, nonce_change: Option<FieldElement>, num_changes: u64) -> FieldElement {
    let something_to_test = encdoe_da_word(class_flag, nonce_change, num_changes);
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

    something_to_test
}

fn encdoe_da_word(
    class_flag: bool,
    nonce_change: Option<FieldElement>,
    num_changes: u64,
) -> FieldElement {
    let mut binary_string = "0".repeat(127);

    if (class_flag) {
        binary_string += "1"
    } else {
        binary_string += "0"
    }

    assert_eq!(
        binary_string.len(),
        128,
        "The length of binary_string is not 128."
    );

    if let Some(new_nonce) = nonce_change {
        // word += new_nonce + FieldElement::from_hex_be(NONCE_BASE).unwrap();
        let bytes: [u8; 32] = nonce_change.unwrap().to_bytes_be();
        let biguint = BigUint::from_bytes_be(&bytes);
        let binary_string_local = format!("{:b}", biguint);
        let padded_binary_string = format!("{:0>64}", binary_string_local);
        binary_string += &padded_binary_string;
    } else {
        let mut binary_string_local = "0".repeat(64);
        binary_string += &binary_string_local;
    }

    assert_eq!(
        binary_string.len(),
        192,
        "The length of binary_string is not 192."
    );

    let binary_representation = format!("{:b}", num_changes);
    let padded_binary_string = format!("{:0>64}", binary_representation);
    binary_string += &padded_binary_string;

    assert_eq!(
        binary_string.len(),
        256,
        "The length of binary_string is not 256."
    );

    let biguint =
        BigUint::from_str_radix(binary_string.as_str(), 2).expect("Invalid binary string");

    // Now convert the BigUint to a decimal string
    let decimal_string = biguint.to_str_radix(10);

    let word = FieldElement::from_dec_str(&decimal_string)
        .expect("issue while converting to fieldElement");

    word
}

fn decode_da_word(word: FieldElement) -> (bool, u64, u64) {
    let bytes = word.to_bytes_be();

    // Extract the class flag (1 bit)
    let class_flag = (bytes[0] & 0b10000000) != 0;

    // Extract the new nonce (64 bits)
    let mut nonce_bytes = [0u8; 8];
    nonce_bytes.copy_from_slice(&bytes[16..24]);
    let new_nonce = u64::from_be_bytes(nonce_bytes);

    // Extract the number of changes (64 bits)
    let mut num_changes_bytes = [0u8; 8];
    num_changes_bytes.copy_from_slice(&bytes[24..32]);
    let num_changes = u64::from_be_bytes(num_changes_bytes);

    (class_flag, new_nonce, num_changes)
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

fn compare_the_field_element_with_da_word(element: FieldElement, location: String) {
    let field_element_here = FieldElement::from_dec_str(
        "853719037260241292551246424765742504283484368995629135147152722137610948318",
    )
    .expect("issue while comparing the fieldElement");

    if (field_element_here == element) {
        println!("so the element is same at {:?}", location);
    }
}
#[tokio::main]
async fn main() -> Result<()> {
    send_Tx_4844(638353).await?;
    Ok(())
}

// impl PartialEq for StorageUpdate {
//     fn eq(&self, other: &Self) -> bool {
//         self.key == other.key && self.value == other.value
//     }
// }

// impl Eq for StorageUpdate {}

// impl ContractUpdate {
//     // Helper function to create a hash set from storage_updates
//     fn storage_updates_set(&self) -> HashSet<StorageUpdate> {
//         self.storage_updates.iter().cloned().collect()
//     }
// }

// impl PartialEq for ContractUpdate {
//     fn eq(&self, other: &Self) -> bool {
//         self.address == other.address
//             && self.nonce == other.nonce
//             && self.number_of_storage_updates == other.number_of_storage_updates
//             && self.new_class_hash == other.new_class_hash
//             && self.storage_updates_set() == other.storage_updates_set()
//     }
// }

// impl Eq for ContractUpdate {}

// // Function to compare two vectors of ContractUpdate
// fn are_vectors_equal(vec1: &[ContractUpdate], vec2: &[ContractUpdate]) -> bool {
//     if vec1.len() != vec2.len() {
//         return false;
//     }

//     let set1: HashSet<_> = vec1.iter().collect();
//     let set2: HashSet<_> = vec2.iter().collect();

//     set1 == set2
// }
