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
use std::ops::{Add, Mul, Rem};

mod constant;
mod local_serde;
mod local_state_diff;

use constant::GENERATOR;
use rayon::prelude::*;

// use eyre::Result;

pub const BLOB_LEN: usize = 4096;
lazy_static! {
    pub static ref TWO: BigUint = 2u32.to_biguint().unwrap();
    pub static ref ONE: BigUint = 1u32.to_biguint().unwrap();
}
// 239662865362034240643029484020826104800516253213877576945698001454992152761
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
    let BLS_MODULUS: BigUint = BigUint::from_str(
        "52435875175126190479447740508185965837690552500527637822603658699938581184513",
    )
    .unwrap();
    let xs: Vec<BigUint> = (0..BLOB_LEN)
        .map(|i| {
            let bin = format!("{:012b}", i);
            let bin_rev = bin.chars().rev().collect::<String>();
            GENERATOR.modpow(&BigUint::from_str_radix(&bin_rev, 2).unwrap(), &BLS_MODULUS)
        })
        .collect();

    let write_to_file_v0 =
        write_biguint_to_file(&xs, format!("eval_points_{}.txt", block_no).as_str())
            .expect("issue while writing it to the file");

    let data_after_ntt = some_algo_optimized_V2(biguint_vec.clone(), xs, &BLS_MODULUS);

    let write_to_file_v1 = write_biguint_to_file(
        &data_after_ntt,
        format!("state_file_after_ntt_from_block_{}.txt", block_no).as_str(),
    )
    .expect("issue while writing it to the file");

    let write_to_file_v2 = write_biguint_to_file(
        &biguint_vec,
        format!("state_file_from_block_{}.txt", block_no).as_str(),
    )
    .expect("issue while writing it to the file");

    // let self_state_diffs = local_serde::parse_state_diffs(biguint_vec.as_slice());
    // let self_state_diffs_json = local_serde::to_json(self_state_diffs.clone());
    // fs::write(
    //     format!("parsed_state_file_from_block_{}.txt", block_no).as_str(),
    //     self_state_diffs_json,
    // )?;

    // let blob_data =
    //     local_serde::parse_file_to_blob_data(format!("./test_blob_ {}.txt", block_no).as_str());

    // // Recover the original data
    // let original_data = blob::recover(blob_data);

    // let write_to_file = write_biguint_to_file(
    //     &original_data,
    //     format!("state_file_from_blob_{}.txt", block_no).as_str(),
    // )
    // .expect("issue while writing it to the file");
    // let state_diffs = local_serde::parse_state_diffs(original_data.as_slice());
    // let state_diffs_json = local_serde::to_json(state_diffs.clone());
    // fs::write(
    //     format!("parsed_state_file_from_blob_{}.txt", block_no).as_str(),
    //     state_diffs_json,
    // )?;
    // assert!(
    //     self_state_diffs.class_declaration_size == state_diffs.class_declaration_size,
    //     "class declaration should be equal"
    // );
    // assert!(
    //     self_state_diffs.state_update_size == state_diffs.state_update_size,
    //     "state diff should be equal should be equal"
    // );

    // assert!(
    //     have_identical_class_declarations(&state_diffs, &self_state_diffs),
    //     "value of class declaration should be identical"
    // );

    // assert!(
    //     state_diffs.has_same_contract_updates(&self_state_diffs),
    //     "state diff values should match as well"
    // );

    Ok(())
}
fn some_algo_optimized(arr: Vec<BigUint>, xs: Vec<BigUint>, p: &BigUint) -> Vec<BigUint> {
    let n = arr.len();
    let mut transform = vec![Zero::zero(); n];

    for i in 0..n {
        let mut xi_pow_j: BigUint = ONE.clone(); // Initialize to xs[i]**0
        println!("on i we are here: {:?}", i);
        for j in 0..n {
            let temp = &arr[j] * &xi_pow_j;
            transform[i] += &temp % p;
            xi_pow_j = (&xi_pow_j * &xs[i]) % p; // Update power for next iteration
        }
        transform[i] %= p; // Apply modulo once per outer loop
    }

    transform
}

fn some_algo_optimized_V2(arr: Vec<BigUint>, xs: Vec<BigUint>, p: &BigUint) -> Vec<BigUint> {
    let n = arr.len();
    let mut transform: Vec<BigUint> = vec![BigUint::zero(); n];

    for i in 0..n {
        println!("on i we are here: {:?}", i);
        // let mut xi_pow_j: BigUint = One::one(); // Initialize to xs[i]**0
        for j in (0..n).rev() {
            transform[i] = (transform[i].clone().mul(&xs[i]).add(&arr[j])).rem(p);
        }
    }

    transform
}
// copilot version 1
// fn ntt(arr: Vec<BigUint>, xs: Vec<BigUint>, p: BigUint) -> Vec<BigUint> {
//     let n = arr.len();
//     println!("the size here is: {:?}", n);
//     let mut transform = vec![BigUint::zero(); n];

//     for i in 0..n {
//         println!("on i we are here: {:?}", i);
//         let mut sum = BigUint::zero();
//         for j in 0..n {
//             // println!("on j we are here: {:?}", j);
//             let exponent = xs[i].modpow(&BigUint::from(j), &p);
//             sum = (sum.add(arr[j].clone().mul(exponent))).rem(&p);
//         }
//         transform[i] = sum;
//     }

//     transform
// }

// fn fft(mut a: Vec<BigUint>, points: Vec<BigUint>, p: BigUint) -> Vec<BigUint> {
//     let n = a.len();

//     // Handle base case (n = 1)
//     if n == 1 {
//         return a;
//     }

//     // Split into even and odd elements using indexing
//     let even = a.to_vec(); // Collect even-indexed elements
//     let odd = a[1..].iter().step(2).cloned().collect::<Vec<_>>(); // Collect odd-indexed elements, skipping the first element

//     // Recursively compute FFTs for even and odd subarrays
//     let even_fft = fft(even, points.clone(), p.clone());
//     let odd_fft = fft(odd, points.clone(), p.clone());

//     // Combine results using point-value evaluation
//     let mut result = vec![BigUint::from(0); n];
//     for k in 0..n / 2 {
//         let mut term1 = even_fft[k].clone();
//         let mut term2 = odd_fft[k].clone();
//         for i in 0..n / 2 {
//             let mut weight = points[i].clone() * points[k + n / 2].clone() % p.clone();
//             weight = weight.pow(modulus(i, n as u64)) % p.clone(); // Efficiently calculate i^k (modulo p)

//             term1 = (term1 + weight * odd_fft[i].clone()) % p.clone();
//             term2 = (term2 - weight * even_fft[i].clone() + p.clone()) % p.clone();
//         }
//         result[k] = term1;
//         result[k + n / 2] = term2;
//     }

//     result
// }

fn modulus(a: u64, b: u64) -> u64 {
    (a % b + b) % b // Efficient modulo operation
}

// pub fn fft(arr: Vec<BigUint>, xs: Vec<BigUint>, p: &BigUint) -> Vec<BigUint> {
//     // Leverage the IFFT function with a twist: use ONE as denominator for inverses
//     ifft(
//         arr,
//         xs.iter()
//             .map(|x| x.modpow(&(TWO.clone() - ONE.clone()), p))
//             .collect(),
//         p,
//     )
// }

pub fn ifft(arr: Vec<BigUint>, xs: Vec<BigUint>, p: &BigUint) -> Vec<BigUint> {
    // Base case: return immediately if the array length is 1
    if arr.len() == 1 {
        return arr;
    }

    let n = arr.len() / 2;
    let mut res0 = Vec::with_capacity(n);
    let mut res1 = Vec::with_capacity(n);
    let mut new_xs = Vec::with_capacity(n);

    for i in (0..2 * n).step_by(2) {
        let a = &arr[i];
        let b = &arr[i + 1];
        let x = &xs[i];

        res0.push(div_mod(a + b, TWO.clone(), p));
        // Handle subtraction to avoid underflow
        let diff = if b > a { p - (b - a) } else { a - b };
        res1.push(div_mod(diff, TWO.clone() * x, p));

        new_xs.push(x.modpow(&TWO.clone(), p));
    }

    // Recursive calls
    let merged_res0 = ifft(res0, new_xs.clone(), p);
    let merged_res1 = ifft(res1, new_xs, p);

    // Merging the results
    let mut merged = Vec::with_capacity(arr.len());
    for i in 0..n {
        merged.push(merged_res0[i].clone());
        merged.push(merged_res1[i].clone());
    }
    merged
}

pub fn div_mod(a: BigUint, b: BigUint, p: &BigUint) -> BigUint {
    a * b.modpow(&(p - TWO.clone()), p) % p
}

fn ntt(arr: Vec<BigUint>, xs: Vec<BigUint>, p: &BigUint) -> Vec<BigUint> {
    let n = arr.len();
    let mut transform: Vec<BigUint> = vec![BigUint::zero(); n];

    for i in 0..n {
        let mut sum = BigUint::zero();
        println!("on i we are here: {:?}", i);
        for j in 0..n {
            let term = (arr[j].clone() * xs[i].modpow(&BigUint::from(j), &p.clone())) % p;
            sum = (sum + term) % p.clone();
        }
        transform[i] = sum;
    }

    transform
}
// fn ntt(arr: Vec<BigUint>, xs: Vec<BigUint>, p: BigUint) -> Vec<BigUint> {
//     let n = arr.len();
//     let mut transform = vec![BigUint::zero(); n];

//     transform.par_iter_mut().enumerate().for_each(|(i, t)| {
//         *t = (0..n)
//             .map(|j| {
//                 println!("on i we are here: {:?}", i);
//                 let exponent = xs[i].modpow(&BigUint::from(j), &p);
//                 arr[j].clone() * exponent % &p
//             })
//             .fold(BigUint::zero(), |acc, val| acc + val)
//             % &p;
//     });

//     transform
// }
// fn ntt_recursive(
//     arr: &Vec<BigUint>,
//     xs: &Vec<BigUint>,
//     p: &BigUint,
//     wn: &[BigUint],
// ) -> Vec<BigUint> {
//     let n = arr.len();

//     if n == 1 {
//         return vec![arr[0].clone()];
//     }

//     // Bit reversal for butterfly algorithm
//     let mut reversed_arr = vec![BigUint::zero(); n];
//     for (i, x) in arr.iter().enumerate() {
//         let mut reversed_index = 0;
//         for j in 0..(n.bits() - 1) {
//             reversed_index |= (i >> j & 1) << (n.bits() - 1 - j);
//         }
//         reversed_arr[reversed_index] = x.clone();
//     }

//     // Divide and conquer
//     let half = n / 2;
//     let even_terms = ntt_recursive(&reversed_arr[..half], &xs[..half], &p, &wn);
//     let odd_terms = ntt_recursive(&reversed_arr[half..], &xs[half..], p, wn);

//     // Combine using butterfly algorithm (consider Montgomery multiplication here)
//     let mut transform: Vec<BigUint> = vec![BigUint::zero(); n];
//     for i in 0..half {
//         let mut u = even_terms[i].clone();
//         let mut v = (&odd_terms[i] * &wn[i]) % p; // Consider Montgomery multiplication
//         transform[i] = (u + v) % p;
//         transform[i + half] = (u - v + p) % p; // Utilize modular addition properties
//     }

//     transform
// }

// pub fn ntt(arr: &Vec<BigUint>, xs: &Vec<BigUint>, p: &BigUint) -> Vec<BigUint> {
//     // Precompute w^n (primitive root powers) for butterfly algorithm
//     let mut wn: Vec<BigUint> = vec![BigUint::one(); arr.len()];
//     let mut root = xs[0].clone();
//     let inv_two = BigUint::from_u64(2).modpow(p.mod_inverse(2u64), p);
//     for i in 1..arr.len() {
//         wn[i] = (root * &inv_two) % p;
//         root = (root * &wn[i]) % p;
//     }

//     ntt_recursive(arr, xs, p, &wn)
// }
impl ContractUpdate {
    // Helper function to create a key for sorting
    fn sort_key(&self) -> BigUint {
        self.address.clone()
    }

    fn has_same_storage_updates(&self, other: &ContractUpdate) -> bool {
        let mut self_storage = self.storage_updates.clone();
        let mut other_storage = other.storage_updates.clone();

        // Sort the storage updates by the unique key
        self_storage.sort_by_key(|update| update.sort_key_storage());
        other_storage.sort_by_key(|update| update.sort_key_storage());

        if self_storage.len() != other_storage.len() {
            return false;
        }

        self_storage
            .iter()
            .zip(other_storage.iter())
            .all(|(self_update, other_update)| self_update == other_update)
    }
}

impl StorageUpdate {
    // Helper function to create a key for sorting
    fn sort_key_storage(&self) -> BigUint {
        self.key.clone()
    }
}
impl PartialEq for ContractUpdate {
    fn eq(&self, other: &Self) -> bool {
        println!(
            "so this is the data in eq, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}",
            self.address,
            other.address,
            self.new_class_hash,
            other.new_class_hash,
            self.number_of_storage_updates,
            other.number_of_storage_updates,
            self.nonce,
            other.nonce
        );

        if (self.new_class_hash != other.new_class_hash) {
            println!("class hash is not mathcing here");
        }

        if (self.nonce != other.nonce) {
            println!("nonce is not mathcing here");
        }

        self.address == other.address
            && self.nonce == other.nonce
            && self.number_of_storage_updates == other.number_of_storage_updates
            && self.new_class_hash == other.new_class_hash
            && self.has_same_storage_updates(other)
        // storage_updates is not compared
    }
}

impl DataJson {
    pub fn has_same_contract_updates(&self, other: &DataJson) -> bool {
        let mut self_updates = self.state_update.clone();
        let mut other_updates = other.state_update.clone();

        // Sort the updates by the unique identifier (address)
        self_updates.sort_by_key(|update| update.sort_key());
        other_updates.sort_by_key(|update| update.sort_key());

        if self_updates.len() != other_updates.len() {
            return false;
        }

        for (update_self, update_other) in self_updates.iter().zip(other_updates.iter()) {
            if update_self != update_other {
                return false;
            }
        }

        true
    }
}
pub fn get_env_var(key: &str) -> Result<String> {
    dotenv().ok();
    let variable_here = env::var(key).expect("PK must be set");
    Ok(variable_here)
    // std::env::var(key).map_err(|e| e.into())
}

pub fn have_identical_class_declarations(a: &DataJson, b: &DataJson) -> bool {
    let set_a: HashSet<_> = a.class_declaration.iter().cloned().collect();
    let set_b: HashSet<_> = b.class_declaration.iter().cloned().collect();

    set_a == set_b
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
        // if (addr
        //     == FieldElement::from_dec_str(
        //         "239662865362034240643029484020826104800516253213877576945698001454992152761",
        //     )
        //     .unwrap())
        // {
        //     println!("we got our target, {:?}", nonce);
        // }
        let da_word_here = da_word(class_flag.is_some(), nonce, writes.len() as u64);
        let (class_flag_here, nonce_here, change_here) = decode_da_word(da_word_here);
        if (da_word_here
            == FieldElement::from_dec_str("340282366920938464846880412959984582656").unwrap())
        {
            println!(
                "all the data here is {:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?}, ",
                addr,
                class_flag,
                nonce,
                writes.len(),
                da_word_here,
                class_flag_here,
                nonce_here,
                change_here
            );
        }
        //all the data here is None,None,3,FieldElement { inner: 0x0000000000000000000000000000000000000000000000000000000000000003 },false,0,3,
        //all the data here is None,Some(FieldElement { inner: 0x0000000000000000000000000000000000000000000000000000000000000200 }),0,FieldElement { inner: 0x00000000000000000000000000000000000000000000000100000000000001ff },false,1,511,
        // all the data here is None,Some(FieldElement { inner: 0x00000000000000000000000000000000000000000000000000000000000004e2 }),0,FieldElement { inner: 0x00000000000000000000000000000000000000000000000100000000000004e1 },false,1,1249,

        if (addr == FieldElement::ONE && da_word_here == FieldElement::ONE) {
            continue;
        }
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
    send_Tx_4844(640641).await?;
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
