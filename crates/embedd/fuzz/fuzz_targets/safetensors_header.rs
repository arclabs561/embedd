#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzz target: safetensors header validation.
//
// Invariant: this function must never panic, regardless of input bytes.
fuzz_target!(|data: &[u8]| {
    // Interpret the first few bytes as a small-ish data_len to keep the fuzzer fast.
    let data_len = data.get(0).copied().unwrap_or(0) as usize;
    let header = data.get(1..).unwrap_or(&[]);
    let _ = embedd::safetensors::validate_header_and_data_len(header, data_len);
});
