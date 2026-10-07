#[cfg(feature = "simd")]
use innr::{cosine, dot, norm};

// Safe portable reductions adapted from innr 0.4.0 src/dense.rs.
// Copyright (c) 2024 arclabs561. MIT OR Apache-2.0; see repository licenses.
// https://docs.rs/crate/innr/0.4.0/source/src/dense.rs
#[cfg(not(feature = "simd"))]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector slice length mismatch");
    let n = a.len();
    let chunks = n / 4;

    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
    }

    let mut result = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..n {
        result += a[i] * b[i];
    }
    result
}

#[cfg(not(feature = "simd"))]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector slice length mismatch");
    let n = a.len();
    let chunks = n / 4;

    let mut ab0 = 0.0f32;
    let mut ab1 = 0.0f32;
    let mut ab2 = 0.0f32;
    let mut ab3 = 0.0f32;
    let mut aa0 = 0.0f32;
    let mut aa1 = 0.0f32;
    let mut aa2 = 0.0f32;
    let mut aa3 = 0.0f32;
    let mut bb0 = 0.0f32;
    let mut bb1 = 0.0f32;
    let mut bb2 = 0.0f32;
    let mut bb3 = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        let a0 = a[base];
        let b0 = b[base];
        let a1 = a[base + 1];
        let b1 = b[base + 1];
        let a2 = a[base + 2];
        let b2 = b[base + 2];
        let a3 = a[base + 3];
        let b3 = b[base + 3];
        ab0 += a0 * b0;
        aa0 += a0 * a0;
        bb0 += b0 * b0;
        ab1 += a1 * b1;
        aa1 += a1 * a1;
        bb1 += b1 * b1;
        ab2 += a2 * b2;
        aa2 += a2 * a2;
        bb2 += b2 * b2;
        ab3 += a3 * b3;
        aa3 += a3 * a3;
        bb3 += b3 * b3;
    }

    let mut ab = ab0 + ab1 + ab2 + ab3;
    let mut aa = aa0 + aa1 + aa2 + aa3;
    let mut bb = bb0 + bb1 + bb2 + bb3;

    for i in (chunks * 4)..n {
        let ai = a[i];
        let bi = b[i];
        ab += ai * bi;
        aa += ai * ai;
        bb += bi * bi;
    }

    if aa > (NORM_EPSILON * NORM_EPSILON) && bb > (NORM_EPSILON * NORM_EPSILON) {
        ab / (aa.sqrt() * bb.sqrt())
    } else {
        0.0
    }
}

#[cfg(not(feature = "simd"))]
fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Threshold below which a vector is considered near-zero.
const NORM_EPSILON: f32 = 1e-9;

/// Compute the L2 norm of a vector.
pub fn l2_norm(v: &[f32]) -> f32 {
    norm(v)
}

/// In-place L2 normalization.
///
/// Returns the original norm. If the vector is near-zero (<= `NORM_EPSILON`),
/// this is a no-op and returns 0.0.
pub fn l2_normalize_in_place(v: &mut [f32]) -> f32 {
    let n = norm(v);
    if n <= NORM_EPSILON {
        return 0.0;
    }
    let inv = 1.0 / n;
    for x in v.iter_mut() {
        *x *= inv;
    }
    n
}

/// Dot product.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

/// Cosine similarity (handles zero vectors).
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    cosine(a, b)
}
