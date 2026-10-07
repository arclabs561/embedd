use embedd::vector::{cosine_f32, dot_f32, l2_norm, l2_normalize_in_place};

#[test]
fn public_vector_contract() {
    assert_eq!(dot_f32(&[], &[]), 0.0);
    assert_eq!(cosine_f32(&[], &[]), 0.0);
    assert_eq!(dot_f32(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
    assert_eq!(cosine_f32(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    assert_eq!(cosine_f32(&[1.0], &[-1.0]), -1.0);
    assert_eq!(cosine_f32(&[0.0], &[1.0]), 0.0);
    assert_eq!(cosine_f32(&[f32::NAN], &[1.0]), 0.0);
    assert!(cosine_f32(&[f32::INFINITY], &[1.0]).is_nan());
    assert!(std::panic::catch_unwind(|| dot_f32(&[1.0], &[])).is_err());
    assert!(std::panic::catch_unwind(|| cosine_f32(&[1.0], &[])).is_err());
    let mut tiny = [1e-10];
    assert_eq!(l2_normalize_in_place(&mut tiny), 0.0);
    assert_eq!(tiny, [1e-10]);
    let mut nan = [f32::NAN];
    assert!(l2_normalize_in_place(&mut nan).is_nan());
    assert!(nan[0].is_nan());
}

#[test]
fn finite_vectors_agree_with_independent_f64_oracle() {
    for n in [1, 3, 4, 7, 16, 31, 32, 63, 64, 127, 128, 384, 768, 1536] {
        let a: Vec<f32> = (0..n)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 31.0)
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) / 29.0)
            .collect();
        let ab: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let aa: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum();
        let bb: f64 = b.iter().map(|&x| f64::from(x).powi(2)).sum();
        let magnitude: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| (f64::from(x) * f64::from(y)).abs())
            .sum();
        let tol = f64::from(f32::EPSILON) * n as f64 * magnitude.max(1.0);
        assert!((f64::from(dot_f32(&a, &b)) - ab).abs() <= tol);
        assert!((f64::from(l2_norm(&a)) - aa.sqrt()).abs() < 1e-5 * aa.sqrt());
        assert!((f64::from(cosine_f32(&a, &b)) - ab / (aa * bb).sqrt()).abs() < 1e-5);
    }
}

#[test]
fn exceptional_values_in_simd_blocks_and_tails() {
    // 32/128 enter the runtime SIMD paths; +1 also exercises scalar tails.
    for n in [32, 33, 128, 129] {
        for position in [0, n / 2, n - 1] {
            let mut a = vec![0.0; n];
            let mut b = vec![0.0; n];
            b[position] = 1.0;
            assert_eq!(cosine_f32(&a, &b), 0.0);
            for tiny in [0.0, 1e-10] {
                a[position] = tiny;
                let original = a.clone();
                assert_eq!(l2_normalize_in_place(&mut a), 0.0);
                assert_eq!(a, original);
                assert_eq!(cosine_f32(&a, &b), 0.0);
            }
            a[position] = f32::NAN;
            assert!(dot_f32(&a, &b).is_nan());
            assert!(l2_norm(&a).is_nan());
            assert_eq!(cosine_f32(&a, &b), 0.0);
            assert!(l2_normalize_in_place(&mut a).is_nan());
            assert!(a.iter().all(|x| x.is_nan()));
            a.fill(0.0);
            a[position] = f32::INFINITY;
            assert_eq!(l2_norm(&a), f32::INFINITY);
            assert!(cosine_f32(&a, &b).is_nan());
            assert_eq!(l2_normalize_in_place(&mut a), f32::INFINITY);
            assert!(a[position].is_nan());
        }
    }
}

#[test]
fn adjacent_f32_values_at_normalization_threshold() {
    let epsilon = 1e-9_f32;
    let below = f32::from_bits(epsilon.to_bits() - 1);
    let above = f32::from_bits(epsilon.to_bits() + 1);
    for n in [1, 32, 33, 128, 129] {
        for position in [0, n - 1] {
            for value in [below, epsilon, above] {
                let mut a = vec![0.0; n];
                let mut b = vec![0.0; n];
                a[position] = value;
                b[position] = 1.0;
                if value <= epsilon {
                    assert_eq!(cosine_f32(&a, &b), 0.0);
                    assert_eq!(l2_normalize_in_place(&mut a), 0.0);
                    assert_eq!(a[position], value);
                } else {
                    assert!((cosine_f32(&a, &b) - 1.0).abs() < 1e-6);
                    assert!((l2_normalize_in_place(&mut a) - value).abs() <= f32::EPSILON * value);
                    assert!((a[position] - 1.0).abs() < 1e-6);
                }
            }
        }
    }
}
