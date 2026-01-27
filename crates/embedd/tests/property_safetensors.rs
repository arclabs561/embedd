use proptest::prelude::*;

#[test]
fn safetensors_rejects_hole_example() {
    // One tensor covering 0..4, but data_len claims 8.
    let header = br#"{"t":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
    let err = embedd::safetensors::validate_header_and_data_len(header, 8).unwrap_err();
    assert!(err.to_string().contains("does not fully cover buffer"));
}

proptest! {
    #[test]
    fn safetensors_accepts_contiguous_layout(
        sizes in prop::collection::vec(0usize..64, 1..16),
    ) {
        // Build a contiguous layout with dtype=U8 and shape=[size].
        let mut offset = 0usize;
        let mut entries = Vec::new();
        for (i, sz) in sizes.iter().enumerate() {
            let begin = offset;
            let end = offset + *sz;
            offset = end;
            entries.push(format!(
                "\"t{i}\":{{\"dtype\":\"U8\",\"shape\":[{sz}],\"data_offsets\":[{begin},{end}]}}"
            ));
        }
        let json = format!("{{{}}}", entries.join(","));
        embedd::safetensors::validate_header_and_data_len(json.as_bytes(), offset).unwrap();
    }

    #[test]
    fn safetensors_rejects_unknown_dtype(
        data_len in 0usize..256,
    ) {
        // One tensor with unknown dtype must fail regardless of offsets (as long as offsets are in range).
        let end = data_len.min(8);
        let header = format!(
            "{{\"t\":{{\"dtype\":\"NOPE\",\"shape\":[0],\"data_offsets\":[0,{end}]}}}}"
        );
        let err = embedd::safetensors::validate_header_and_data_len(header.as_bytes(), data_len).unwrap_err();
        prop_assert!(err.to_string().contains("unknown dtype"));
    }
}
