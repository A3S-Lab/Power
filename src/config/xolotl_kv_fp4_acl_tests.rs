use super::*;

#[test]
fn xolotl_fp4_kv_acl_is_opt_in_and_rejects_coercion() {
    assert!(
        !deserialize("xolotl { enabled = true }\n")
            .unwrap()
            .xolotl
            .kv_fp4
    );
    let config = deserialize("xolotl { kv_fp4 = true }\n").unwrap();
    assert!(config.xolotl.kv_fp4);
    assert!(
        deserialize(&serialize(&config).unwrap())
            .unwrap()
            .xolotl
            .kv_fp4
    );
    for text in [
        "xolotl { kv_fp4 = \"true\" }",
        "xolotl { kv_fp4 = 1 }",
        "xolotl { kv_fp8 = true }",
    ] {
        assert!(deserialize(text).is_err(), "must refuse {text}");
    }
}

#[test]
fn xolotl_fp4_kv_preserves_acl_block_last_assignment_semantics() {
    // a3s-acl keeps the last assignment inside a block. This option does not
    // redefine the shared language's parser contract.
    let config = deserialize("xolotl { kv_fp4 = true kv_fp4 = false }").unwrap();
    assert!(!config.xolotl.kv_fp4);
}
