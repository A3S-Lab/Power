use super::*;

fn host_device() -> CudaDevice {
    parse_nvidia_smi_csv("7, NVIDIA GeForce RTX 4090 D, 8.9, 24564\n")
        .unwrap()
        .remove(0)
}

#[test]
fn real_host_4090d_smi_inventory_is_admitted() {
    let device = host_device();
    require_rtx_4090(7, &[device]).unwrap();
}

#[test]
fn admitted_4090d_still_requires_sm89_and_24gib() {
    let device = host_device();
    for (major, minor) in [(8, 6), (9, 0)] {
        let mut wrong = device.clone();
        wrong.compute_major = major;
        wrong.compute_minor = minor;
        assert!(require_rtx_4090(7, &[wrong])
            .unwrap_err()
            .to_string()
            .contains("not 8.9"));
    }
    for mib in [16_384, 48_000] {
        let mut wrong = device.clone();
        wrong.memory_bytes = mib * 1024 * 1024;
        assert!(require_rtx_4090(7, &[wrong])
            .unwrap_err()
            .to_string()
            .contains("MiB"));
    }
}

#[test]
fn product_suffixes_and_arbitrary_prefixes_are_not_inferred() {
    for name in [
        "NVIDIA GeForce RTX 4090 Ti",
        "NVIDIA GeForce RTX 4090 Laptop",
        "NVIDIA GeForce RTX 4090 D Laptop",
        "NVIDIA GeForce RTX 4090 D2",
        "Other GeForce RTX 4090",
    ] {
        let mut device = host_device();
        device.name = name.into();
        assert!(require_rtx_4090(7, &[device]).is_err(), "{name}");
    }
}
