use super::*;
use crate::config::PowerConfig;
#[test]
fn xolotl_inline_acl_is_explicit_bounded_and_uses_the_native_schema() {
    let parsed =
        Cli::try_parse_from(["a3s-power", "serve", "--config-inline", "port = 18107"]).unwrap();
    let Some(Command::Serve(args)) = parsed.command else {
        panic!("serve")
    };
    assert!(args.config.is_none());
    let config = PowerConfig::from_acl(args.config_inline.as_deref().unwrap()).unwrap();
    assert_eq!(config.port, 18107);
    assert!(Cli::try_parse_from([
        "a3s-power",
        "serve",
        "--config",
        "a.acl",
        "--config-inline",
        "port = 18107"
    ])
    .is_err());
    assert!(PowerConfig::from_acl("unregistered_runtime_setting = true").is_err());
    let error = PowerConfig::from_acl(&" ".repeat(65537))
        .unwrap_err()
        .to_string();
    assert!(error.contains("64 KiB byte bound"));
}

#[test]
fn workload_acl_environment_is_an_explicit_exclusive_input() {
    let parsed = Cli::try_parse_from(["a3s-power", "serve", "--config-from-workload-env"]).unwrap();
    let Some(Command::Serve(args)) = parsed.command else {
        panic!("serve")
    };
    assert!(args.config_from_workload_env);
    for other in ["--config", "--config-inline"] {
        assert!(Cli::try_parse_from([
            "a3s-power",
            "serve",
            "--config-from-workload-env",
            other,
            "port = 18107"
        ])
        .is_err());
    }
}
