use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// The `embedd` binary exists only when `--features cli` is enabled.
#[cfg(feature = "cli")]
#[test]
fn embedd_help_is_self_describing() {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("embedd"));
    cmd.arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("embedd"))
        .stdout(predicate::str::contains("EXAMPLES"));
}
