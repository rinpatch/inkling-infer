//! Compiles kernels/*.metal into one metallib with Apple's offline compiler, so shader errors
//! fail the cargo build instead of surfacing at runtime.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kdir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels");
    println!("cargo:rerun-if-changed={}", kdir.display());

    let mut airs = Vec::new();
    let mut srcs: Vec<_> = std::fs::read_dir(&kdir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "metal"))
        .collect();
    srcs.sort();
    for src in srcs {
        println!("cargo:rerun-if-changed={}", src.display());
        let air = out.join(src.file_stem().unwrap()).with_extension("air");
        let st = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-std=metal3.1",
                "-O3",
                "-ffast-math",
                "-Wall",
                "-Werror",
                "-I",
            ])
            .arg(&kdir)
            .arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&air)
            .status()
            .expect("xcrun metal");
        assert!(st.success(), "metal compile failed: {}", src.display());
        airs.push(air);
    }
    let lib = out.join("inkling.metallib");
    let st = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .args(&airs)
        .arg("-o")
        .arg(&lib)
        .status()
        .expect("xcrun metallib");
    assert!(st.success(), "metallib failed");
    println!("cargo:rustc-env=INKLING_METALLIB={}", lib.display());
}
