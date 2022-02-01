# Zettel 1

## Install the Rust toolchain

The easiest way to get Rust is via the `rustup` command line tool:
 1. To install rustup (as described in `https://www.rust-lang.org/learn/get-started`) run `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` inside your terminal.
 2. Install the stable toolchain by running `rustup install stable`


## Build and run
`cd` to the project directory (the one containing `Cargo.toml`) and run `cargo run --release`.
Cargo will now pull the necessary dependencies, build everything and run it.