mod model;
mod tensor;
mod tokenizer;
mod training;
mod utils;

use crate::training::trainer::{Dataset, Trainer};
use std::env;
use std::error::Error;
use std::io;
use std::io::ErrorKind;

#[derive(Debug)]
struct CliArgs {
    data_path: Option<String>,
    block_size: usize,
    show_help: bool,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let bin_name = args.first().map(String::as_str).unwrap_or("rust-gpt");
    let cli = parse_cli_args(&args)?;

    if cli.show_help {
        print_usage(bin_name);
        return Ok(());
    }

    let Some(data_path) = cli.data_path else {
        println!("RustGPT bootstrap ready");
        print_usage(bin_name);
        return Ok(());
    };

    let dataset = Dataset::from_file(&data_path)?;
    let trainer = Trainer::new(cli.block_size)?;
    let samples = trainer.build_samples(&dataset)?;

    let preview_tokens = dataset.token_count().min(120);
    let preview = dataset.decode_prefix(preview_tokens)?;

    println!("Loaded dataset: {data_path}");
    println!("chars: {}", dataset.char_count());
    println!("tokens: {}", dataset.token_count());
    println!("vocab size: {}", dataset.vocab_size());
    println!("block size: {}", trainer.block_size());
    println!("training samples: {}", samples.len());

    println!("\nTokenizer info:");
    println!("bos id: {}", dataset.tokenizer().bos_id());
    println!("bos token: {}", dataset.tokenizer().bos_token());

    println!("\nPreview (first {preview_tokens} tokens decoded):");
    println!("{preview}");

    Ok(())
}

fn parse_cli_args(args: &[String]) -> Result<CliArgs, io::Error> {
    let mut data_path: Option<String> = None;
    let mut block_size = 64usize;
    let mut show_help = false;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                show_help = true;
                i += 1;
            }
            "--data" => {
                let value = args.get(i + 1).ok_or_else(|| {
                    io::Error::new(ErrorKind::InvalidInput, "--data requires a file path")
                })?;
                data_path = Some(value.clone());
                i += 2;
            }
            "--block-size" => {
                let value = args.get(i + 1).ok_or_else(|| {
                    io::Error::new(ErrorKind::InvalidInput, "--block-size requires a number")
                })?;
                block_size = value.parse::<usize>().map_err(|_| {
                    io::Error::new(
                        ErrorKind::InvalidInput,
                        "--block-size must be a valid usize",
                    )
                })?;
                i += 2;
            }
            flag => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("unknown argument: {flag}"),
                ));
            }
        }
    }

    Ok(CliArgs {
        data_path,
        block_size,
        show_help,
    })
}

fn print_usage(bin_name: &str) {
    println!("Usage:");
    println!("  {bin_name} --data <path/to/corpus.txt> [--block-size <N>]");
    println!("  {bin_name} --help");
}
