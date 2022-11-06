use clap::{Parser, ValueEnum};
use logic_simulator::blueprint::{VcbBoard, VcbParser};
use std::fs::read_to_string;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum RunMode {
    /// Print initial state of board
    Print,
    /// Run and display state of board
    Run,
    /// Run a number of iterations and print time (make sure to compile with `--release` and lto)
    Bench,
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about,
    long_about = "Logic simulator, currently the old VCB blueprints are implemented."
)]
struct Args {
    /// Filepath to (old) VCB blueprint string
    #[arg(long, group = "blueprint")]
    blueprint_file: Option<PathBuf>,

    /// VCB blueprint string
    #[arg(long, group = "blueprint")]
    blueprint_string: Option<String>,

    /// What mode to run the program in
    #[arg(value_enum, requires = "blueprint")]
    mode: RunMode,

    /// Iterations to run in bench
    #[arg(short, long, default_value_t = 10_000_000)]
    iterations: usize,
}

fn main() {
    let args = Args::parse();

    println!("File {:?}!", args.blueprint_file);
    println!("String {:?}!", args.blueprint_string);
    println!("Mode {:?}!", args.mode);

    let string: String = match args.blueprint_string {
        Some(s) => s,
        None => read_to_string(args.blueprint_file.unwrap()).expect("File should exist"),
    };

    //let string = include_str!("../test_files/big_decoder.blueprint");
    //let string = include_str!("../test_files/intro.blueprint");
    //let string = include_str!("../test_files/gates.blueprint");
    //let string = include_str!("../test_files/small_decoder.blueprint");
    //let string = include_str!("../test_files/circle.blueprint");
    //let string = include_str!("../test_files/gates.blueprint");
    //let string = include_str!("../test_files/invalid_base64.blueprint");
    //let string = include_str!("../test_files/invalid_zstd.blueprint");

    let mut board: VcbBoard<1> = VcbParser::parse(&string, true);

    match args.mode {
        RunMode::Print => board.print(),
        RunMode::Run => loop {
            board.print();
            board.update();
            std::thread::sleep(Duration::from_millis(500));
        },
        RunMode::Bench => {
            let iterations = args.iterations;
            let iterations_f32: f32 = iterations as f32;
            println!("Running {iterations} iterations");
            let now = std::time::Instant::now();
            (0..iterations).for_each(|_| {
                board.update();
            });
            let elapsed = now.elapsed().as_millis() as f32 / 1000.0;
            let millisecond_per_iteration = elapsed / iterations_f32 * 1000.0;
            let microsecond_per_iteration = millisecond_per_iteration * 1000.0;
            let nanosecond_per_iteration = microsecond_per_iteration * 1000.0;
            let iterations_per_second = iterations_f32 / elapsed;
            println!("Elapsed: {elapsed} seconds");
            println!("ms/iteration: {millisecond_per_iteration}");
            println!("μs/iteration: {microsecond_per_iteration}");
            println!("ns/iteration: {nanosecond_per_iteration}");
            println!(
                "iteration/s: {} million",
                iterations_per_second / 1_000_000.0
            );
        },
    }
}
