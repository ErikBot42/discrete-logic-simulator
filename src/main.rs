use clap::{Parser, ValueEnum};
use crossterm::execute;
use crossterm::terminal::{ClearType, EnterAlternateScreen, LeaveAlternateScreen};
use logic_simulator::blueprint::{VcbParseInput, VcbParser};
use logic_simulator::logic::UpdateStrategy;
use std::fs::read_to_string;
use std::io::stdout;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
#[deny(missing_docs)]
///
pub enum RunMode {
    /// Print board using regular emojis
    Emoji,
    /// Print board using vcb emoji
    EmojiVcb,
    /// Print initial state of board
    Print,
    /// Run and display state of board
    Run,
    /// Run a number of iterations and print time
    Bench,
    /// Copy image of board to clipboard.
    Clip,
    /// Make an animated gif of the board
    Gif,
}

#[derive(Parser, Debug)]
#[command(
//    author,
//    version,
//    about,
    long_about = None,
)]
/// Logic simulator, currently the VCB blueprints are implemented.
#[deny(missing_docs)]
pub struct Args {
    /// Filepath to VCB blueprint string
    #[arg(short = 'f', long, group = "blueprint")]
    pub blueprint_file: Option<PathBuf>,

    /// VCB blueprint string
    #[arg(short = 'b', long, group = "blueprint")]
    pub blueprint_string: Option<String>,

    /// Filepath to VCB world
    #[arg(short = 'w', long, group = "blueprint")]
    pub world_file: Option<PathBuf>,

    /// What mode to run the program in
    #[arg(value_enum, requires = "blueprint")]
    pub mode: RunMode,

    /// Print legend with emoji
    #[arg(short = 'l', long, requires = "blueprint")]
    pub legend: bool,

    /// What implementation to use
    #[arg(value_enum, requires = "blueprint", default_value_t = UpdateStrategy::default())]
    pub implementation: UpdateStrategy,

    /// Iterations to run in bench
    #[arg(short = 'i', long, default_value_t = 10_000_000)]
    pub iterations: usize,
}

fn main() {
    //colored::control::set_override(true);
    std::env::set_var("RUST_BACKTRACE", "1");
    let args = Args::parse();

    let read_file = |s: PathBuf| read_to_string(s).expect("File should exist");
    let parser_input: VcbParseInput = args
        .blueprint_string
        .clone()
        .or(args.blueprint_file.clone().map(read_file))
        .map(VcbParseInput::VcbBlueprint)
        .or(args
            .world_file
            .clone()
            .map(read_file)
            .map(VcbParseInput::VcbWorld))
        .unwrap();

    // branch to specific type here to remove overhead later.
    match args.implementation {
        UpdateStrategy::Reference => {
            handle_board::<{ UpdateStrategy::Reference as u8 }>(&args, parser_input)
        },
        UpdateStrategy::ScalarSimd => {
            handle_board::<{ UpdateStrategy::ScalarSimd as u8 }>(&args, parser_input)
        },
        UpdateStrategy::Simd => handle_board::<{ UpdateStrategy::Simd as u8 }>(&args, parser_input),
    }
}

fn handle_board<const STRATEGY: u8>(args: &Args, parser_input: VcbParseInput) {
    let mut board = { VcbParser::<STRATEGY>::parse(parser_input, true).unwrap() };
    match args.mode {
        RunMode::Gif => board.print_to_gif(),
        RunMode::Clip => board.print_to_clipboard(),
        RunMode::Emoji => board.print_regular_emoji(args.legend),
        RunMode::EmojiVcb => board.print_vcb_discord_emoji(args.legend),
        RunMode::Print => {
            board.update();
            board.print();
        },
        RunMode::Run => {
            execute!(
                stdout(),
                EnterAlternateScreen,
                crossterm::cursor::Hide,
                crossterm::terminal::DisableLineWrap
            )
            .unwrap();

            loop {
                //use std::time::Instant;
                execute!(stdout(), crossterm::cursor::MoveTo(0, 0),).unwrap();
                board.print();
                crossterm::terminal::enable_raw_mode().unwrap();
                //let prev = Instant::now();
                //while prev.elapsed().as_millis() < 16 {
                board.update();
                //}
                std::thread::sleep(Duration::from_millis(16));

                if crossterm::event::poll(Duration::from_secs(0)).unwrap() {
                    let term_event = crossterm::event::read().unwrap();
                    crossterm::terminal::disable_raw_mode().unwrap();
                    match term_event {
                        crossterm::event::Event::Resize(..) => {
                            execute!(stdout(), crossterm::terminal::Clear(ClearType::All),).unwrap()
                        },
                        _ => break,
                    }
                }
                crossterm::terminal::disable_raw_mode().unwrap();
            }
            execute!(
                stdout(),
                LeaveAlternateScreen,
                crossterm::cursor::Show,
                crossterm::terminal::EnableLineWrap
            )
            .unwrap()
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
