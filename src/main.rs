use clap::{Parser, ValueEnum};
use crossterm::cursor::{Hide, MoveTo};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, ClearType, DisableLineWrap, EnterAlternateScreen,
    LeaveAlternateScreen,
};
use logic_simulator::blueprint::{VcbInput, VcbParser};
use logic_simulator::logic::{
    BitPackSim, LogicSim, ReferenceSim, ScalarSim, SimdSim, UpdateStrategy,
};
use std::fs::read_to_string;
use std::io::stdout;
use std::path::PathBuf;
use std::time::{Duration, Instant};

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
    /// Print board with internal ids
    PrintDebug,
    /// Run and display state of board
    Run,
    /// Run and render state of board
    RunGpu,
    /// Run a number of iterations and print time
    Bench,
    /// Copy image of board to clipboard.
    Clip,
    /// Make an animated gif of the board
    Gif,
    /// Only parse board
    Parse,
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

    /// Filepath to legacy VCB blueprint string
    #[arg(long, group = "blueprint")]
    pub blueprint_file_legacy: Option<PathBuf>,

    /// VCB blueprint string
    #[arg(short = 'b', long, group = "blueprint")]
    pub blueprint_string: Option<String>,

    /// legacy VCB blueprint string
    #[arg(long, group = "blueprint")]
    pub blueprint_string_legacy: Option<String>,

    /// Filepath to VCB world
    #[arg(short = 'w', long, group = "blueprint")]
    pub world_file: Option<PathBuf>,

    /// Filepath to legacy VCB world
    #[arg(long, group = "blueprint")]
    pub world_file_legacy: Option<PathBuf>,

    /// What mode to run the program in
    #[arg(value_enum, requires = "blueprint")]
    pub mode: RunMode,

    /// Print legend with emoji
    #[arg(short = 'l', long, requires = "blueprint")]
    pub legend: bool,

    /// What implementation to use (default to `reference`)
    #[arg(value_enum, requires = "blueprint", default_value_t = UpdateStrategy::default())]
    pub implementation: UpdateStrategy,

    /// Iterations to run in bench, or iterations per frame in run mode
    #[arg(short = 'i', long)]
    pub iterations: Option<usize>,
}

fn main() {
    //colored::control::set_override(true);
    std::env::set_var("RUST_BACKTRACE", "1");
    let args = Args::parse();

    let read_file = |s: PathBuf| read_to_string(s).expect("File should exist");
    let parser_input: VcbInput = (args
        .blueprint_string_legacy
        .clone()
        .or(args.blueprint_file_legacy.clone().map(read_file))
        .map(VcbInput::BlueprintLegacy))
    .or(args
        .blueprint_string
        .clone()
        .or(args.blueprint_file.clone().map(read_file))
        .map(VcbInput::Blueprint))
    .or(args
        .world_file_legacy
        .clone()
        .map(read_file)
        .map(VcbInput::WorldLegacy))
    .or(args.world_file.clone().map(read_file).map(VcbInput::World))
    .unwrap();

    // branch to specific type here to remove overhead later.
    match args.implementation {
        UpdateStrategy::Reference => {
            handle_board::<ReferenceSim>(&args, parser_input);
        },
        UpdateStrategy::ScalarSimd => {
            handle_board::<ScalarSim>(&args, parser_input);
        },
        UpdateStrategy::Simd => {
            handle_board::<SimdSim>(&args, parser_input);
        },
        UpdateStrategy::BitPack => {
            handle_board::<BitPackSim>(&args, parser_input);
        },
    }
}

fn handle_board<T: LogicSim>(args: &Args, parser_input: VcbInput) {
    let now = Instant::now();
    let mut board = { VcbParser::parse_compile::<T>(parser_input, true).unwrap() };
    println!("parsed entire board in {:?}", now.elapsed());
    match args.mode {
        RunMode::RunGpu => {
            board.run_gpu();
        }
        RunMode::Parse => (),
        RunMode::Gif => board.print_to_gif(args.iterations.unwrap_or(100)),
        RunMode::Clip => board.print_to_clipboard(),
        RunMode::Emoji => board.print_regular_emoji(args.legend),
        RunMode::EmojiVcb => board.print_vcb_discord_emoji(args.legend),
        RunMode::PrintDebug => {
            board.update();
            board.print_debug_constrain();
        },
        RunMode::Print => {
            board.update();
            board.print().unwrap();
        },
        RunMode::Run => {
            use crossterm::style::Print;
            use crossterm::QueueableCommand;

            execute!(stdout(), EnterAlternateScreen, Hide, DisableLineWrap).unwrap();

            let time_start = Instant::now();
            let mut time_prev = time_start;
            let mut updates_per_frame = args.iterations.unwrap_or(1);

            let mut total_ticks = 0;
            loop {
                //use std::time::Instant;
                execute!(stdout(), MoveTo(0, 0),).unwrap();
                board.print().unwrap();
                stdout()
                    .queue(Print(format!(
                        "{} {}",
                        (total_ticks as f64 / time_start.elapsed().as_secs_f64()) as u64,
                        updates_per_frame
                    )))
                    .unwrap();
                enable_raw_mode().unwrap();
                board.update_i(updates_per_frame);
                total_ticks += updates_per_frame;

                let target_delay = Duration::from_millis(16);
                let elapsed = time_prev.elapsed();

                updates_per_frame = (updates_per_frame as f64 * (target_delay.as_nanos() as f64)
                    / (elapsed.as_nanos() as f64)) as usize;

                updates_per_frame = updates_per_frame.min(10000).max(1);

                let delay = target_delay.saturating_sub(elapsed);

                time_prev = Instant::now();

                if crossterm::event::poll(delay).unwrap() {
                    let term_event = crossterm::event::read().unwrap();
                    disable_raw_mode().unwrap();
                    match term_event {
                        crossterm::event::Event::Resize(..) => {
                            execute!(stdout(), crossterm::terminal::Clear(ClearType::All)).unwrap();
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
            .unwrap();
        },
        RunMode::Bench => run_bench(args, board),
    }
    println!("Exiting...");
}

fn run_bench<T: LogicSim>(args: &Args, mut board: logic_simulator::blueprint::VcbBoard<T>) {
    #![allow(clippy::cast_precision_loss)]
    let iterations = args.iterations.unwrap_or(10_000_000);
    let iterations_f32 = iterations as f32;
    println!("Running {iterations} iterations");
    let now = std::time::Instant::now();
    board.update_i(iterations);
    let elapsed_raw = now.elapsed();
    let elapsed = now.elapsed().as_secs_f32();
    let millisecond_per_iteration = elapsed / iterations_f32 * 1000.0;
    let microsecond_per_iteration = millisecond_per_iteration * 1000.0;
    let nanosecond_per_iteration = microsecond_per_iteration * 1000.0;
    let iterations_per_second = iterations_f32 / elapsed;
    println!("Elapsed: {elapsed_raw:?}");
    println!("ms/iteration: {millisecond_per_iteration}");
    println!("μs/iteration: {microsecond_per_iteration}");
    println!("ns/iteration: {nanosecond_per_iteration}");
    println!("TPS: {iterations_per_second}");
    println!(
        "iteration/s: {} million",
        iterations_per_second / 1_000_000.0
    );
}
