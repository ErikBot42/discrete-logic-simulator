//extern crate base64;
//extern crate zstd;
//extern crate colored;

use logic_simulator::blueprint::Parser;

fn main() {
    let string = include_str!("../test_files/big_decoder.blueprint");
    //let string = include_str!("../test_files/intro.blueprint");
    //let string = include_str!("../test_files/gates.blueprint");
    //let string = include_str!("../test_files/small_decoder.blueprint");
    //let string = include_str!("../test_files/circle.blueprint");
    //let string = include_str!("../test_files/gates.blueprint");
    //let string = include_str!("../test_files/invalid_base64.blueprint");
    //let string = include_str!("../test_files/invalid_zstd.blueprint");

    let mut board = Parser::parse(string, true);

    let args: Vec<String> = std::env::args().collect();

    match &args[1][..] {
        "bench" | "-bench" => {
            const ITERATIONS: usize = 10_000_000;
            const ITERATIONS_F32: f32 = ITERATIONS as f32;
            println!("Running {ITERATIONS} iterations");
            let now = std::time::Instant::now();
            (0..ITERATIONS).for_each(|_| {
                board.update();
            });
            let elapsed = now.elapsed().as_millis() as f32 / 1000.0;
            let millisecond_per_iteration = elapsed / ITERATIONS_F32 * 1000.0;
            let microsecond_per_iteration = millisecond_per_iteration * 1000.0;
            let nanosecond_per_iteration = microsecond_per_iteration * 1000.0;
            let iterations_per_second = ITERATIONS_F32 / elapsed;
            println!("Elapsed: {elapsed} seconds");
            println!("ms/iteration: {millisecond_per_iteration}");
            println!("μs/iteration: {microsecond_per_iteration}");
            println!("ns/iteration: {nanosecond_per_iteration}");
            println!(
                "iteration/s: {} million",
                iterations_per_second / 1_000_000.0
            );
        },
        "run" => loop {
            board.update();
            board.print();
            let mut child = std::process::Command::new("sleep")
                .arg("0.5")
                .spawn()
                .unwrap();
            child.wait().unwrap();
        },
        "forever_bench" => loop {
            board.update_simd()
        },

        _ => loop {
            //use std::process::Command;
            //print!("\x1B[0;0H");
            board.update();
            //board.print();
            //let mut child = Command::new("sleep").arg("0.5").spawn().unwrap();
            //child.wait().unwrap();
        },
    }
}
