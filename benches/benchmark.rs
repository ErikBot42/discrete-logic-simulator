
#![feature(bench_black_box)]
use criterion::{criterion_group, criterion_main, Criterion};

use logic_simulator::blueprint::*;

fn criterion_benchmark(c: &mut Criterion) {
    // Test parsing speed and execution speed for this list of blueprints.
    
    // hopefully this works at all times.
    fn black_box<T>(data: T) -> T {
        criterion::black_box(std::hint::black_box(data))
    }
    let tests = vec![
        (
            "big_decoder",
            include_str!("../test_files/big_decoder.blueprint"),
        ),
        (
            "bcd_count",
            include_str!("../test_files/bcd_count.blueprint"),
        ),
        ("intro", include_str!("../test_files/intro.blueprint")),
        // literally turns into a noop...
        // ("gates", include_str!("../test_files/gates.blueprint")),
    ];
    let mut pre_parsed: Vec<(&str, VcbBoard)> = tests
        .clone()
        .into_iter()
        .map(|x| (x.0, black_box(Parser::parse(x.1, true))))
        .collect();
    pre_parsed = black_box(pre_parsed);

    let mut c_run = c.benchmark_group("run");
    for pre in pre_parsed.iter() {
        pre.1.print(); // make sure optimizer does not remove everything
    }
    for pre in pre_parsed.iter_mut() {
        c_run.bench_function(pre.0, |b| {
            b.iter(|| {
                pre.1.update()
            })
        });
    }
    for pre in pre_parsed.iter() {
        //println!("{}", pre.1.compiled_network.iterations);
        pre.1.print(); // make sure optimizer does not remove everything
    }
    c_run.finish();
    //let mut c_parse = c.benchmark_group("parse");
    //for test in tests {
    //c_parse.bench_function(
    //    test.0,
    //    |b| b.iter(|| {
    //        let mut board = Parser::parse(black_box(test.1));
    //        board.update()
    //    }
    //    ));
    //}
    //c_parse.finish()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
