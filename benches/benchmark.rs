use criterion::{black_box, criterion_group, criterion_main, Criterion};


use logic_simulator::blueprint::*;

fn criterion_benchmark(c: &mut Criterion) {
    // Test parsing speed and execution speed for this list of blueprints.
    let tests = vec![
        ("intro",       include_str!("../test_files/intro.blueprint")),
        ("big_decoder", include_str!("../test_files/big_decoder.blueprint")),
        ("bcd_count",   include_str!("../test_files/bcd_count.blueprint")),
        ("gates",       include_str!("../test_files/gates.blueprint")),
    ];
    let pre_parsed: Vec<(&str,VcbBoard)> = tests.clone().into_iter().map(|x| (x.0,Parser::parse(x.1))).collect();
    
    let mut c_run = c.benchmark_group("run");
    for mut pre in pre_parsed {
        c_run.bench_function(
            pre.0, 
            |b| b.iter(|| pre.1.update())
            );
    }
    c_run.finish();

    let mut c_parse = c.benchmark_group("parse");
    for test in tests {
    c_parse.bench_function(
        test.0, 
        |b| b.iter(|| {
            let mut board = Parser::parse(black_box(test.1));
            board.update()
        }
        ));
    }
    c_parse.finish()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
