#![feature(bench_black_box)]
use criterion::{criterion_group, criterion_main, Criterion};

use logic_simulator::blueprint::*;

fn black_box<T>(data: T) -> T {
    criterion::black_box(std::hint::black_box(data))
}

fn pre_parse_tests<const STRATEGY: u8>() -> Vec<(&'static str, VcbBoard<STRATEGY>)> {
    let mut pre_parsed = {
        let mut pre_parsed: Vec<(&str, VcbBoard<STRATEGY>)> = {
            vec![
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
            ]
        }
            .clone()
            .into_iter()
            //.map(|x| (x.0, black_box(Parser::parse(x.1, true))))
            .map(|x| (x.0, black_box(VcbParser::parse_to_board(x.1, false))))
            .collect();
        pre_parsed
    };
    pre_parsed
}

fn criterion_benchmark(c: &mut Criterion) {
    // Test parsing speed and execution speed for this list of blueprints.
    use logic_simulator::logic::UpdateStrategy;
    // hopefully this works at all times.
    const STRATEGY_REF: u8 = UpdateStrategy::Reference as u8;
    const STRATEGY_SIMD: u8 = UpdateStrategy::Simd as u8;
    const STRATEGY_SCALAR: u8 = UpdateStrategy::ScalarSimd as u8;

    let mut pre_parsed = black_box(pre_parse_tests::<STRATEGY_REF>());
    let mut pre_parsed_simd = black_box(pre_parse_tests::<STRATEGY_SIMD>());
    let mut pre_parsed_scalar = black_box(pre_parse_tests::<STRATEGY_SCALAR>());

    bench_pre_parsed("run_scalar", c, &mut pre_parsed_scalar);
    bench_pre_parsed("run_reference", c, &mut pre_parsed);
    bench_pre_parsed("run_simd", c, &mut pre_parsed_simd);

}

fn bench_pre_parsed<const STRATEGY: u8>(
    group_name: &'static str,
    c: &mut Criterion,
    pre_parsed: &mut Vec<(&str, VcbBoard<STRATEGY>)>,
) {
    let mut c_run = c.benchmark_group(group_name);
    for pre in pre_parsed.iter_mut() {
        c_run.bench_function(pre.0, |b| b.iter(|| pre.1.update()));
    }
    c_run.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
