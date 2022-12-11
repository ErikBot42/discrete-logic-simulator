#![feature(bench_black_box)]
use criterion::{criterion_group, criterion_main, Criterion};

use logic_simulator::blueprint::*;
use std::hint::black_box;

fn input_data() -> Vec<(&'static str, &'static str)> {
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

fn pre_parse_tests<const STRATEGY: u8>() -> Vec<(&'static str, VcbBoard<STRATEGY>)> {
    input_data()
            .clone()
            .into_iter()
            //.map(|x| (x.0, black_box(Parser::parse(x.1, true))))
            .map(|x| (x.0, black_box(VcbParser::parse_compile(VcbInput::BlueprintLegacy(x.1.to_string()), false).unwrap())))
            .collect()
}

fn criterion_benchmark_runtime(c: &mut Criterion) {
    use logic_simulator::logic::UpdateStrategy;
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

fn bench_parsing<const STRATEGY: u8>(
    group_name: &'static str,
    c: &mut Criterion,
    input: &[(&'static str, &'static str)],
) {
    let mut c_run = c.benchmark_group(group_name);
    for (name, data) in input {
        c_run.bench_function(*name, |b| {
            b.iter(|| {
                black_box(
                    VcbParser::parse_compile::<STRATEGY>(
                        black_box(VcbInput::BlueprintLegacy(data.to_string())),
                        true,
                    )
                    .unwrap(),
                )
            })
        });
    }
    c_run.finish();
}

fn criterion_benchmark_parsing(c: &mut Criterion) {
    use logic_simulator::logic::UpdateStrategy;
    let input = input_data();
    bench_parsing::<{ UpdateStrategy::Reference as u8 }>("parse_reference", c, &input);
    bench_parsing::<{ UpdateStrategy::Simd as u8 }>("parse_simd", c, &input);
    bench_parsing::<{ UpdateStrategy::ScalarSimd as u8 }>("parse_scalar", c, &input);
}

criterion_group!(
    benches,
    criterion_benchmark_runtime,
    criterion_benchmark_parsing
);
criterion_main!(benches);
