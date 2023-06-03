//#![feature(bench_black_box)]
use criterion::{criterion_group, criterion_main, Criterion};

use logic_simulator::blueprint::*;
use logic_simulator::logic::*;
use std::hint::black_box;

fn input_data() -> Vec<(&'static str, VcbInput)> {
    vec![
        (
            "serial mult",
            VcbInput::World(include_str!("../test_files/serial_multipliers.vcb").to_string()),
        ),
        (
            "big_decoder",
            VcbInput::BlueprintLegacy(
                include_str!("../test_files/big_decoder.blueprint").to_string(),
            ),
        ),
        (
            "bcd_count",
            VcbInput::BlueprintLegacy(
                include_str!("../test_files/bcd_count.blueprint").to_string(),
            ),
        ),
        (
            "intro",
            VcbInput::BlueprintLegacy(include_str!("../test_files/intro.blueprint").to_string()),
        ),
        // literally turns into a noop...
        // ("gates", include_str!("../test_files/gates.blueprint")),
    ]
}

fn pre_parse_tests<S: LogicSim>() -> Vec<(&'static str, VcbBoard<S>)> {
    input_data()
            .clone()
            .into_iter()
            //.map(|x| (x.0, black_box(Parser::parse(x.1, true))))
            .map(|x| (x.0, black_box(VcbParser::parse_compile(x.1, true).unwrap())))
            .collect()
}

#[rustfmt::skip]
fn criterion_benchmark_runtime(c: &mut Criterion) {
    bench_pre_parsed("run_bitpack", c, &mut black_box(pre_parse_tests::<BitPackSim>()));
    //bench_pre_parsed("run_reference", c, &mut black_box(pre_parse_tests::<ReferenceSim>()));
    //bench_pre_parsed("run_scalar", c, &mut black_box(pre_parse_tests::<ScalarSim>()));
    //bench_pre_parsed("run_simd", c, &mut black_box(pre_parse_tests::<SimdSim>()));
}

fn bench_pre_parsed<T: LogicSim>(
    group_name: &'static str,
    c: &mut Criterion,
    pre_parsed: &mut Vec<(&str, VcbBoard<T>)>,
) {
    let mut c_run = c.benchmark_group(group_name);
    //c_run.measurement_time(std::time::Duration::from_secs(30));
    //c_run.warm_up_time(std::time::Duration::from_secs(5));
    for pre in pre_parsed.iter_mut() {
        c_run.bench_function(pre.0, |b| b.iter(|| pre.1.update()));
    }
    c_run.finish();
}

fn bench_parsing<T: LogicSim>(
    group_name: &'static str,
    c: &mut Criterion,
    input: &[(&'static str, VcbInput)],
) {
    let mut c_run = c.benchmark_group(group_name);
    //c_run.sample_size(10);
    for (name, data) in input {
        c_run.bench_function(*name, |b| {
            b.iter(|| {
                black_box(VcbParser::parse_compile::<T>(black_box(data.clone()), true).unwrap())
            })
        });
    }
    c_run.finish();
}

fn criterion_benchmark_parsing(c: &mut Criterion) {
    let input = input_data();
    //bench_parsing::<ReferenceSim>("parse_reference", c, &input);
    //bench_parsing::<SimdSim>("parse_simd", c, &input);
    //bench_parsing::<ScalarSim>("parse_scalar", c, &input);
    bench_parsing::<BitPackSim>("parse_bit", c, &input);
}

criterion_group!(
    benches,
    criterion_benchmark_runtime,
    //criterion_benchmark_parsing
);
criterion_main!(benches);
