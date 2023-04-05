#![feature(try_blocks)]
#![feature(portable_simd)]
#![feature(core_intrinsics)]
#![feature(generic_arg_infer)]
#![feature(let_chains)]
#![feature(iter_array_chunks)]
#![feature(iter_next_chunk)]
#![feature(is_sorted)]
#![feature(array_chunks)]
#![feature(array_try_map)]
#![feature(stdsimd)]
#![feature(unchecked_math)]
#![feature(allocator_api)]
#![feature(build_hasher_simple_hash_one)]
macro_rules! timed {
    ($block:expr, $print_str:expr) => {{
        let now = std::time::Instant::now();
        let a = ($block);
        println!($print_str, now.elapsed());
        a
    }};
}
macro_rules! unwrap_or_else {
    ($expression:expr, $block:expr) => {
        match $expression {
            Some(value) => value,
            _ => $block,
        }
    };
}
macro_rules! assert_le {
    ($first:expr, $second:expr) => {
        let a = $first;
        let b = $second;
        assert!(a <= b, "{a} > {b}");
    };
}
macro_rules! assert_eq_len {
    ($first:expr, $second:expr) => {
        assert_eq!($first.len(), $second.len());
    };
}

pub mod blueprint;
pub mod logic;
pub mod raw_list;
pub mod render;

#[cfg(test)]
macro_rules! gen_logic_tests {
    () => {
        gen_logic_tests!(
            intro,
            VcbInput::BlueprintLegacy(include_str!("../test_files/intro.blueprint").to_string()),
            "KLUv/WAAY50BAOgAAIAKAKgAEBEAgAoACAIAcAAAAACAAwAAAAAcAAcAWeLA2oEXoD5ABZABCAyS8aEAFg==",
            0, 100);
        gen_logic_tests!(
            basic_gate,
            VcbInput::BlueprintLegacy(include_str!("../test_files/gates.blueprint").to_string()),
            "KLUv/WBAANUBAEQCAAARAQAAAKiACgAgIhVQAQAAAEQEAAAAoAIqAAAAgIhUQAUACAD1IVCAI5IS+iU8iKESMBhnAFg=",
            0, 5);
    };
    ($test_case_name:ident, $input:expr, $expect:expr, $pre_iter:expr, $iter:expr) => {
        gen_logic_tests!(
            $test_case_name, $input, $expect, $pre_iter, $iter,
            [ReferenceSim, reference_sim],
            [BitPackSim, bit_pack]
        );
    };
    ($name:ident, $input:expr, $expect:expr, $pre_iter:expr, $iter:expr, [$stype:ty, $sname:ident], $([$stypes:ty, $snames:ident]), +) => {
        gen_logic_tests!($name, $input, $expect, $pre_iter, $iter, [$stype, $sname]);
        gen_logic_tests!($name, $input, $expect, $pre_iter, $iter, $([$stypes, $snames]), +);
    };
    ($test_case_name:ident, $input:expr, $expect:expr, $pre_iter:expr, $iter:expr, [$sim_type:ty, $sim_name:ident]) => {
        gen_logic_tests!(true, optimized, $test_case_name, $input, $expect, $pre_iter, $iter, $sim_type, $sim_name);
        gen_logic_tests!(false, unoptimized, $test_case_name, $input, $expect, $pre_iter, $iter, $sim_type, $sim_name);
    };
    ( $optim:expr, $optim_str:ident, $test_case_name:ident, $input:expr, $expect:expr, $pre_iter:expr, $iter:expr, $sim_type:ty, $sim_name:ident) => {
        paste::paste! {
            #[test]
            fn [<$sim_name _ test _ $test_case_name _ $optim_str>]() {
                do_test::<$sim_type>(true, $input, $expect.to_string(), $pre_iter, $iter);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    gen_logic_tests!();

    use crate::blueprint::{VcbBoard, VcbInput, VcbParser};
    use crate::logic::{BitPackSim, LogicSim, ReferenceSim}; //, BatchSim};

    fn prep_cases_closure<SIM: LogicSim>(
        optimize: bool,
    ) -> Vec<(&'static str, Box<dyn FnOnce() -> VcbBoard<SIM>>)> {
        let cases: Vec<(&str, _)> = vec![
            (
                "gates",
                VcbInput::BlueprintLegacy(
                    include_str!("../test_files/gates.blueprint").to_string(),
                ),
            ),
            (
                "big_decoder",
                VcbInput::BlueprintLegacy(
                    include_str!("../test_files/big_decoder.blueprint").to_string(),
                ),
            ),
            (
                "intro",
                VcbInput::BlueprintLegacy(
                    include_str!("../test_files/intro.blueprint").to_string(),
                ),
            ),
            (
                "bcd_count",
                VcbInput::BlueprintLegacy(
                    include_str!("../test_files/bcd_count.blueprint").to_string(),
                ),
            ),
            (
                "xnor edge case",
                VcbInput::Blueprint(
                    include_str!("../test_files/xnor_edge_case.blueprint").to_string(),
                ),
            ),
        ];
        cases
            .into_iter()
            .map(|x| {
                let optimize = optimize;
                (
                    x.0,
                    Box::from(move || VcbParser::parse_compile::<SIM>(x.1, optimize).unwrap()) as _,
                )
            })
            .collect::<Vec<_>>()
    }

    fn prep_cases<SIM: LogicSim>(optimize: bool) -> Vec<(&'static str, VcbBoard<SIM>)> {
        let cases: Vec<(&str, _)> = vec![
            (
                "gates",
                VcbInput::BlueprintLegacy(
                    include_str!("../test_files/gates.blueprint").to_string(),
                ),
            ),
            (
                "big_decoder",
                VcbInput::BlueprintLegacy(
                    include_str!("../test_files/big_decoder.blueprint").to_string(),
                ),
            ),
            (
                "intro",
                VcbInput::BlueprintLegacy(
                    include_str!("../test_files/intro.blueprint").to_string(),
                ),
            ),
            (
                "bcd_count",
                VcbInput::BlueprintLegacy(
                    include_str!("../test_files/bcd_count.blueprint").to_string(),
                ),
            ),
            (
                "xnor edge case",
                VcbInput::Blueprint(
                    include_str!("../test_files/xnor_edge_case.blueprint").to_string(),
                ),
            ),
        ];
        cases
            .clone()
            .into_iter()
            .map(|x| (x.0, VcbParser::parse_compile(x.1, optimize).unwrap()))
            .collect::<Vec<(&str, VcbBoard<SIM>)>>()
    }

    #[test]
    fn optimization_regression_test() {
        run_test_o::<ReferenceSim, ReferenceSim>(false, true, 30);
    }

    fn run_test<Reference: LogicSim, Other: LogicSim>(optimized: bool, iterations: usize) {
        run_test_o::<Reference, Other>(optimized, optimized, iterations);
    }
    fn run_test_o<Reference: LogicSim, Other: LogicSim>(
        optimized: bool,
        optimized_other: bool,
        iterations: usize,
    ) {
        let optimized_board = prep_cases_closure::<Reference>(optimized);
        let optimized_scalar = prep_cases_closure::<Other>(optimized_other);
        for ((name, optimized), (_, optimized_scalar)) in optimized_board
            .into_iter()
            .zip(optimized_scalar.into_iter())
        {
            dbg!(name);
            compare_boards_iter(
                &mut (optimized()),
                &mut (optimized_scalar()),
                iterations,
                name,
            );
        }
    }

    fn compare_boards_iter(
        reference: &mut VcbBoard<impl LogicSim>,
        other: &mut VcbBoard<impl LogicSim>,
        iterations: usize,
        name: &str,
    ) {
        for i in 0..iterations {
            compare_boards(reference, other, i, name);
            other.update();
            reference.update();
        }
    }

    fn compare_boards(
        reference: &mut VcbBoard<impl LogicSim>,
        other: &mut VcbBoard<impl LogicSim>,
        iteration: usize,
        name: &str,
    ) {
        //let acc_reference = reference.compiled_network.get_acc_test();
        //let acc_other = other.compiled_network.get_acc_test();
        let state_reference = reference.make_inner_state_vec();
        let state_other = other.make_inner_state_vec();
        assert_eq!(state_reference.len(), state_other.len());
        let diff: Vec<_> = state_reference
            .into_iter()
            .zip(state_other)
            //.zip(acc_reference.zip(acc_other))
            .enumerate()
            .filter(|(_, (bool_a, bool_b))| bool_a != bool_b)
            .collect();
        println!("--------------------------------------");
        println!("OTHER:");
        other.print_debug();
        println!("REFERENCE:");
        reference.print_debug();
        if diff.len() != 0 {
            println!(
                "diff ids: \n{}",
                diff.iter()
                    .map(|(i, (ba, bb))| format!("{i}: {ba} {bb}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );

            //reference.print_marked(&diff.iter().map(|d| d.0).collect::<Vec<_>>());

            panic!("state differs with reference after {iteration} iterations in test {name}");
        }
    }

    #[test]
    fn bitpack_regression_test_unoptimized() {
        run_test::<ReferenceSim, BitPackSim>(false, 20);
    }
    #[test]
    fn bitpack_regression_test_optimized() {
        run_test::<ReferenceSim, BitPackSim>(true, 20);
    }
    pub(crate) fn do_test<SIM: LogicSim>(
        optimize: bool,
        input: VcbInput,
        expected: String,
        pre_iter: usize,
        iter: usize,
    ) {
        let mut board: VcbBoard<SIM> = VcbParser::parse_compile(input, optimize).unwrap();
        assert_eq!(board.encode_state_base64(pre_iter, iter), expected);
    }
}
