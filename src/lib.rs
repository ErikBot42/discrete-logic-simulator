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
//macro_rules! assert_assume {
//    ($statement:expr) => {
//        #[inline(always)]
//        unsafe fn foo() {}
//        foo();
//
//        #[cfg(debug_assertions)]
//        assert!($statement);
//        #[cfg(not(debug_assertions))]
//        std::intrinsics::assume($statement);
//    };
//}

pub mod blueprint;
pub mod logic;
pub mod raw_list;

#[cfg(test)]
mod tests {
    use crate::blueprint::{VcbBoard, VcbInput, VcbParser};
    use crate::logic::{BitPackSim, LogicSim, ReferenceSim, ScalarSim, SimdSim};

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

    fn simd_test(optimized: bool) -> bool {
        let optimized_board = prep_cases::<ReferenceSim>(optimized);
        let optimized_simd = prep_cases::<SimdSim>(optimized);
        //let mut correct: bool = true;
        for ((name, mut optimized), (_, mut optimized_simd)) in
            optimized_board.into_iter().zip(optimized_simd.into_iter())
        {
            //let width = optimized.width;
            for i in 1..30 {
                let optimized_state = optimized.make_state_vec();
                let optimized_state_simd = optimized_simd.make_state_vec();
                let diff_ids: Vec<usize> = optimized_state
                    .into_iter()
                    .zip(optimized_state_simd)
                    .enumerate()
                    .filter(|(_, (optim_bool, optim_bool_simd))| optim_bool != optim_bool_simd)
                    .map(|(j, (_, _))| j)
                    .collect();
                if diff_ids.len() != 0 {
                    //optimized.print_marked(&diff_ids);
                    panic!("simd/non simd mismatch for test {name}, in iteration {i} at positions {diff_ids:?}");
                    //correct = false;
                    //break;
                }
                optimized_simd.logic_sim.update();
                optimized.update();
            }
        }
        //correct
        true
    }

    fn run_test<Reference: LogicSim, Other: LogicSim>(optimized: bool, iterations: usize) {
        run_test_o::<Reference, Other>(optimized, optimized, iterations);
    }
    fn run_test_o<Reference: LogicSim, Other: LogicSim>(
        optimized: bool,
        optimized_other: bool,
        iterations: usize,
    ) {
        let optimized_board = prep_cases::<Reference>(optimized);
        let optimized_scalar = prep_cases::<Other>(optimized_other);
        for ((name, mut optimized), (_, mut optimized_scalar)) in optimized_board
            .into_iter()
            .zip(optimized_scalar.into_iter())
        {
            dbg!(name);
            compare_boards_iter(&mut optimized, &mut optimized_scalar, iterations, name);
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

    //#[test]
    //fn scalar_regression_test_unoptimized() {
    //    run_test::<ReferenceSim, ScalarSim>(false, 20);
    //}
    //#[test]
    //fn scalar_regression_test_optimized() {
    //    run_test::<ReferenceSim, ScalarSim>(true, 20);
    //}
    #[test]
    fn bitpack_regression_test_unoptimized() {
        run_test::<ReferenceSim, BitPackSim>(false, 20);
    }
    #[test]
    fn bitpack_regression_test_optimized() {
        run_test::<ReferenceSim, BitPackSim>(true, 20);
    }
    //#[test]
    //fn simd_regression_test_unoptimized() {
    //    simd_test(false);
    //}
    //#[test]
    //fn simd_regression_test_optimized() {
    //    simd_test(true);
    //}

    //#[test]
    //fn basic_gate_test_scalar() {
    //    generic_basic_gate_test_w::<ScalarSim>();
    //}
    //#[test]
    //fn basic_gate_test_reference() {
    //    generic_basic_gate_test_w::<ReferenceSim>();
    //}
    //#[test]
    //fn basic_gate_test_simd() {
    //    generic_basic_gate_test_w::<SimdSim>();
    //}
    //#[test]
    //fn basic_gate_test_bitpack() {
    //    generic_basic_gate_test_w::<BitPackSim>();
    //}
    fn generic_basic_gate_test_w<SIM: LogicSim>() {
        basic_gate_test::<SIM>(false, false);
        basic_gate_test::<SIM>(false, true);
        basic_gate_test::<SIM>(true, false);
        basic_gate_test::<SIM>(true, true);
    }

    fn basic_gate_test<SIM: LogicSim>(optimize: bool, add_all: bool) {
        dbg!(optimize, add_all);
        //const STRATEGY: u8 = UpdateStrategy::Reference as u8;
        let mut board: VcbBoard<SIM> = VcbParser::parse_compile(
            VcbInput::BlueprintLegacy(include_str!("../test_files/gates.blueprint").to_string()),
            optimize,
        )
        .unwrap();
        board.print_debug();
        assert_eq!(
            board.make_state_vec(),
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
            .iter()
            .map(|x| *x != 0)
            .collect::<Vec<bool>>()
        );
        println!("OK0");
        //if add_all {
        //    board.compiled_network.add_all_to_update_list()
        //};
        board.update();
        board.print_debug();
        assert_eq!(
            board.make_state_vec(),
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,
                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
            .iter()
            .map(|x| *x != 0)
            .collect::<Vec<bool>>()
        );
        println!("OK1");
        board.update();
        board.print_debug();
        assert_eq!(
            board.make_state_vec(),
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,
                1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
            .iter()
            .map(|x| *x != 0)
            .collect::<Vec<bool>>()
        );
        println!("OK2");
        board.update();
        board.print_debug();
        assert_eq!(
            board.make_state_vec(),
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,
                1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
            .iter()
            .map(|x| *x != 0)
            .collect::<Vec<bool>>()
        );
    }
}
