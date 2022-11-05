//#![warn(clippy::cargo)]
//#![warn(clippy::all)]
#![feature(portable_simd)]
#![feature(core_intrinsics)]
#![feature(generic_arg_infer)]
#![feature(let_chains)]
macro_rules! unwrap_or_else {
    ($expression:expr, $block:expr) => {
        match $expression {
            Some(value) => value,
            _ => $block,
        }
    };
}
macro_rules! assert_ge {
    ($first:expr, $second:expr) => {
        let a = $first;
        let b = $second;
        assert!(a >= b, "{a} < {b}");
    };
}
macro_rules! assert_le {
    ($first:expr, $second:expr) => {
        let a = $first;
        let b = $second;
        assert!(a <= b, "{a} > {b}");
    };
}
macro_rules! assert_le_len {
    ($first:expr, $second:expr) => {
        assert_le!($first.len(), $second.len());
    };
}
macro_rules! assert_eq_len {
    ($first:expr, $second:expr) => {
        assert_eq!($first.len(), $second.len());
    };
}

///// Assume in release, assert in debug.
//macro_rules! assume_debug_assert {
//    ($expression:expr) => {
//        debug_assert!($expression);
//        unsafe { std::intrinsics::assume($expression) };
//    };
//}

pub mod blueprint;
pub mod logic;
pub mod raw_list;

#[cfg(test)]
mod tests {
    use crate::blueprint::{Parser, VcbBoard};
    use crate::logic::UpdateStrategy;

    //#[cfg(test)]
    //use pretty_assertions::assert_eq;

    fn prep_cases<const STRATEGY: u8>(optimize: bool) -> Vec<(&'static str, VcbBoard<STRATEGY>)> {
        let cases: Vec<(&str, &str)> = vec![
            ("gates", include_str!("../test_files/gates.blueprint")),
            (
                "big_decoder",
                include_str!("../test_files/big_decoder.blueprint"),
            ),
            ("intro", include_str!("../test_files/intro.blueprint")),
            (
                "bcd_count",
                include_str!("../test_files/bcd_count.blueprint"),
            ),
        ];
        cases
            .clone()
            .into_iter()
            .map(|x| (x.0, Parser::parse(x.1, optimize)))
            .collect::<Vec<(&str, VcbBoard<STRATEGY>)>>()
    }

    #[test]
    fn optimization_regression_test() {
        for add_all_optimized in [true, false] {
            for add_all_unoptimized in [true, false] {
                const STRATEGY: u8 = UpdateStrategy::Reference as u8;
                let unoptimized = prep_cases::<STRATEGY>(false);
                let optimized = prep_cases::<STRATEGY>(true);
                for ((name, mut unoptimized), (_, mut optimized)) in
                    unoptimized.into_iter().zip(optimized.into_iter())
                {
                    for i in 0..30 {
                        assert_eq!(
                            unoptimized.make_state_vec(),
                            optimized.make_state_vec(),
                            "optimized/unoptimized mismatch for test {name}, in iteration {i} {add_all_unoptimized} {add_all_optimized}"
                        );
                        if add_all_optimized {
                            optimized.compiled_network.add_all_to_update_list()
                        };
                        if add_all_unoptimized {
                            optimized.compiled_network.add_all_to_update_list()
                        };
                        optimized.update();
                        unoptimized.update();
                    }
                }
            }
        }
    }

    fn simd_test(optimized: bool) -> bool {
        const STRATEGY_REF: u8 = UpdateStrategy::Reference as u8;
        const STRATEGY_SIMD: u8 = UpdateStrategy::Simd as u8;
        let optimized_board = prep_cases::<STRATEGY_REF>(optimized);
        let optimized_simd = prep_cases::<STRATEGY_SIMD>(optimized);
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
                    optimized.print_marked(&diff_ids);
                    panic!("simd/non simd mismatch for test {name}, in iteration {i} at positions {diff_ids:?}");
                    //correct = false;
                    //break;
                }
                optimized_simd.compiled_network.update();
                optimized.update();
            }
        }
        //correct
        true
    }
    fn scalar_test(optimized: bool) -> bool {
        const STRATEGY_REF: u8 = UpdateStrategy::Reference as u8;
        const STRATEGY_SCALAR: u8 = UpdateStrategy::ScalarSimd as u8;
        let optimized_board = prep_cases::<STRATEGY_REF>(optimized);
        let optimized_scalar = prep_cases::<STRATEGY_SCALAR>(optimized);
        //let mut correct: bool = true;
        for ((name, mut optimized), (_, mut optimized_scalar)) in optimized_board
            .into_iter()
            .zip(optimized_scalar.into_iter())
        {
            //let width = optimized.width;
            compare_boards_iter(&mut optimized, &mut optimized_scalar, 30);
        }
        //correct
        true
    }

    fn compare_boards_iter<const STRATEGY_REF: u8, const STRATEGY_OTHER: u8>(
        reference: &mut VcbBoard<STRATEGY_REF>,
        other: &mut VcbBoard<STRATEGY_OTHER>,
        iterations: usize,
    ) {
        for _ in 0..iterations {
            compare_boards(reference, other);
            other.update();
            reference.update();
        }
    }

    fn compare_boards<const STRATEGY_REF: u8, const STRATEGY_OTHER: u8>(
        reference: &VcbBoard<STRATEGY_REF>,
        other: &VcbBoard<STRATEGY_OTHER>,
    ) {
        let acc_reference = reference.compiled_network.get_acc_test();
        let acc_other = other.compiled_network.get_acc_test();
        let state_reference = reference.make_inner_state_vec();
        let state_other = other.make_inner_state_vec();
        //let diff_ids_acc: Vec<(usize, (u8, u8))> = optimized
        //    .compiled_network
        //    .get_acc_test()
        //    .zip(optimized_scalar.compiled_network.get_acc_test())
        //    .enumerate()
        //    .filter(|(_, (a, b))| a != b)
        //    .collect();
        let diff: Vec<_> = state_reference
            .into_iter()
            .zip(state_other)
            .zip(acc_reference.zip(acc_other))
            .enumerate()
            .filter(|(_, ((bool_a, bool_b), (acc_a, acc_b)))| bool_a != bool_b || acc_a != acc_b)
            .collect();
        println!("--------------------------------------");
        println!("OTHER:");
        other.print();
        println!("REFERENCE:");
        reference.print();
        if diff.len() != 0
        /* || diff_ids_acc.len() != 0*/
        {
            //println!("got");
            //other.print();
            //println!("expected:");
            //reference.print();
            //optimized.print_marked(&diff_ids);
            //scalar/non scalar mismatch for test {name}, in iteration {i} at nodes

            panic!(
                "diff ids: \n{}",
                diff.iter()
                    .map(|(i, ((ba, bb), (aa, ab)))| format!("{i}: {ba} {bb}, {aa} {ab}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            //\n{diff_ids_acc:?}
            //correct = false;
            //break;
        }
    }
    #[test]
    fn scalar_regression_test_unoptimized() {
        assert!(scalar_test(false));
    }
    #[test]
    fn scalar_regression_test_optimized() {
        assert!(scalar_test(true));
    }

    #[test]
    fn simd_regression_test_unoptimized() {
        assert!(simd_test(false));
    }

    #[test]
    fn simd_regression_test_optimized() {
        assert!(simd_test(true));
    }

    //#[test]
    //fn simd_repeated() {
    //    let mut correct: bool = true;
    //    for _ in 0..10 {
    //        correct &= simd_test(true);
    //        correct &= simd_test(false);
    //    }
    //    assert!(correct);
    //}

    #[test]
    fn basic_gate_test_optimized() {
        basic_gate_test(true, false);
        basic_gate_test(true, true);
    }
    #[test]
    fn basic_gate_test_unoptimimized() {
        basic_gate_test(false, false);
        basic_gate_test(false, true);
    }

    fn basic_gate_test(optimize: bool, add_all: bool) {
        const STRATEGY: u8 = UpdateStrategy::Reference as u8;
        let mut board: VcbBoard<STRATEGY> =
            Parser::parse(include_str!("../test_files/gates.blueprint"), optimize);
        board.print();
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
        if add_all {
            board.compiled_network.add_all_to_update_list()
        };
        board.update();
        board.print();
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
        if add_all {
            board.compiled_network.add_all_to_update_list()
        };
        board.update();
        board.print();
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
        if add_all {
            board.compiled_network.add_all_to_update_list()
        };
        board.update();
        board.print();
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
