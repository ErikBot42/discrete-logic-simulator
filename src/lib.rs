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
            xnor_edge_case,
            VcbInput::Blueprint(include_str!("../test_files/xnor_edge_case.blueprint").to_string()),
            "KLUv/SAAAQAA",
            0, 50);
        gen_logic_tests!(
            bcd_display,
            VcbInput::BlueprintLegacy(include_str!("../test_files/bcd_decoder_display.blueprint").to_string()),
            "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",
            0, 50);
        gen_logic_tests!(
            big_decoder,
            VcbInput::BlueprintLegacy(include_str!("../test_files/big_decoder.blueprint").to_string()),
            "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",
            0, 50);
        gen_logic_tests!(
            intro,
            VcbInput::BlueprintLegacy(include_str!("../test_files/intro.blueprint").to_string()),
            "KLUv/WAADh0SANZURTZgaZPGhQWYp3HcAVhJXMKOOOsdhfAj6hlAoUvwsdpAtuRwRPQqQIAR3hkvOa1ksR2bSJNkyxQzADwAOgDUw6ti/S/CCfK3BTEIE1cda7BwIm0DivmSPvPlVMfwFSgsy4oAp4eoWYkXu7EL0C839gONisAu7WwS3Yw4aD/lsSjQCYgHCQA0NJmKgyulevbTMcwFzU6QV8dwBKoGFxAG/AzLmEqpqosJDJhEAAwO/SVJECPamI7JQs+L1k4MMQ9IyHzJJ8fwzlaFYmSh50U7NRa52twKMCgLT6OoUBWnUTwwzUHq8VwI/7O2EObt/t3/uhfCvOidd8MhwlquCYj9Neih3uN5Q1aEY1KpLpEc2pkTTV0pJxqAzKjhrIokKdVoDjAjEZE4HRKACBVSTBkJRCQQkUAikVACiRKUFCQdFXAA2QHpIbOXMapvT+q4mV4PVE0PA5BhPjAQjQOXtUAhjmwObH0php4tBgJ3TmIAN7QBcA4gZo6ptzjqsp9QzQAc0Mc5sPyU7Z+QTGhGskLzD5IhuqQBJO4yH9nKfsifiD2xTzwvBab4gDCAdD4cM6Q3iIIMyCETOYDNoRrQqgMPwwDQ6ozUnTMSPoBB83cAkqwWbXiV2e4Ad08PQOUrm4kD4TgAfPNtzsoUqCT2mbVAcuDkH4gDXG4/YxUE5CapxGEpDqhayXj0dCbfcit+nLScFYf/EcWM2+zMD2eDt2cg9LSK2vcmNI5ptdLodZmMI9OBT0or+ixOjCP6Ti54+zlAPQ==",
            0, 15);
        gen_logic_tests!(
            basic_gate,
            VcbInput::BlueprintLegacy(include_str!("../test_files/gates.blueprint").to_string()),
            "KLUv/WBAAOUDAGQFAAARAQAAAKiACgAgIhVQAQAAAEQEAAAAoAIqAAAAgIhUQAUqsQNUQAUQEUAldoAKqCICqMQOUAEVQEQAldgBKqACO6ECEQFAJ1YiAgDshApAQECAldAQAAVACygFth5iBwoWgHxA5kD4EHKJSSMl9Et4EEMlYDDOALA=",
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
                do_test::<$sim_type>($optim, $input, $expect.to_string(), $pre_iter, $iter);
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
