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
//#![feature(build_hasher_simple_hash_one)]
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

#[cfg(feature = "render")]
pub mod render;

#[cfg(test)]
macro_rules! gen_logic_tests {
    () => {
        gen_logic_tests!(
            xnor_edge_case,
            VcbInput::Blueprint(include_str!("../test_files/xnor_edge_case.blueprint").to_string()),
            "KLUv/WDoAp0AAIKBAYAQxoUCAQMAr5eqAEiHAeo=",
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
            "KLUv/aAgkwEAhSMANi0ZgBAXABcAFwBP//n9c5D/X//vH+pf7ffe5ed6kMqdKmj8ELx8D9T/B/irCAXMwCfv/u2HYPw/+j8Pjb98AAAfAD9neX/feoAAfwBPvzzAB/Snv/FD8Prr/+/KPwXsIzXJuw/vH4K3qKL8ypJJmgMyrUakZIRm0xwSiCA4CENAJEAIEAIEQRAEQRCEACGAEAKE4QoYIxLCDFJmm3NjXct3m2zDB5xhNc7GO2VbYECgtZs2o9G6xwpgxP4EviC7zxwuOSh55AvNg/CJBH7tkI5UFzkgcNNkO6tA0njo53G8o4b0tiYNwLGSrEjSXIDyU6kBgbaKmUSuzOLQk1hJR6BIF8He13qAdMSUF4VQbRlIP92Q4EVwW3m+HgG6Sb2Gyx1bZ8HOG2wBkMni3MVNYWF0BR3oXQ1CKEaH3j8DPJjNHMldISoLis7s3LXA9l9fYeuyJUdgyMSiMDSASSeUoMgLhLwAXcdP48YQCOUEUMGt3xmCEonDpJL0GSzoBtp+yjBdJRGe3rF36GUzyXyeRF5YUBcIySuC619/ADbXgycw/OZkYIJjCceIZJhZPljQxR7aEHhAmzgIxf0P87Uj8PAjW2hFBk73gmGPDroZVR3FwL1/MEw47/h6KJbkhiDNuVC/gnme3fwaEvJeq2/duj8BlJ39tHwTw9iK0egiBSl8NrTxzKoQ8QsesLxe+Hf83gOFNMMFDK41HwK7rSzNJcmegwB79jBz6IUQzsNjNG+dQK0wdwT+6glNCEHDrHQr9wWElpTvLI4IBx6P+60Znig+jkifNiX7ewGxBtPXjtLdEiAxJ6u8g5F88MrsF8/f8/uAM4cpZiNcDfycrpDD/x1KNXQHPprFDwBcJ5mSJQYCMvsT5R+wMrORWJ+p2QMfHQOrtIyIsjiagYCMJGWDL82vq6NyAf8OcFhTk8iBmjUCThga24rvjmfKvozXquubyRCPzK7Rsjk83YjANpB+uK9HoNZYX9XmrKCCmaOh2WhNvApGn+o1jnHADd37sjupy4EwPmewit7gDHQuzjk2d804sGBLrORvg6LJpoDsHCDRgYA7e4VgSXlSoCCPaUdDT7baf4lfmmJz4GCm5QRU+00S6IRhguRShwZu7DWmhjult66NA9PpV9keW+PFdHoiZsvtQkM1PO8Al1NF+44K2M9nNThJU34JNMQUjArPwNBwXGIYEwKJN0Lo2rueNvdA2FlE4esvWU4Yenr09RB4AAUGvZvFKqO09Yw4RzHDW5CzHoHgd09f9RPBlcNmlRkedHdqjwsjpxSLOX9gSiUP1TBBCBU1ZKIj5bwwI6itdcRSLCsUAc7bqkxgsy/3As5hWs9KHOEAqe9vjRN3AjHQXvh6k7hqOMmLQr4IGgUfYDNwxgJuAT6aZHLogD35KLZs19ugOAHO/hIfjyyOUqf6oBN3eSBfRQgyfNE6zLuLbPjPx2BLdqDjBX1hTWcicau6AV99nDZIsCJAEuKPopPRV3UUDoD/4N1L3GdwRjoAhgE=",
            0, 50);
        gen_logic_tests!(
            basic_gate,
            VcbInput::BlueprintLegacy(include_str!("../test_files/gates.blueprint").to_string()),
            "KLUv/WCKZd0BAKKDAoAQkNq5b6RWEQUQALEgBLHrqr7fN2k8QOfG8T7yCe4IXPuev0CBDqjAcwZhXGOAUcHLocGIlMAV",
            0, 50);
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
                do_test::<$sim_type>($optim, $input, $expect.to_string(), $iter);
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
        #[cfg(feature = "print_sim")]
        other.print_debug();
        println!("REFERENCE:");
        #[cfg(feature = "print_sim")]
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
        iter: usize,
    ) {
        let mut board: VcbBoard<SIM> = VcbParser::parse_compile(input, optimize).unwrap();
        assert_eq!(board.encode_state_base64(iter), expected);
    }
}
