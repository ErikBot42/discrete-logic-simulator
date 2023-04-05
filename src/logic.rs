//! logic.rs: Contains the simulaion engine itself.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::inline_always)]
//#![allow(dead_code)]

pub mod batch_sim;
pub mod bitmanip;
pub mod bitpack_sim;
pub mod gate_status;
pub mod network;
pub mod reference_sim;

pub(crate) use crate::logic::network::{GateNetwork, InitializedNetwork};
use network::Csr;
use std::simd::{Mask, Simd};
use strum_macros::EnumIter;

//pub type ReferenceSim = CompiledNetwork<{ UpdateStrategy::Reference as u8 }>;
pub type ReferenceSim = reference_sim::ReferenceLogicSim;
pub type BitPackSim = bitpack_sim::BitPackSimInner;
//pub type BatchSim = batch_sim::ReferenceBatchSim;

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone, PartialOrd, Ord, Default)]
/// A = active inputs
/// T = total inputs
/// S = State
/// P = Previous inputs
pub(crate) enum GateType {
    /// A == T
    And,
    /// A > 0
    Or,
    /// A == 0
    Nor,
    /// A != T
    Nand,
    /// A % 2 == 1
    Xor,
    /// A % 2 == 0
    Xnor,
    /// S != (A % 2 == 1 && P % 2 == 0)
    Latch,
    /// S != (A % 2 == 1 && P % 2 == 0)
    Interface(Option<u8>),
    /// A > 0
    #[default]
    Cluster, // equivalent to OR
}
impl GateType {
    /// can a pair of identical connections be removed without changing behaviour
    fn can_delete_double_identical_inputs(self) -> bool {
        match self {
            GateType::Xor | GateType::Xnor | GateType::Latch | GateType::Interface(_) => true,
            GateType::And | GateType::Or | GateType::Nor | GateType::Nand | GateType::Cluster => {
                false
            },
        }
    }
    /// can one connection in pair of identical connections be removed without changing behaviour
    fn can_delete_single_identical_inputs(self) -> bool {
        match self {
            GateType::And | GateType::Or | GateType::Nor | GateType::Nand | GateType::Cluster => {
                true
            },
            GateType::Xor | GateType::Xnor | GateType::Latch | GateType::Interface(_) => false,
        }
    }
    fn is_cluster(self) -> bool {
        matches!(self, GateType::Cluster)
    }

    /// Is this gate always off (constant)
    fn constant_analysis(
        k: GateType,
        initial_state: bool,
        max_active: usize,
        inputs: usize,
    ) -> bool {
        let acc0: AccType = Gate::calc_acc_i(inputs, k);
        let acc1: AccType = acc0.wrapping_add(AccType::try_from(max_active).unwrap());
        // extreme points for acc will always be [0, max_active] for and, or, nor, nand
        // AndNor => acc == 0,
        // OrNand => acc != 0,
        // XorXnor => acc & 1 == 1,
        // Latch => state != ((acc & 1 == 1) && (acc_prev & 1 == 0)),
        // TODO: EXPAND MATCHING TO INCLUDE LATCH
        use GateType::*;
        match k {
            And | Nor => acc0 != 0 && acc1 != 0,
            Or | Nand | Cluster => acc0 == 0 && acc1 == 0,
            Xor | Xnor => acc0 == 0 && acc1 == 0,
            Latch => !initial_state && acc0 == 0 && acc1 == 0,
            Interface(_) => false,
        }
    }
}

/// The only cases that matter at the hot code sections.
/// For example And/Nor can be evalutated in the same way
/// with an offset to the input count (acc).
#[derive(Debug, Copy, Clone, PartialEq, EnumIter)]
pub(crate) enum RunTimeGateType {
    OrNand,
    AndNor,
    XorXnor,
    Latch,
}
impl RunTimeGateType {
    const fn new(kind: GateType) -> Self {
        match kind {
            GateType::And | GateType::Nor => RunTimeGateType::AndNor,
            GateType::Or | GateType::Nand | GateType::Cluster => RunTimeGateType::OrNand,
            GateType::Xor | GateType::Xnor => RunTimeGateType::XorXnor,
            GateType::Latch => RunTimeGateType::Latch,
            GateType::Interface(_) => RunTimeGateType::Latch,
        }
    }

    /// (`is_inverted`, `is_xor`)
    const fn calc_flags(kind: RunTimeGateType) -> (bool, bool) {
        match kind {
            RunTimeGateType::OrNand => (false, false),
            RunTimeGateType::AndNor => (true, false),
            RunTimeGateType::XorXnor => (false, true),
            RunTimeGateType::Latch => (true, true),
        }
    }

    /// Required `acc` value to force the gate to never change state
    /// Assumes `acc` never changes (`prev_acc` = `acc`)
    const fn acc_to_never_activate(&self) -> u8 {
        let state = false;
        if state == Gate::evaluate(0, 0, state, *self) {
            return 0;
        }
        if state == Gate::evaluate(1, 1, state, *self) {
            return 1;
        }
        panic!();
    }
}

// speed: u8 < u16 = u32
// Gates only care about this equalling 0 or parity, ergo signed/unsigned is irrelevant.
// let n = 1 << bits_in_value(AccType)
// Or, Nand: max active: n
// Nor, And: max inactive: n
// Xor, Xnor: no limitation
// u16 and u32 have similar speeds for this
type AccTypeInner = u8;
type AccType = AccTypeInner;

type SimdLogicType = AccTypeInner;

// tests don't need that many indexes, but this is obviously a big limitation.
// u16 enough for typical applications (65536), u32
// u32 > u16, u32
type IndexType = u32; //AccTypeInner;
type UpdateList = crate::raw_list::RawList<IndexType>;

type GateKey = (GateType, Vec<IndexType>, bool);

use network::GateNode;

pub trait LogicSim {
    // ===============================================
    //
    //              Required functions
    //
    // ==============================================

    // outputs: impl Iterator<Item = impl Iterator<Item = usize>>
    // nodes: impl Iterator<Item = impl Iterator<Item = usize>>

    /// Create `LogicSim` struct from optimized network and a id translation table to convert from
    /// external gate ids to internal gate ids.
    //fn create(network: InitializedNetwork) -> (Vec<IndexType>, Self);
    fn create(
        outputs: impl IntoIterator<Item = impl IntoIterator<Item = usize>>,
        nodes: Vec<GateNode>,
        table: Vec<IndexType>,
    ) -> (Vec<IndexType>, Self);

    /// Get state from *internal* id
    fn get_state_internal(&self, gate_id: usize) -> bool;

    /// 0..num_gates_internal should be a valid range to query for gate state
    /// TODO: put in wrapper
    fn num_gates_internal(&self) -> usize;

    /// Run 1 tick, use [`LogicSim::update_i`] for optimized repeated iteration.
    fn update(&mut self);

    // ===============================================
    //
    //              Provided functions
    //
    // ==============================================

    /// Update network `iterations` times.
    /// Sim may override this to perform optimizations
    fn update_i(&mut self, iterations: usize) {
        for _ in 0..iterations {
            self.update();
        }
    }

    const STRATEGY: UpdateStrategy;
}
pub trait RenderSim: LogicSim {
    /// Simulate 1 tick.
    #[inline(always)]
    fn rupdate(&mut self) {
        self.update();
    }
    /// Clear and write state bitvec
    fn get_state_in(&mut self, v: &mut Vec<u64>) {
        use bitmanip::{pack_bits, pack_bits_remainder};
        let mut chunks = (0..(self.num_gates_internal()))
            .map(|i| self.get_state_internal(i))
            .array_chunks::<64>();
        v.clear();
        while let Some(chunk) = chunks.next() {
            v.push(pack_bits(chunk));
        }
        if let Some(remainder) = chunks.into_remainder() {
            v.push(pack_bits_remainder(remainder));
        }
    }
}

/// data needed after processing network
#[derive(Debug, Clone)]
pub(crate) struct Gate {
    inputs: Vec<IndexType>,  // list of ids
    outputs: Vec<IndexType>, // list of ids
    kind: GateType,
    initial_state: bool,
}
impl Gate {
    fn new(kind: GateType, initial_state: bool) -> Self {
        Gate {
            inputs: Vec::new(),
            outputs: Vec::new(),
            kind,
            initial_state,
        }
    }
    fn is_propably_constant(&self) -> bool {
        self.inputs.is_empty()
    }
    fn acc(&self) -> AccType {
        self.calc_acc()
    }
    fn calc_acc(&self) -> AccType {
        Self::calc_acc_i(self.inputs.len(), self.kind)
    }
    fn calc_acc_i(inputs: usize, kind: GateType) -> AccType {
        match kind {
            GateType::And | GateType::Nand => {
                let a: AccType = 0;
                a.wrapping_sub(AccType::try_from(inputs).unwrap())
            },
            GateType::Or
            | GateType::Nor
            | GateType::Xor
            | GateType::Cluster
            | GateType::Latch
            | GateType::Interface(_) => 0,
            GateType::Xnor => 1,
        }
    }

    fn kind_runtime(&self) -> RunTimeGateType {
        RunTimeGateType::new(self.kind)
    }
    fn is_cluster_a_xor_is_cluster_b_and_no_type_overlap_equal_cardinality(
        &self,
        other: &Gate,
    ) -> bool {
        self.is_cluster_a_xor_is_cluster_b(other)
            || (self.kind_runtime() != other.kind_runtime())
            || self.outputs.len() != other.outputs.len()
    }

    fn is_cluster_a_xor_is_cluster_b_and_no_type_overlap(&self, other: &Gate) -> bool {
        self.is_cluster_a_xor_is_cluster_b(other) || (self.kind_runtime() != other.kind_runtime())
    }

    fn is_cluster_a_xor_is_cluster_b(&self, other: &Gate) -> bool {
        self.kind.is_cluster() != other.kind.is_cluster()
    }

    /// add inputs and handle internal logic for them
    /// # NOTE
    /// Redundant connections are allowed here.
    fn add_inputs_vec(&mut self, inputs: &mut Vec<IndexType>) {
        self.inputs.append(inputs);
        self.inputs.sort_unstable(); // TODO: probably not needed
    }
    const fn evaluate_simple<T>(
        acc: AccType,
        acc_prev_parity: bool,
        state: bool,
        kind: RunTimeGateType,
    ) -> bool {
        let acc_prev = acc_prev_parity as AccType;
        match kind {
            RunTimeGateType::OrNand => acc != 0,
            RunTimeGateType::AndNor => acc == 0,
            RunTimeGateType::XorXnor => acc & 1 == 1,
            RunTimeGateType::Latch => state != ((acc & 1 == 1) && (acc_prev & 1 == 0)),
        }
    }
    #[inline]
    const fn evaluate(acc: AccType, acc_prev: AccType, state: bool, kind: RunTimeGateType) -> bool {
        match kind {
            RunTimeGateType::OrNand => acc != 0,
            RunTimeGateType::AndNor => acc == 0,
            RunTimeGateType::XorXnor => acc & 1 == 1,
            RunTimeGateType::Latch => state != ((acc & 1 == 1) && (acc_prev & 1 == 0)),
        }
    }
    #[inline]
    #[cfg(test)]
    const fn evaluate_from_flags(
        acc: AccType,
        acc_prev: AccType,
        state: bool,
        (is_inverted, is_xor): (bool, bool),
    ) -> bool {
        // inverted from perspective of or gate
        // hopefully this generates branchless code.
        if !is_xor {
            (acc != 0) != is_inverted
        } else {
            if is_inverted {
                state != ((acc & 1 == 1) && (acc_prev & 1 == 0))
            } else {
                acc & 1 == 1
            }
        }
    }
    #[inline]
    #[cfg(test)]
    const fn evaluate_branchless(
        acc: AccType,
        acc_prev: AccType,
        state: bool,
        (is_inverted, is_xor): (bool, bool),
    ) -> bool {
        !is_xor && ((acc != 0) != is_inverted)
            || is_xor
                && ((!is_inverted && (acc & 1 == 1))
                    || (is_inverted && (state != ((acc & 1 == 1) && (acc_prev & 1 == 0)))))
    }
    //#[must_use]
    #[cfg(test)]
    fn evaluate_simd<const LANES: usize>(
        // only acc is not u8...
        acc: Simd<AccType, LANES>,
        is_inverted: Simd<SimdLogicType, LANES>,
        is_xor: Simd<SimdLogicType, LANES>,
        old_state: Simd<SimdLogicType, LANES>,
    ) -> (Simd<SimdLogicType, LANES>, Simd<SimdLogicType, LANES>)
    where
        std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount,
    {
        use std::simd::SimdPartialEq;
        let acc_logic = acc.cast::<SimdLogicType>();
        let acc_not_zero = acc_logic
            .simd_ne(Simd::splat(0))
            .select(Simd::splat(1), Simd::splat(0)); // 0|1
        let xor_term = is_xor & acc_logic; // 0|1
                                           // TODO is this just subtracting 1?
        let not_xor = !is_xor; // 0|1111...
                               //let xor_term = is_xor & acc & Simd::splat(1);
                               //let not_xor = !is_xor & Simd::splat(1);
        let acc_term = not_xor & (is_inverted ^ acc_not_zero); //0|1
        let new_state = acc_term | xor_term; //0|1
        (new_state, old_state ^ new_state)
    }

    /// calculate a key that is used to determine if the gate
    /// can be merged with other gates.
    fn calc_key(&self) -> GateKey {
        // TODO: can potentially include inverted.
        // but then every connection would have to include
        // connection information

        // TODO: merge latch further
        let kind = self.kind;
        let inputs_len = self.inputs.len();
        let kind = match inputs_len {
            0 | 1 => match kind {
                GateType::Nand | GateType::Xnor | GateType::Nor => GateType::Nor,
                GateType::And | GateType::Or | GateType::Xor => GateType::Or,
                GateType::Cluster => GateType::Cluster, // merging cluster with gate is invalid
                GateType::Latch => GateType::Latch,     // TODO: is merging latch with gate invalid?
                GateType::Interface(s) => GateType::Interface(s), // never merge interface
            },
            _ => kind,
        };
        assert!(self.inputs.is_sorted());
        (kind, self.inputs.clone(), self.initial_state)
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
#[repr(u8)]
pub enum UpdateStrategy {
    #[default]
    /// Used to compare performance and check correctness.
    Reference = 0,
    /// Bit manipulation
    BitPack = 3,
    /// Partial updates
    Batch = 4,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inactive_acc_possible() {
        use strum::IntoEnumIterator;
        for g in RunTimeGateType::iter() {
            g.acc_to_never_activate();
        }
    }

    #[test]
    fn gate_evaluation_regression() {
        for (kind, cluster) in [
            (RunTimeGateType::OrNand, true),
            (RunTimeGateType::OrNand, false),
            (RunTimeGateType::AndNor, false),
            (RunTimeGateType::XorXnor, false),
            (RunTimeGateType::Latch, false),
        ] {
            for acc in [
                (0 as AccType).wrapping_sub(2),
                (0 as AccType).wrapping_sub(1),
                0,
                1,
                2,
            ] {
                for acc_prev in [
                    (0 as AccType).wrapping_sub(2),
                    (0 as AccType).wrapping_sub(1),
                    0,
                    1,
                    2,
                ] {
                    for state in [true, false] {
                        let flags = RunTimeGateType::calc_flags(kind);
                        let in_update_list = true;
                        let mut status = gate_status::new(in_update_list, state, kind);
                        let status_delta = if cluster {
                            gate_status::eval_mut::<true>(&mut status, acc, acc_prev)
                        } else {
                            gate_status::eval_mut::<false>(&mut status, acc, acc_prev)
                        };

                        const LANES: usize = 64;
                        let mut status_simd: Simd<gate_status::Inner, LANES> =
                            Simd::splat(gate_status::new(in_update_list, state, kind));
                        let status_delta_simd = if cluster {
                            gate_status::eval_mut_simd::<true, LANES>(
                                &mut status_simd,
                                Simd::splat(acc),
                                Simd::splat(acc_prev),
                            )
                        } else {
                            gate_status::eval_mut_simd::<false, LANES>(
                                &mut status_simd,
                                Simd::splat(acc),
                                Simd::splat(acc_prev),
                            )
                        };
                        let mut status_scalar =
                            gate_status::splat_u32(gate_status::new(in_update_list, state, kind));
                        let status_scalar_pre = status_scalar;
                        let acc_scalar = gate_status::splat_u32(acc);
                        let acc_scalar_prev = gate_status::splat_u32(acc_prev);
                        let status_delta_scalar = if cluster {
                            gate_status::eval_mut_scalar::<true>(
                                &mut status_scalar,
                                acc_scalar,
                                acc_scalar_prev,
                            )
                        } else {
                            gate_status::eval_mut_scalar::<false>(
                                &mut status_scalar,
                                acc_scalar,
                                acc_scalar_prev,
                            )
                        };

                        let mut res = vec![
                            Gate::evaluate_from_flags(acc, acc_prev, state, flags),
                            Gate::evaluate_branchless(acc, acc_prev, state, flags),
                            Gate::evaluate(acc, acc_prev, state, kind),
                            gate_status::state(status),
                        ];

                        assert!(
                            res.windows(2).all(|r| r[0] == r[1]),
                            "Some gate evaluators have diffrent behavior:
                        res: {res:?},
                        kind: {kind:?},
                        flags: {flags:?},
                        acc: {acc},
                        acc_prev: {acc_prev},
                        prev state: {state}"
                        );

                        let mut scalar_state_vec: Vec<bool> =
                            gate_status::packed_state_vec(status_scalar);
                        res.append(&mut scalar_state_vec);

                        assert!(
                            res.windows(2).all(|r| r[0] == r[1]),
                            "Scalar gate evaluators have diffrent behavior:
                        res: {res:?},
                        kind: {kind:?},
                        flags: {flags:?},
                        acc: {acc},
                        acc4: {acc_scalar},
                        status_pre: {:?},
                        status_scalar: {:?},
                        prev state: {state},
                        state_vec: {:?}",
                            gate_status::unpack_single(status_scalar_pre),
                            gate_status::unpack_single(status_scalar),
                            gate_status::packed_state_vec(status_scalar)
                        );

                        let mut simd_state_vec: Vec<bool> = status_simd
                            .as_array()
                            .iter()
                            .cloned()
                            .map(|s| gate_status::state(s))
                            .collect();

                        res.append(&mut simd_state_vec);

                        assert!(
                            res.windows(2).all(|r| r[0] == r[1]),
                            "SIMD gate evaluators have diffrent behavior:
                        res: {res:?},
                        kind: {kind:?},
                        flags: {flags:?},
                        acc: {acc},
                        prev state: {state}"
                        );

                        let expected_status_delta = if res[0] != state {
                            if res[0] {
                                1
                            } else {
                                (0 as AccType).wrapping_sub(1)
                            }
                        } else {
                            0
                        };
                        assert_eq!(status_delta, expected_status_delta);
                        for delta in status_delta_simd.as_array().iter() {
                            assert_eq!(*delta, expected_status_delta);
                        }
                        for delta in gate_status::unpack_single(status_delta_scalar) {
                            assert_eq!(
                                delta,
                                expected_status_delta,
                                "
packed scalar has wrong value.
got delta: {delta:?},
expected: {expected_status_delta:?}
res: {res:?},
kind: {kind:?},
flags: {flags:?},
acc: {acc},
acc4: {acc_scalar},
acc4_u: {:?},
status_pre: {:?},
status_scalar: {:?},
prev state: {state},
state_vec: {:?}",
                                gate_status::unpack_single(acc_scalar),
                                gate_status::unpack_single(status_scalar_pre),
                                gate_status::unpack_single(status_scalar),
                                gate_status::packed_state_vec(status_scalar),
                            );
                        }
                    }
                }
            }
        }
    }
}
