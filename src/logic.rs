//! logic.rs: Contains the simulaion engine itself.

#![allow(clippy::inline_always)]
//#![allow(dead_code)]

pub mod gate_status;
pub mod network;
pub(crate) use crate::logic::network::GateNetwork;

use crate::logic::network::*;
use itertools::iproduct;
use std::mem::transmute;
use std::simd::{Mask, Simd, SimdPartialEq};

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone, PartialOrd, Ord, Default)]
/// A = active inputs
/// T = total inputs
pub(crate) enum GateType {
    ///A == T
    And,
    ///A > 0
    Or,
    ///A == 0
    Nor,
    ///A != T
    Nand,
    ///A % 2 == 1
    Xor,
    ///A % 2 == 0
    Xnor,
    ///A > 0
    #[default]
    Cluster, // equivalent to OR
}
impl GateType {
    /// guaranteed to activate immediately
    fn will_update_at_start(self) -> bool {
        matches!(self, GateType::Nor | GateType::Nand | GateType::Xnor)
    }
    /// can a pair of identical connections be removed without changing behaviour
    fn can_delete_double_identical_inputs(self) -> bool {
        match self {
            GateType::Xor | GateType::Xnor => true,
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
            GateType::Xor | GateType::Xnor => false,
        }
    }
    fn is_cluster(self) -> bool {
        matches!(self, GateType::Cluster)
    }
}

/// the only cases that matter at the hot code sections
/// for example And/Nor can be evalutated in the same way
/// with an offset to the input count (acc)
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) enum RunTimeGateType {
    OrNand,
    AndNor,
    XorXnor,
}
impl RunTimeGateType {
    fn new(kind: GateType) -> Self {
        match kind {
            GateType::And | GateType::Nor => RunTimeGateType::AndNor,
            GateType::Or | GateType::Nand | GateType::Cluster => RunTimeGateType::OrNand,
            GateType::Xor | GateType::Xnor => RunTimeGateType::XorXnor,
        }
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

type GateKey = (GateType, Vec<IndexType>);

/// data needed after processing network
#[derive(Debug, Clone, Default)]
pub(crate) struct Gate {
    // constant:
    inputs: Vec<IndexType>,  // list of ids
    outputs: Vec<IndexType>, // list of ids
    kind: GateType,

    // variable:
    acc: AccType,
    state: bool,
    in_update_list: bool,
    //TODO: "do not merge" flag for gates that are "volatile", for example handling IO
}
impl Gate {
    fn has_overlapping_outputs(&self, other: &Gate) -> bool {
        iproduct!(self.outputs.iter(), other.outputs.iter()).any(|(&a, &b)| a == b)
    }
    fn has_overlapping_outputs_at_same_index(&self, other: &Gate) -> bool {
        self.outputs
            .iter()
            .zip(other.outputs.iter())
            .any(|(&a, &b)| a == b)
    }
    fn has_overlapping_outputs_at_same_index_with_alignment_8(&self, other: &Gate) -> bool {
        self.outputs
            .iter()
            .zip(other.outputs.iter())
            .any(|(&a, &b)| (a as i32 - b as i32).abs() <= 8)
    }
    fn is_cluster_a_xor_is_cluster_b(&self, other: &Gate) -> bool {
        (self.kind == GateType::Cluster) != (other.kind == GateType::Cluster)
    }
    fn new(kind: GateType, outputs: Vec<IndexType>) -> Self {
        let start_acc = match kind {
            GateType::Xnor => 1,
            _ => 0,
        };
        Gate {
            inputs: Vec::new(),
            outputs,
            acc: start_acc,
            kind,
            state: false, // all gates/clusters initialize to off
            in_update_list: false,
        }
    }
    fn from_gate_type(kind: GateType) -> Self {
        Self::new(kind, Vec::new())
    }
    /// Change number of inputs to handle logic correctly
    /// Can be called multiple times for *different* inputs
    fn add_inputs(&mut self, inputs: i32) {
        let diff: AccType = inputs.try_into().unwrap();
        match self.kind {
            GateType::And | GateType::Nand => self.acc = self.acc.wrapping_sub(diff),
            GateType::Or | GateType::Nor | GateType::Xor | GateType::Xnor | GateType::Cluster => (),
        }
    }
    /// add inputs and handle internal logic for them
    fn add_inputs_vec(&mut self, inputs: &mut Vec<IndexType>) {
        self.add_inputs(inputs.len() as i32);
        self.inputs.append(inputs);
    }
    #[inline]
    #[cfg(test)]
    const fn evaluate(acc: AccType, kind: RunTimeGateType) -> bool {
        match kind {
            RunTimeGateType::OrNand => acc != (0),
            RunTimeGateType::AndNor => acc == (0),
            RunTimeGateType::XorXnor => acc & (1) == (1),
        }
    }
    #[inline]
    #[cfg(test)]
    const fn evaluate_from_flags(acc: AccType, (is_inverted, is_xor): (bool, bool)) -> bool {
        // inverted from perspective of or gate
        // hopefully this generates branchless code.
        if !is_xor {
            (acc != 0) != is_inverted
        } else {
            acc & 1 == 1
        }
    }
    #[inline]
    #[cfg(test)]
    const fn evaluate_branchless(acc: AccType, (is_inverted, is_xor): (bool, bool)) -> bool {
        !is_xor && ((acc != 0) != is_inverted) || is_xor && (acc & 1 == 1)
    }
    #[inline(always)] // inline always required to keep SIMD in registers.
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

    const fn calc_flags(kind: RunTimeGateType) -> (bool, bool) {
        match kind {
            // (is_inverted, is_xor)
            RunTimeGateType::OrNand => (false, false),
            RunTimeGateType::AndNor => (true, false),
            RunTimeGateType::XorXnor => (false, true),
        }
    }

    /// calculate a key that is used to determine if the gate
    /// can be merged with other gates.
    fn calc_key(&self) -> GateKey {
        // TODO: can potentially include inverted.
        // but then every connection would have to include
        // connection information
        let kind = match self.kind {
            GateType::Cluster => GateType::Or,
            _ => self.kind,
        };
        let inputs_len = self.inputs.len();
        let kind = match inputs_len {
            0 | 1 => match kind {
                GateType::Nand | GateType::Xnor | GateType::Nor => GateType::Nor,
                GateType::And | GateType::Or | GateType::Xor | GateType::Cluster => GateType::Or,
            },
            _ => kind,
        };
        (kind, self.inputs.clone())
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct CompiledNetworkInner {
    packed_outputs: Vec<IndexType>,
    packed_output_indexes: Vec<IndexType>,

    //state: Vec<u8>,
    in_update_list: Vec<bool>,
    //runtime_gate_kind: Vec<RunTimeGateType>,
    acc_packed: Vec<gate_status::Packed>,
    acc: Vec<AccType>,

    status_packed: Vec<gate_status::Packed>,
    status: Vec<gate_status::Inner>,
    translation_table: Vec<IndexType>,
    pub iterations: usize,

    //#[cfg(test)]
    kind: Vec<GateType>,
    number_of_gates: usize,
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
#[repr(u8)]
pub enum UpdateStrategy {
    #[default]
    /// Used to compare performance and check correctness.
    Reference = 0,
    /// Pack values inside single register instead of simd
    ScalarSimd = 1,
    /// Update gates with simd
    Simd = 2,
}
impl UpdateStrategy {
    const fn from(value: u8) -> Self {
        match value {
            0 => UpdateStrategy::Reference,
            1 => UpdateStrategy::ScalarSimd,
            2 => UpdateStrategy::Simd,
            _ => panic!(),
        }
    }
}

/// Contains prepared datastructures to run the network.
#[derive(Debug, Default, Clone)]
pub(crate) struct CompiledNetwork<const STRATEGY: u8> {
    pub(crate) i: CompiledNetworkInner,

    update_list: UpdateList,
    cluster_update_list: UpdateList,
}
impl<const STRATEGY_I: u8> CompiledNetwork<STRATEGY_I> {
    const STRATEGY: UpdateStrategy = UpdateStrategy::from(STRATEGY_I);

    //unsafe { transmute::<u8, UpdateStrategy>(STRATEGY_I)};
    #[cfg(test)]
    pub(crate) fn get_acc_test(&self) -> Box<dyn Iterator<Item = u8>> {
        match Self::STRATEGY {
            UpdateStrategy::Reference => Box::new(self.i.acc.clone().into_iter()),
            UpdateStrategy::Simd => Box::new(self.i.acc.clone().into_iter()),
            UpdateStrategy::ScalarSimd => Box::new(
                self.i
                    .acc_packed
                    .clone()
                    .into_iter()
                    .map(|x| gate_status::unpack_single(x))
                    .flatten(),
            ),
        }
    }

    /// Adds all non-cluster gates to update list
    #[cfg(test)]
    pub(crate) fn add_all_to_update_list(&mut self) {
        for (s, k) in self.i.status.iter_mut().zip(self.i.kind.iter()) {
            if *k != GateType::Cluster {
                gate_status::mark_in_update_list(s)
            } else {
                assert!(!gate_status::in_update_list(*s));
            }
        }
        self.update_list.clear();
        self.update_list.collect(
            (0..self.i.number_of_gates as IndexType)
                .into_iter()
                .zip(self.i.kind.iter())
                .filter(|(_, k)| **k != GateType::Cluster)
                .map(|(i, _)| i),
        );
        assert_eq!(self.cluster_update_list.len(), 0);
    }
    fn create(mut network: NetworkWithGaps) -> Self {
        let number_of_gates = network.gates.len();

        let update_list: Vec<IndexType> = network
            .gates
            .iter_mut()
            .enumerate()
            .filter(|(_, gate)| {
                gate.as_ref()
                    .map(|gate| gate.kind.will_update_at_start())
                    .unwrap_or_default()
            })
            .filter_map(|(gate_id, gate)| {
                gate.as_mut().map(|g| {
                    g.in_update_list = true;
                    gate_id as IndexType
                })
            })
            .collect();
        let gates = &network.gates;

        let (packed_output_indexes, packed_outputs) = Self::pack_outputs(gates);
        let runtime_gate_kind: Vec<RunTimeGateType> = gates
            .iter()
            .map(|gate| RunTimeGateType::new(gate.as_ref().map(|g| g.kind).unwrap_or_default()))
            .collect();
        let in_update_list: Vec<bool> = gates
            .iter()
            .map(|gate| gate.as_ref().map(|g| g.in_update_list).unwrap_or_default())
            .collect();
        let state: Vec<u8> = gates
            .iter()
            .map(|gate| u8::from(gate.as_ref().map(|g| g.state).unwrap_or_default()))
            .collect();
        let acc: Vec<u8> = gates
            .iter()
            .map(|gate| gate.as_ref().map(|g| g.acc).unwrap_or_default())
            .collect();
        let status: Vec<gate_status::Inner> = in_update_list
            .iter()
            .zip(state.iter())
            .zip(runtime_gate_kind.iter())
            .map(|((i, s), r)| gate_status::new(*i, *s != 0, *r))
            .collect::<Vec<gate_status::Inner>>();

        let (update_list, in_update_list) = if Self::STRATEGY == UpdateStrategy::ScalarSimd {
            assert_eq!(in_update_list.len() % gate_status::PACKED_ELEMENTS, 0);
            //assert_eq!(update_list.len() % gate_status::PACKED_ELEMENTS, 0);
            let in_update_list: Vec<_> = in_update_list
                .iter()
                .cloned()
                .array_chunks::<{ gate_status::PACKED_ELEMENTS }>()
                .map(|x| x.into_iter().any(|x| x))
                .collect();
            let scalar_update_list: Vec<_> = in_update_list
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, b)| *b)
                .map(|(i, _)| i as IndexType)
                .collect();
            (scalar_update_list, in_update_list)
        } else {
            (update_list, in_update_list)
        };
        let update_list = UpdateList::collect_size(update_list.into_iter(), number_of_gates);

        let status_packed = gate_status::pack(status.iter().copied());
        let acc_packed = gate_status::pack(acc.iter().copied());
        acc_packed
            .iter()
            .copied()
            .flat_map(gate_status::unpack_single)
            .zip(acc.iter().copied())
            .enumerate()
            .for_each(|(i, (a, b))| debug_assert_eq!(a, b, "{a}, {b}, {i}"));

        let mut kind: Vec<GateType> = gates
            .iter()
            .map(|g| g.as_ref().map(|g| g.kind).unwrap_or_default())
            .collect();
        for _ in 0..gate_status::PACKED_ELEMENTS {
            kind.push(GateType::Or);
        }

        let in_update_list: Vec<bool> = (0..acc_packed.len()).map(|_| false).collect();

        Self {
            i: CompiledNetworkInner {
                acc_packed,
                acc,
                packed_outputs,
                packed_output_indexes,
                //state,
                in_update_list,
                //runtime_gate_kind,
                status,
                status_packed,

                iterations: 0,
                translation_table: network.translation_table,
                number_of_gates,
                kind,
            },
            update_list,
            cluster_update_list: UpdateList::new(number_of_gates),
        } //.clone()
    }
    fn pack_outputs(gates: &[Option<Gate>]) -> (Vec<IndexType>, Vec<IndexType>) {
        // TODO: potentially optimized overlapping outputs/indexes
        // (requires 2 pointers/gate)
        // TODO: pack into single array
        let mut packed_output_indexes: Vec<IndexType> = Vec::new();
        let mut packed_outputs: Vec<IndexType> = Vec::new();
        for gate in gates.iter() {
            packed_output_indexes.push(packed_outputs.len().try_into().unwrap());
            if let Some(gate) = gate {
                packed_outputs.append(&mut gate.outputs.clone());
            }
        }
        packed_output_indexes.push(packed_outputs.len().try_into().unwrap());
        (packed_output_indexes, packed_outputs)
    }
    /// # Panics
    /// Not initialized, if `gate_id` is out of range
    #[must_use]
    pub(crate) fn get_state(&self, gate_id: usize) -> bool {
        let gate_id = self.i.translation_table[gate_id];
        match Self::STRATEGY {
            UpdateStrategy::ScalarSimd => {
                gate_status::get_state_from_packed_slice(&self.i.status_packed, gate_id as usize)
            },
            UpdateStrategy::Reference | UpdateStrategy::Simd => {
                gate_status::state(self.i.status[gate_id as usize])
            },
        }
    }
    pub(crate) fn get_state_vec(&self) -> Vec<bool> {
        (0..self.i.translation_table.len())
            .map(|i| self.get_state(i))
            .collect()
    }
    //#[inline(always)]
    //pub(crate) fn update_simd(&mut self) {
    //    self.update_internal();
    //}
    /// Updates state of all gates.
    /// # Panics
    /// Not initialized (debug)
    //#[inline(always)] //<- results in slight regression
    pub(crate) fn update(&mut self) {
        //let iterations = self.i.iterations;
        //let t = Self::STRATEGY;
        //println!("UPDATE START {iterations} {t:?} *****************");
        self.update_internal();
        //println!("UPDATE END   {iterations} {t:?} *****************");
    }

    //#[inline(always)]
    #[inline]
    fn update_internal(&mut self) {
        self.i.iterations += 1;
        // This somehow improves performance, even when update list is non-zero.
        // It should also be very obvious to the compiler...
        //if self.update_list.len() == 0 {
        //    return;
        //}
        self.update_gates::<false>();
        self.update_list.clear();
        self.update_gates::<true>();
        self.cluster_update_list.clear();
    }
    //#[inline(always)]
    #[inline]
    fn update_gates<const CLUSTER: bool>(&mut self) {
        match Self::STRATEGY {
            UpdateStrategy::Simd => {
                Self::update_gates_in_list_simd_wrapper::<CLUSTER>(
                    &mut self.i,
                    &mut self.update_list,
                    &mut self.cluster_update_list,
                );
            },
            UpdateStrategy::Reference => {
                Self::update_gates_in_list_wrapper::<CLUSTER>(
                    &mut self.i,
                    &mut self.update_list,
                    &mut self.cluster_update_list,
                );
            },
            UpdateStrategy::ScalarSimd => {
                Self::update_gates_scalar::<CLUSTER>(
                    &mut self.i,
                    &mut self.update_list,
                    &mut self.cluster_update_list,
                );
            },
        }
    }
    #[inline]
    fn update_gates_in_list_wrapper<const CLUSTER: bool>(
        inner: &mut CompiledNetworkInner,
        gate_update_list: &mut UpdateList,
        cluster_update_list: &mut UpdateList,
    ) {
        let (update_list, next_update_list) = if CLUSTER {
            (unsafe { cluster_update_list.get_slice() }, gate_update_list)
        } else {
            (unsafe { gate_update_list.get_slice() }, cluster_update_list)
        };
        Self::update_gates_in_list::<CLUSTER>(inner, update_list, next_update_list);
    }

    //TODO: Proof of concept, use an update list later
    //TODO: Separation of CLUSTER and non CLUSTER
    #[inline]
    fn update_gates_scalar<const CLUSTER: bool>(
        inner: &mut CompiledNetworkInner,
        gate_update_list: &mut UpdateList,
        cluster_update_list: &mut UpdateList,
    ) {
        let (update_list, next_update_list) = if CLUSTER {
            (unsafe { cluster_update_list.get_slice() }, gate_update_list)
        } else {
            (unsafe { gate_update_list.get_slice() }, cluster_update_list)
        };
        // this updates EVERY gate
        // TODO: unchecked reads.
        // TODO: case where entire group of deltas is zero
        for (id_packed, status_p) in inner.status_packed.iter_mut().enumerate() {
            let acc_p = &inner.acc_packed[id_packed];
            let is_cluster = [0, 1, 2, 3, 4, 5, 6, 7].map(|x| {
                inner.kind[x + id_packed * gate_status::PACKED_ELEMENTS] == GateType::Cluster
            });

            let delta_p =
                gate_status::eval_mut_scalar_masked::<CLUSTER>(status_p, *acc_p, is_cluster);
            if delta_p == 0 {
                continue;
            }
            let packed_output_indexes = &inner.packed_output_indexes;
            let packed_outputs = &inner.packed_outputs;
            let acc_packed = &mut inner.acc_packed;
            Self::propagate_delta_to_accs_scalar(
                delta_p,
                id_packed,
                acc_packed,
                packed_output_indexes,
                packed_outputs,
                |id: IndexType| {
                    let id = id / gate_status::PACKED_ELEMENTS as u32;
                    let id_usize = id as usize;
                    unsafe {
                        if !*(inner.in_update_list).get_unchecked(id_usize) {
                            next_update_list.push(id);
                            *(inner.in_update_list).get_unchecked_mut(id_usize) = true;
                        }
                    }
                },
            );
        }
    }

    /// Update all gates in update list.
    /// Appends next update list.
    #[inline]
    fn update_gates_in_list<const CLUSTER: bool>(
        inner: &mut CompiledNetworkInner,
        update_list: &[IndexType],
        next_update_list: &mut UpdateList,
    ) {
        if update_list.is_empty() {
            return;
        }
        for id in update_list.iter().map(|id| *id as usize) {
            let delta = unsafe {
                gate_status::eval_mut::<CLUSTER>(
                    inner.status.get_unchecked_mut(id),
                    *inner.acc.get_unchecked(id),
                )
            };
            Self::propagate_delta_to_accs(
                id,
                delta,
                &mut inner.acc,
                &inner.packed_output_indexes,
                &inner.packed_outputs,
                |output_id| {
                    let other_status =
                        unsafe { inner.status.get_unchecked_mut(output_id as usize) };
                    if !gate_status::in_update_list(*other_status) {
                        unsafe { next_update_list.push(output_id) };
                        gate_status::mark_in_update_list(other_status);
                    }
                },
            );
        }
    }

    /// NOTE: this assumes that all outputs are non overlapping.
    /// this HAS to be resolved when network is compiled
    /// TODO: `debug_assert` this property
    /// TODO: PERF: unchecked reads
    #[inline(always)]
    fn propagate_delta_to_accs_scalar_simd_ref(
        delta_p: gate_status::Packed,
        id_packed: usize,
        acc_packed: &mut [gate_status::Packed],
        packed_output_indexes: &[IndexType],
        packed_outputs: &[IndexType],
    ) {
        let group_id_offset = id_packed * gate_status::PACKED_ELEMENTS;
        let acc: &mut [u8] = bytemuck::cast_slice_mut(acc_packed);
        let deltas = gate_status::unpack_single(delta_p);
        let deltas_simd = Simd::from_array(deltas);

        let from_index_simd = Simd::from_slice(unsafe {
            packed_output_indexes
                .get_unchecked(group_id_offset..group_id_offset + gate_status::PACKED_ELEMENTS)
        })
        .cast();
        let to_index_simd = Simd::from_slice(unsafe {
            packed_output_indexes.get_unchecked(
                group_id_offset + 1..group_id_offset + gate_status::PACKED_ELEMENTS + 1,
            )
        })
        .cast();

        let mut output_id_index_simd: Simd<usize, _> = from_index_simd;
        //TODO: done if delta = 0
        let mut not_done_mask: Mask<isize, _> = deltas_simd.simd_ne(Simd::splat(0)).into();
        //let mut not_done_mask: Mask<isize, _> = Mask::splat(true);
        loop {
            not_done_mask &= output_id_index_simd.simd_ne(to_index_simd);

            if not_done_mask == Mask::splat(false) {
                break;
            }

            let output_id_simd = unsafe {
                Simd::gather_select_unchecked(
                    packed_outputs,
                    not_done_mask,
                    output_id_index_simd,
                    Simd::splat(0),
                )
            };
            let output_id_simd = output_id_simd.cast();

            let acc_simd = unsafe {
                Simd::gather_select_unchecked(acc, not_done_mask, output_id_simd, Simd::splat(0))
            } + deltas_simd;
            unsafe { acc_simd.scatter_select_unchecked(acc, not_done_mask, output_id_simd) };
            output_id_index_simd += Simd::splat(1);

            //TODO: "add to update list" functionality
        }
    }
    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    fn assert_avx2() {}
    #[cfg(not(target_feature = "avx2"))]
    #[inline(always)]
    fn assert_avx2() {
        #[cfg(not(debug_assertions))]
        panic!("This program was compiled without avx2, which is needed")
    }

    // This uses direct intrinsics instead.
    #[inline(always)]
    fn propagate_delta_to_accs_scalar_simd(
        delta_p: gate_status::Packed,
        id_packed: usize,
        acc_packed: &mut [gate_status::Packed],
        packed_output_indexes: &[IndexType],
        packed_outputs: &[IndexType],
    ) {
        use core::arch::x86_64::*;
        Self::assert_avx2();
        let zero_mm = unsafe { _mm256_setzero_si256() };
        let ones_mm = unsafe { _mm256_set1_epi32(-1) };
        let group_id_offset = id_packed * gate_status::PACKED_ELEMENTS;
        let acc: &mut [u8] = bytemuck::cast_slice_mut(acc_packed);

        // TODO: These reads need to be 256-bit (32 byte) aligned to work with _mm256_load_si256
        // _mm256_loadu_si256 has no alignment requirements
        // TODO: Both of these could be using the aligned variant easily,
        // iff both where separate _mm256 lists
        let from_index_mm = unsafe {
            _mm256_loadu_si256(transmute(
                packed_output_indexes.as_ptr().add(group_id_offset),
            ))
        };
        let to_index_mm = unsafe {
            _mm256_loadu_si256(transmute(
                packed_output_indexes.as_ptr().add(group_id_offset + 1),
            ))
        };
        let mut deltas_mm: __m256i = Simd::from_array(gate_status::unpack_single(delta_p))
            .cast::<u32>()
            .into();
        let mut output_id_index_mm = from_index_mm;

        //TODO: PERF: there is another intrinsic for this.
        let mut not_done_mm =
            unsafe { _mm256_andnot_si256(_mm256_cmpeq_epi32(deltas_mm, zero_mm), ones_mm) };
        // NOTE: this will be diffrent when using intrinsics
        // hopefully the compiler understands that these are constant.
        // using `gather_u32`, I can still use indexes

        loop {
            // check equality: _mm256_cmpeq_epi32
            // bitwise and: _mm256_and_si256
            // bitwise andnot: _mm256_andnot_si256
            let is_index_at_end = unsafe { _mm256_cmpeq_epi32(output_id_index_mm, to_index_mm) };
            not_done_mm = unsafe { _mm256_andnot_si256(is_index_at_end, not_done_mm) };

            //not_done_mm =
            //    unsafe { _mm256_andnot_si256(_mm256_cmpeq_epi32(deltas_mm, zero_mm), ones_mm) };

            // check if mask is zero
            if -1 == unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi32(not_done_mm, zero_mm)) } {
                break;
            }
            // Could use masked gather, but that does not seem to be faster.
            let output_id_mm = unsafe {
                const SCALE: i32 = std::mem::size_of::<u32>() as i32;
                _mm256_i32gather_epi32::<SCALE>(
                    transmute(packed_outputs.as_ptr()),
                    output_id_index_mm,
                )
            };

            //TODO: PERF: this only needs align(4)
            // Increment acc, but only one byte is relevant, so it is masked out.
            let acc_mm = unsafe {
                const SCALE: i32 = std::mem::size_of::<u8>() as i32;
                _mm256_i32gather_epi32::<SCALE>(transmute(acc.as_ptr()), output_id_mm)
            };

            // NOTE: acc is 8 bit

            // TODO: Masking the deltas is now needed for some reason, oops...
            deltas_mm = unsafe { _mm256_and_si256(deltas_mm, not_done_mm) };
            let acc_incremented_mm = unsafe { _mm256_add_epi32(acc_mm, deltas_mm) };

            // There is no scatter on avx2, it therefore has to be done using scalars.

            // NOTE: it is possible to left pack and conditionally write valid elements of vector.
            // https://deplinenoise.files.wordpress.com/2015/03/gdc2015_afredriksson_simd.pdf
            let acc_incremented_cast: [[u8; 4]; 8] = bytemuck::cast(acc_incremented_mm);
            let output_ids: [u32; 8] = bytemuck::cast(output_id_mm);
            let not_done: [i32; 8] = bytemuck::cast(not_done_mm);

            //TODO: problem is caused by shared ids overlapping in the packed array, adding dummy
            // outputs is a potential fix, but may cause memory issues.

            // delta is zero here
            //assert_eq!(deltas[i], 0);
            //let prev_acc = acc_read_using_simd[i][0];
            //let new_acc = acc_incremented_cast[i][0];

            //// delta = 0 => acc is constant.
            //assert_eq!(prev_acc, new_acc);
            //let output_range: [i32; 8] =
            //    bytemuck::cast(unsafe { _mm256_sub_epi32(to_index_mm, from_index_mm) });

            //let acc_in_memory_now = acc[output_ids[i] as usize];
            //assert_eq!(prev_acc, acc_in_memory_now, "output_range: {output_range:?}, {acc_read_using_simd:?}");
            for i in 0..8 {
                let not_done = not_done[i];
                unsafe {
                    std::intrinsics::assume(not_done == 0 || not_done == -1);
                    if not_done != 0 {
                        *acc.get_unchecked_mut(output_ids[i] as usize) = acc_incremented_cast[i][0];
                    };
                }
            }

            output_id_index_mm =
                unsafe { _mm256_add_epi32(output_id_index_mm, _mm256_set1_epi32(1)) };
        }
    }

    /// Reference impl
    #[inline(never)]
    fn propagate_delta_to_accs_scalar<F: FnMut(IndexType)>(
        delta_p: gate_status::Packed,
        id_packed: usize,
        acc_packed: &mut [gate_status::Packed],
        packed_output_indexes: &[IndexType],
        packed_outputs: &[IndexType],
        mut update_list_handler: F,
    ) {
        let group_id_offset = id_packed * gate_status::PACKED_ELEMENTS;
        let acc = bytemuck::cast_slice_mut(acc_packed);
        let deltas = gate_status::unpack_single(delta_p);

        for (id_inner, delta) in deltas.into_iter().enumerate() {
            Self::propagate_delta_to_accs(
                group_id_offset + id_inner,
                delta,
                acc,
                packed_output_indexes,
                packed_outputs,
                &mut update_list_handler,
            );
        }
    }
    #[inline(always)]
    fn propagate_delta_to_accs<F: FnMut(IndexType)>(
        id: usize,
        delta: AccTypeInner,
        acc: &mut [AccTypeInner],
        packed_output_indexes: &[IndexType],
        packed_outputs: &[IndexType],
        update_list_handler: F,
    ) {
        if delta == 0 {
            return;
        }
        let from_index = *unsafe { packed_output_indexes.get_unchecked(id) } as usize;
        let to_index = *unsafe { packed_output_indexes.get_unchecked(id + 1) } as usize;

        for output_id in unsafe { packed_outputs.get_unchecked(from_index..to_index) } {
            let other_acc = unsafe { acc.get_unchecked_mut(*output_id as usize) };
            *other_acc = other_acc.wrapping_add(delta);
        }
        unsafe { packed_outputs.get_unchecked(from_index..to_index) }
            .iter()
            .copied()
            .for_each(update_list_handler);
    }

    #[inline]
    fn update_gates_in_list_simd_wrapper<const CLUSTER: bool>(
        inner: &mut CompiledNetworkInner,
        gate_update_list: &mut UpdateList,
        cluster_update_list: &mut UpdateList,
    ) {
        let (update_list, next_update_list) = if CLUSTER {
            (unsafe { cluster_update_list.get_slice() }, gate_update_list)
        } else {
            (unsafe { gate_update_list.get_slice() }, cluster_update_list)
        };
        Self::update_gates_in_list_simd::<CLUSTER>(inner, update_list, next_update_list);
    }

    #[inline]
    fn update_gates_in_list_simd<const CLUSTER: bool>(
        inner: &mut CompiledNetworkInner,
        update_list: &[IndexType],
        next_update_list: &mut UpdateList,
    ) {
        const LANES: usize = 8;
        let (packed_pre, packed_simd, packed_suf): (
            &[IndexType],
            &[Simd<IndexType, LANES>],
            &[IndexType],
        ) = update_list.as_simd::<LANES>();
        Self::update_gates_in_list::<CLUSTER>(inner, packed_pre, next_update_list);
        Self::update_gates_in_list::<CLUSTER>(inner, packed_suf, next_update_list);

        for id_simd in packed_simd {
            let id_simd_c = id_simd.cast();

            let acc_simd = unsafe {
                Simd::gather_select_unchecked(
                    &inner.acc,
                    Mask::splat(true),
                    id_simd_c,
                    Simd::splat(0),
                )
            };
            let mut status_simd = unsafe {
                Simd::gather_select_unchecked(
                    &inner.status,
                    Mask::splat(true),
                    id_simd_c,
                    Simd::splat(0),
                )
            };
            let delta_simd =
                gate_status::eval_mut_simd::<CLUSTER, LANES>(&mut status_simd, acc_simd);

            unsafe {
                status_simd.scatter_select_unchecked(
                    &mut inner.status,
                    Mask::splat(true),
                    id_simd_c,
                );
            };
            let all_zeroes = delta_simd == Simd::splat(0);
            if all_zeroes {
                continue;
            }
            for (delta, id) in delta_simd
                .as_array()
                .iter()
                .zip((*id_simd).as_array().iter())
                .filter(|(delta, _)| **delta != 0)
                .map(|(delta, id)| (delta, *id as usize))
            {
                let from_index = *unsafe { inner.packed_output_indexes.get_unchecked(id) };
                let to_index = *unsafe { inner.packed_output_indexes.get_unchecked(id + 1) };
                for output_id in unsafe {
                    inner
                        .packed_outputs
                        .get_unchecked(from_index as usize..to_index as usize)
                }
                .iter()
                {
                    let other_acc = unsafe { inner.acc.get_unchecked_mut(*output_id as usize) };
                    *other_acc = other_acc.wrapping_add(*delta);
                    let other_status =
                        unsafe { inner.status.get_unchecked_mut(*output_id as usize) };
                    if !gate_status::in_update_list(*other_status) {
                        unsafe { next_update_list.push(*output_id) };
                        gate_status::mark_in_update_list(other_status);
                    }
                }
            }
        }
    }
}
struct AlignedArray {
    a: [u32; 8],
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gate_evaluation_regression() {
        for (kind, cluster) in [
            (RunTimeGateType::OrNand, true),
            (RunTimeGateType::OrNand, false),
            (RunTimeGateType::AndNor, false),
            (RunTimeGateType::XorXnor, false),
        ] {
            for acc in [
                (0 as AccType).wrapping_sub(2),
                (0 as AccType).wrapping_sub(1),
                0,
                1,
                2,
            ] {
                for state in [true, false] {
                    let flags = Gate::calc_flags(kind);
                    let in_update_list = true;
                    let mut status = gate_status::new(in_update_list, state, kind);
                    let status_delta = if cluster {
                        gate_status::eval_mut::<true>(&mut status, acc)
                    } else {
                        gate_status::eval_mut::<false>(&mut status, acc)
                    };

                    const LANES: usize = 64;
                    let mut status_simd: Simd<gate_status::Inner, LANES> =
                        Simd::splat(gate_status::new(in_update_list, state, kind));
                    let status_delta_simd = if cluster {
                        gate_status::eval_mut_simd::<true, LANES>(
                            &mut status_simd,
                            Simd::splat(acc),
                        )
                    } else {
                        gate_status::eval_mut_simd::<false, LANES>(
                            &mut status_simd,
                            Simd::splat(acc),
                        )
                    };
                    let mut status_scalar =
                        gate_status::splat_u32(gate_status::new(in_update_list, state, kind));
                    let status_scalar_pre = status_scalar;
                    let acc_scalar = gate_status::splat_u32(acc);
                    let status_delta_scalar = if cluster {
                        gate_status::eval_mut_scalar::<true>(&mut status_scalar, acc_scalar)
                    } else {
                        gate_status::eval_mut_scalar::<false>(&mut status_scalar, acc_scalar)
                    };

                    let mut res = vec![
                        Gate::evaluate_from_flags(acc, flags),
                        Gate::evaluate_branchless(acc, flags),
                        Gate::evaluate(acc, kind),
                        gate_status::state(status),
                    ];

                    assert!(
                        res.windows(2).all(|r| r[0] == r[1]),
                        "Some gate evaluators have diffrent behavior:
                        res: {res:?},
                        kind: {kind:?},
                        flags: {flags:?},
                        acc: {acc},
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
