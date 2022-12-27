//! logic.rs: Contains the simulaion engine itself.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::inline_always)]
//#![allow(dead_code)]

pub mod gate_status;
pub mod network;
pub mod reference_sim;
pub(crate) use crate::logic::network::{GateNetwork, InitializedNetwork};
use bytemuck::cast_slice_mut;
use itertools::Itertools;
use std::mem::{align_of, size_of, transmute};
use std::simd::{Mask, Simd};

pub type ReferenceSim = CompiledNetwork<{ UpdateStrategy::Reference as u8 }>;
pub type SimdSim = CompiledNetwork<{ UpdateStrategy::Simd as u8 }>;
pub type ScalarSim = CompiledNetwork<{ UpdateStrategy::ScalarSimd as u8 }>;
pub type BitPackSim = BitPackSimInner;

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

    /// (`is_inverted`, `is_xor`)
    const fn calc_flags(kind: RunTimeGateType) -> (bool, bool) {
        match kind {
            RunTimeGateType::OrNand => (false, false),
            RunTimeGateType::AndNor => (true, false),
            RunTimeGateType::XorXnor => (false, true),
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

pub trait LogicSim {
    // test: get acc optional
    // test: add all to update list
    /// Create `LogicSim` struct from non-optimized network
    fn create(network: InitializedNetwork) -> Self;
    /// Get state from *internal* id
    fn get_state_internal(&self, gate_id: usize) -> bool;
    /// Number of *external* gates
    fn number_of_gates_external(&self) -> usize;
    /// Run 1 tick
    fn update(&mut self);
    /// translate *external* to *internal* id
    fn to_internal_id(&self, gate_id: usize) -> usize;
    /// Get state from *external* id.
    fn get_state(&self, gate_id: usize) -> bool {
        self.get_state_internal(self.to_internal_id(gate_id))
    }
    /// Return vector of state from *external* perspective.
    fn get_state_vec(&self) -> Vec<bool> {
        (0..self.number_of_gates_external())
            .map(|i| self.get_state(i))
            .collect()
    }
    /// Update network `iterations` times.
    fn update_i(&mut self, iterations: usize) {
        for _ in 0..iterations {
            self.update();
        }
    }
    const STRATEGY: UpdateStrategy;
}

/// data needed after processing network
#[derive(Debug, Clone)]
pub(crate) struct Gate {
    // constant:
    inputs: Vec<IndexType>,  // list of ids
    outputs: Vec<IndexType>, // list of ids
    kind: GateType,

    // variable:
    // acc: AccType,
    initial_state: bool,
    // in_update_list: bool,
    // TODO: "do not merge" flag for gates that are "volatile", for example handling IO
}
impl Gate {
    fn new(kind: GateType) -> Self {
        Gate {
            inputs: Vec::new(),
            outputs: Vec::new(),
            kind,
            initial_state: false,
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
            GateType::Or | GateType::Nor | GateType::Xor | GateType::Cluster => 0,
            GateType::Xnor => 1,
        }
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
        let kind = self.kind;
        let inputs_len = self.inputs.len();
        let kind = match inputs_len {
            0 | 1 => match kind {
                GateType::Nand | GateType::Xnor | GateType::Nor => GateType::Nor,
                GateType::And | GateType::Or | GateType::Xor => GateType::Or,
                GateType::Cluster => GateType::Cluster, // merging cluster is invalid
            },
            _ => kind,
        };
        assert!(self.inputs.is_sorted());
        (kind, self.inputs.clone())
    }
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
    /// Bit manipulation
    BitPack = 3,
}
impl UpdateStrategy {
    const fn from(value: u8) -> Self {
        match value {
            0 => UpdateStrategy::Reference,
            1 => UpdateStrategy::ScalarSimd,
            2 => UpdateStrategy::Simd,
            3 => UpdateStrategy::BitPack,
            _ => panic!(),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct CompiledNetworkInner {
    packed_outputs: Vec<IndexType>,
    packed_output_indexes: Vec<IndexType>,

    //state: Vec<u8>,
    //runtime_gate_kind: Vec<RunTimeGateType>,
    acc_packed: Vec<gate_status::Packed>,
    acc: Vec<AccType>,

    status_packed: Vec<gate_status::Packed>,
    status: Vec<gate_status::Inner>,
    translation_table: Vec<IndexType>,
    pub iterations: usize,

    kind: Vec<GateType>,
    #[cfg(test)]
    number_of_gates: usize,
}

/// Contains prepared datastructures to run the network.
#[derive(Debug)]
pub struct CompiledNetwork<const STRATEGY: u8> {
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
            _ => panic!(),
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
    fn create(network: InitializedNetwork) -> Self {
        let mut network = network.with_gaps(Self::STRATEGY);

        let number_of_gates = network.gates.len();

        let mut in_update_list: Vec<bool> = network.gates.iter().map(|_| false).collect();

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
                gate.as_ref().map(|_| {
                    in_update_list[gate_id] = true;
                    gate_id.try_into().unwrap()
                })
            })
            .collect();
        let gates = &network.gates;

        let (packed_output_indexes, packed_outputs) = Self::pack_outputs(gates);
        let runtime_gate_kind: Vec<RunTimeGateType> = gates
            .iter()
            .map(|gate| RunTimeGateType::new(gate.as_ref().map(|g| g.kind).unwrap_or_default()))
            .collect();
        let state: Vec<u8> = gates
            .iter()
            .map(|gate| u8::from(gate.as_ref().map(|g| g.initial_state).unwrap_or_default()))
            .collect();
        let acc: Vec<u8> = gates
            .iter()
            .map(|gate| gate.as_ref().map(Gate::acc).unwrap_or_default())
            .collect();

        let update_list = UpdateList::collect_size(
            if Self::STRATEGY == UpdateStrategy::ScalarSimd {
                assert_eq!(gates.len() % gate_status::PACKED_ELEMENTS, 0);
                assert_eq!(gates.len(), in_update_list.len());
                let scalar_in_update_list: Vec<_> = in_update_list
                    .iter()
                    .array_chunks::<{ gate_status::PACKED_ELEMENTS }>()
                    .map(|x| x.into_iter().any(|x| *x))
                    .collect();
                let scalar_update_list: Vec<_> = scalar_in_update_list
                    .iter()
                    .enumerate()
                    .filter_map(|(i, b)| b.then_some(i.try_into().unwrap()))
                    .collect();
                scalar_update_list
            } else {
                update_list
            }
            .into_iter(),
            number_of_gates,
        );
        let status: Vec<gate_status::Inner> = in_update_list
            .iter()
            .zip(state.iter())
            .zip(runtime_gate_kind.iter())
            .map(|((i, s), r)| gate_status::new(*i, *s != 0, *r))
            .collect::<Vec<gate_status::Inner>>();
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

        let mut in_update_list: Vec<bool> = (0..number_of_gates).map(|_| false).collect();
        update_list.iter().enumerate().for_each(|(_id, i)| {
            in_update_list[i as usize] = true;
            //dbg!(id);
            //if let Some(i) = in_update_list.get_mut(i as usize) {
            //    dbg!(id);
            //    *i = true;
            //};
        });

        Self {
            i: CompiledNetworkInner {
                acc_packed,
                acc,
                packed_outputs,
                packed_output_indexes,
                //state,
                //runtime_gate_kind,
                status,
                status_packed,

                iterations: 0,
                translation_table: network.translation_table,
                #[cfg(test)]
                number_of_gates,
                kind,
            },
            update_list,
            cluster_update_list: UpdateList::new(number_of_gates),
        }
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

    fn get_state_internal(&self, gate_id: usize) -> bool {
        match Self::STRATEGY {
            UpdateStrategy::ScalarSimd => {
                gate_status::get_state_from_packed_slice(&self.i.status_packed, gate_id)
            },
            UpdateStrategy::Reference | UpdateStrategy::Simd => {
                gate_status::state(self.i.status[gate_id])
            },
            _ => panic!(),
        }
    }
    //#[inline(always)]
    //pub(crate) fn update_simd(&mut self) {
    //    self.update_internal();
    //}
    /// Updates state of all gates.
    /// # Panics
    /// Not initialized (debug)
    #[inline(always)]
    //
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
            _ => panic!(),
        }
    }
    #[inline]
    fn update_gates_in_list_wrapper<const CLUSTER: bool>(
        inner: &mut CompiledNetworkInner,
        gate_update_list: &mut UpdateList,
        cluster_update_list: &mut UpdateList,
    ) {
        let (update_list, next_update_list) = if CLUSTER {
            (cluster_update_list.get_slice(), gate_update_list)
        } else {
            (gate_update_list.get_slice(), cluster_update_list)
        };
        Self::update_gates_in_list::<CLUSTER>(inner, update_list, next_update_list);
    }

    //TODO: Proof of concept, use an update list later
    //TODO: Separation of CLUSTER and non CLUSTER
    #[inline]
    fn update_gates_scalar<const CLUSTER: bool>(
        inner: &mut CompiledNetworkInner,
        _gate_update_list: &mut UpdateList,
        _cluster_update_list: &mut UpdateList,
    ) {
        // this updates EVERY gate
        for (id_packed, status_mut) in inner
            .kind
            .iter()
            .map(|x| (*x == GateType::Cluster) == CLUSTER)
            .step_by(gate_status::PACKED_ELEMENTS)
            .enumerate()
            .zip(inner.status_packed.iter_mut())
            .filter_map(|((i, b), s)| b.then_some((i, s)))
        {
            let delta_p = gate_status::eval_mut_scalar::<CLUSTER>(status_mut, *unsafe {
                inner.acc_packed.get_unchecked(id_packed)
            });
            if delta_p == 0 {
                continue;
            }

            Self::propagate_delta_to_accs_scalar(
                delta_p,
                id_packed,
                &mut inner.acc_packed,
                &inner.packed_output_indexes,
                &inner.packed_outputs,
                |_| {},
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
    /*#[inline(always)]
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
    }*/
    //#[cfg(target_feature = "avx2")]
    //#[inline(always)]
    //fn assert_avx2() {}
    //#[cfg(not(target_feature = "avx2"))]
    //#[inline(always)]
    //fn assert_avx2() {
    //    #[cfg(not(debug_assertions))]
    //    panic!("This program was compiled without avx2, which is needed")
    //}

    // This uses direct intrinsics instead.
    /*
    #[inline(always)]
    fn propagate_delta_to_accs_scalar_simd(
        delta_p: gate_status::Packed,
        id_packed: usize,
        acc_packed: &mut [gate_status::Packed],
        packed_output_indexes: &[IndexType],
        packed_outputs: &[IndexType],
    ) {
        use core::arch::x86_64::{
            __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_andnot_si256, _mm256_cmpeq_epi32,
            _mm256_i32gather_epi32, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi32,
            _mm256_setzero_si256,
        };
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
    }*/

    /// Reference impl
    #[inline(always)]
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
impl<const STRATEGY2: u8> LogicSim for CompiledNetwork<STRATEGY2> {
    fn create(network: InitializedNetwork) -> Self {
        Self::create(network)
    }
    fn get_state_internal(&self, gate_id: usize) -> bool {
        self.get_state_internal(gate_id)
    }
    fn number_of_gates_external(&self) -> usize {
        self.i.translation_table.len()
    }
    #[inline(always)]
    fn update(&mut self) {
        self.update();
    }
    fn to_internal_id(&self, gate_id: usize) -> usize {
        self.i.translation_table[gate_id] as usize
    }
    const STRATEGY: UpdateStrategy = UpdateStrategy::from(STRATEGY2);
}

fn pack_sparse_matrix(
    outputs_list: impl Iterator<Item = Vec<IndexType>>,
) -> (Vec<IndexType>, Vec<IndexType>) {
    // TODO: potentially optimized overlapping outputs/indexes
    // (requires 2 pointers/gate)
    // TODO: pack into single array
    let mut packed_output_indexes: Vec<IndexType> = Vec::new();
    let mut packed_outputs: Vec<IndexType> = Vec::new();
    for mut outputs in outputs_list {
        packed_output_indexes.push(packed_outputs.len().try_into().unwrap());
        packed_outputs.append(&mut outputs);
    }
    packed_output_indexes.push(packed_outputs.len().try_into().unwrap());
    (packed_output_indexes, packed_outputs)
}

fn repack_single_sparse_matrix(
    packed_output_indexes: &[IndexType],
    packed_outputs: &[IndexType],
) -> Vec<IndexType> {
    let offset = packed_output_indexes.len();
    let mut arr: Vec<IndexType> =
        Vec::with_capacity(packed_output_indexes.len() + packed_outputs.len());
    arr.extend(
        packed_output_indexes
            .into_iter()
            .map(|x| *x + IndexType::try_from(offset).unwrap()),
    );
    arr.extend_from_slice(packed_outputs);
    arr
}

#[inline(always)]
fn inline_arr_from_fn<T, const N: usize, F>(mut cb: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    let mut idx = 0;
    [(); N].map(|_| {
        let res = cb(idx);
        idx += 1;
        res
    })
}

#[must_use]
#[inline(always)]
fn bit_set(int: BitInt, index: usize, set: bool) -> BitInt {
    int | ((BitInt::from(set)) << index)
}
#[must_use]
#[inline(always)]
fn bit_get(int: BitInt, index: usize) -> bool {
    int & (1 << index) != 0
}
#[must_use]
#[inline(always)]
fn wrapping_bit_get(int: BitInt, index: usize) -> bool {
    // Truncating semantics desired here.
    int & (1 as BitInt).wrapping_shl(index as u32) != 0
}
fn pack_bits(arr: [bool; BIT_PACK_SIM_BITS]) -> BitInt {
    let mut tmp_int: BitInt = 0;
    for (i, b) in arr.into_iter().enumerate() {
        tmp_int = bit_set(tmp_int, i, b);
    }
    tmp_int
}
type BitAcc = u8;
const ACC_GROUP_SIZE: usize = BIT_PACK_SIM_BITS;
const BIT_PACK_SIM_BITS: usize = BitPackSimInner::BITS;
type BitInt = u64;
#[repr(C)]
#[repr(align(64))] // in bits: 64*8 = 512 bits
#[derive(Debug, Copy, Clone)]
struct BitAccPack([BitAcc; ACC_GROUP_SIZE]);
unsafe impl bytemuck::Zeroable for BitAccPack {}
unsafe impl bytemuck::Pod for BitAccPack {}
const _: () = {
    let size = size_of::<BitAccPack>();
    let align = align_of::<BitAccPack>();
    assert!(size == align, "BitAccPack: size diffrent from alignment");
};

fn bit_acc_pack(arr: [BitAcc; BIT_PACK_SIM_BITS]) -> BitAccPack {
    //BitAccPack::from_le_bytes(arr)
    BitAccPack(arr)
}
#[derive(Debug)]
pub struct BitPackSimInner /*<const LATCH: bool>*/ {
    translation_table: Vec<IndexType>,
    acc: Vec<BitAccPack>, // 8x BitInt
    state: Vec<BitInt>,   // intersperse candidate
    //kind: Vec<GateType>,
    is_xor: Vec<BitInt>,      // intersperse candidate
    is_inverted: Vec<BitInt>, // intersperse candidate
    //packed_output_indexes: Vec<IndexType>,
    //packed_outputs: Vec<IndexType>,
    single_packed_outputs: Vec<IndexType>,

    update_list: UpdateList,
    cluster_update_list: UpdateList,
    in_update_list: Vec<bool>,
    group_output_count: Vec<Option<u8>>, //TODO: better encoding
}
use core::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi8, _mm256_load_si256, _mm256_movemask_epi8, _mm256_setzero_si256,
    _mm256_slli_epi64,
};
impl BitPackSimInner /*<LATCH>*/ {
    const BIT_ACC_GROUP: usize = size_of::<BitAccPack>() / size_of::<BitAcc>();
    const BITS: usize = BitInt::BITS as usize;
    #[inline(always)]
    fn calc_group_id(id: usize) -> usize {
        id / Self::BITS
    }
    /// Reference implementation TODO: TEST
    #[inline(always)] // function used at single call site
    fn _acc_parity(acc: &BitAccPack) -> BitInt {
        let acc: &[BitAcc] = &acc.0;
        let mut acc_parity: BitInt = 0;
        for (i, b) in acc.iter().map(|a| a & 1 == 1).enumerate() {
            acc_parity = bit_set(acc_parity, i, b);
        }
        acc_parity
    }
    /// Reference implementation TODO: TEST
    #[inline(always)] // function used at single call site
    fn _acc_zero(acc: &BitAccPack) -> BitInt {
        let acc: &[BitAcc] = &acc.0;
        let mut acc_zero: BitInt = 0;
        for (i, b) in acc.iter().map(|a| *a != 0).enumerate() {
            acc_zero = bit_set(acc_zero, i, b);
        }
        acc_zero
    }

    /// # SAFETY
    /// Pointer MUST be aligned
    #[inline(always)] // function used at single call site
    unsafe fn acc_parity_m256i(acc_ptr: *const __m256i) -> u32 {
        unsafe {
            let data = _mm256_load_si256(acc_ptr); // load value
            let data = _mm256_slli_epi64::<7>(data); // shift LSB to MSB for each byte
            let data = _mm256_movemask_epi8(data); // put MSB of each byte in an int
            transmute(data)
        }
    }
    #[inline(always)] // function used at single call site
    fn acc_parity_simd(acc: &BitAccPack) -> BitInt {
        unsafe {
            assert!(align_of::<BitAccPack>() >= 32);
            let acc_ptr: *const __m256i = (acc as *const BitAccPack).cast();
            let array: [u32; size_of::<BitAccPack>() / u32::BITS as usize] =
                inline_arr_from_fn(|x| Self::acc_parity_m256i(acc_ptr.add(x)));
            transmute(array) // compiler can statically check size here
        }
    }

    /// # SAFETY
    /// Pointer MUST be aligned
    #[inline(always)] // function used at single call site
    unsafe fn acc_zero_m256i(acc_ptr: *const __m256i) -> u32 {
        unsafe {
            let zero = _mm256_setzero_si256();
            let data = _mm256_load_si256(acc_ptr); // load value
            let data = _mm256_cmpeq_epi8(data, zero); // compare with zero
            let data = _mm256_movemask_epi8(data); // put MSB of each byte in an int
            transmute(!data)
        }
    }
    #[inline(always)] // function used at single call site
    fn acc_zero_simd(acc: &BitAccPack) -> BitInt {
        unsafe {
            assert!(align_of::<BitAccPack>() >= 32);
            let acc_ptr: *const __m256i = (acc as *const BitAccPack).cast();
            let array: [u32; size_of::<BitAccPack>() / u32::BITS as usize] =
                inline_arr_from_fn(|x| Self::acc_zero_m256i(acc_ptr.add(x)));
            transmute(array) // compiler can statically check size here
        }
    }
    /// Reference implmentation TODO: TEST
    #[inline(always)] // function used at single call site
    fn _extract_acc_info(acc: &BitAccPack) -> (BitInt, BitInt) {
        (Self::_acc_zero(acc), Self::_acc_parity(acc))
    }
    #[inline(always)] // function used at single call site
    fn extract_acc_info_simd(acc: &BitAccPack) -> (BitInt, BitInt) {
        (Self::acc_zero_simd(acc), Self::acc_parity_simd(acc))
    }
    // pass by reference intentional to use intrinsics.
    #[inline(always)] // function used at single call site
    fn calc_state_pack<const CLUSTER: bool>(
        acc_p: &BitAccPack,
        is_xor: &BitInt,
        is_inverted: &BitInt,
    ) -> BitInt {
        if CLUSTER {
            Self::acc_zero_simd(acc_p)
        } else {
            let (acc_zero, acc_parity) = Self::extract_acc_info_simd(acc_p);
            ((!is_xor) & (acc_zero ^ is_inverted)) | (is_xor & acc_parity)
        }
    }
    #[inline(always)] // function used at 2 call sites
    fn update_inner<const CLUSTER: bool>(&mut self) {
        let (update_list, next_update_list) = if CLUSTER {
            (&mut self.cluster_update_list, &mut self.update_list)
        } else {
            (&mut self.update_list, &mut self.cluster_update_list)
        };

        static mut AVG_ONES: f64 = 6.0;
        const AVG_ONES_WINDOW: f64 = 4_000_000.0;

        //unsafe { println!("{AVG_ONES}") };
        for (group_id, is_inverted, is_xor) in unsafe { update_list.iter() }
            .map(|g| g as usize)
            .map(|group_id| {
                (
                    group_id,
                    unsafe { self.is_inverted.get_unchecked(group_id) },
                    unsafe { self.is_xor.get_unchecked(group_id) },
                )
            })
        {
            *unsafe { self.in_update_list.get_unchecked_mut(group_id) } = false;
            let state = unsafe { self.state.get_unchecked_mut(group_id) };
            let offset = group_id * Self::BITS;
            //debug_assert_eq!(self.kind[offset] == GateType::Cluster, CLUSTER);
            let new_state = Self::calc_state_pack::<CLUSTER>(
                unsafe { self.acc.get_unchecked(group_id) },
                is_xor,
                is_inverted,
            );
            let changed = *state ^ new_state;
            // println!("{changed:#068b}");
            // println!("{}", changed.count_ones());
            //unsafe {
            //    AVG_ONES = changed.count_ones() as f64 / AVG_ONES_WINDOW
            //        + AVG_ONES * (AVG_ONES_WINDOW - 1.0) / AVG_ONES_WINDOW;
            //}

            if changed == 0 {
                continue;
            }
            *state = new_state;
            Self::propagate_acc(
                changed,
                offset,
                *unsafe { self.group_output_count.get_unchecked(group_id) },
                new_state,
                &self.single_packed_outputs,
                cast_slice_mut(&mut self.acc),
                &mut self.in_update_list,
                next_update_list,
            );
        }

        update_list.clear();
    }

    #[inline(always)]
    fn propagate_acc(
        mut changed: BitInt,
        offset: usize,
        group_output_count: Option<u8>,
        new_state: BitInt,
        single_packed_outputs: &[IndexType],
        acc: &mut [u8],
        in_update_list: &mut [bool],
        next_update_list: &mut UpdateList,
    ) {
        while changed != 0 {
            let i_u32 = changed.trailing_zeros();
            let i_usize = i_u32 as usize;

            let gate_id = offset + i_usize;

            let (outputs_start, outputs_end) = group_output_count.map_or_else(
                || {
                    (
                        *unsafe { single_packed_outputs.get_unchecked(gate_id) } as usize,
                        *unsafe { single_packed_outputs.get_unchecked(gate_id + 1) } as usize,
                    )
                },
                |x| {
                    let base = *unsafe { single_packed_outputs.get_unchecked(offset) } as usize;
                    let x = x as usize;
                    (base + x * i_usize, base + x * (i_usize + 1))
                    //unsafe {
                    //    assert_assume!(
                    //        cached.0 == *single_packed_outputs.get_unchecked(gate_id) as usize
                    //    );
                    //};
                    //unsafe {
                    //    assert_assume!(
                    //        cached.1 == *single_packed_outputs.get_unchecked(gate_id + 1) as usize
                    //    );
                    //};
                },
            );

            // TODO: Store # of outputs in a group if it's constant for the entire group.

            //unsafe {
            //    assert_assume!(outputs_start <= outputs_end);
            //}
            changed &= !(1 << i_u32); // ANY

            let delta = (AccType::from(bit_get(new_state, i_usize)) * 2).wrapping_sub(1);
            //unsafe {
            //    assert_assume!(delta == 1 || delta == 0_u8.wrapping_sub(1));
            //}

            for output in unsafe { single_packed_outputs.get_unchecked(outputs_start..outputs_end) }
                .iter()
                .map(|&i| i as usize)
            // Truncating cast needed for performance
            {
                let output_group_id = BitPackSimInner::calc_group_id(output);

                let acc_mut = unsafe { acc.get_unchecked_mut(output) };
                *acc_mut = acc_mut.wrapping_add(delta);

                let in_update_list_mut =
                    unsafe { in_update_list.get_unchecked_mut(output_group_id) };
                if !*in_update_list_mut {
                    // Truncating cast is needed for performance
                    unsafe { next_update_list.push(output_group_id as IndexType) };
                    *in_update_list_mut = true;
                }
            }
        }
    }
}

impl LogicSim for BitPackSimInner /*<LATCH>*/ {
    fn create(network: InitializedNetwork) -> Self {
        let network = network.prepare_for_bitpack_packing(Self::BITS);
        let number_of_gates_with_padding = network.gates.len();
        assert_eq!(number_of_gates_with_padding % Self::BITS, 0);
        let number_of_buckets = number_of_gates_with_padding / Self::BITS;
        let gates = network.gates;
        let translation_table = network.translation_table;
        let (packed_output_indexes, packed_outputs) = pack_sparse_matrix(
            gates
                .iter()
                .map(|g| g.as_ref().map_or_else(Vec::new, |g| g.outputs.clone())),
        );
        let single_packed_outputs =
            repack_single_sparse_matrix(&packed_output_indexes, &packed_outputs);

        let state: Vec<_> = gates
            .iter()
            .map(|g| g.as_ref().map_or(false, |g| g.initial_state))
            .array_chunks()
            .map(pack_bits)
            .collect();
        assert_eq!(state.len(), number_of_buckets);
        let acc: Vec<_> = gates
            .iter()
            .map(|g| g.as_ref().map_or(0, |g| g.acc() as BitAcc))
            .array_chunks()
            .map(bit_acc_pack)
            .collect();
        let kind: Vec<_> = gates
            .iter()
            .map(|g| g.as_ref().map_or(GateType::Cluster, |g| g.kind))
            .collect();
        let (is_inverted, is_xor): (Vec<_>, Vec<_>) = kind
            .iter()
            .copied()
            .map(RunTimeGateType::new)
            .map(RunTimeGateType::calc_flags)
            .array_chunks::<{ BIT_PACK_SIM_BITS }>()
            .map(|arr| (pack_bits(arr.map(|a| a.0)), pack_bits(arr.map(|a| a.1))))
            .unzip();
        let update_list = UpdateList::collect_size(
            kind.iter()
                .step_by(Self::BITS)
                .map(|k| *k != GateType::Cluster)
                .enumerate()
                .filter_map(|(i, b)| b.then_some(i.try_into().unwrap())),
            number_of_buckets,
        );
        let in_update_list = (0..gates.len()).map(|_| false).collect();
        let cluster_update_list = UpdateList::new(update_list.capacity());

        let group_output_count: Vec<_> = {
            let from = packed_output_indexes.array_chunks::<{ Self::BITS }>();
            let to = packed_output_indexes[1..].array_chunks::<{ Self::BITS }>();
            from.zip(to)
                .map(|(from, to)| {
                    let mut local_output_count = from.iter().zip(to).map(|(&a, &b)| b - a);

                    // # inputs/outputs are almost never beyond 255

                    local_output_count.next().and_then(|cmp_val| {
                        local_output_count
                            .all(|x| x == cmp_val)
                            .then_some(cmp_val)
                            .and_then(|x| x.try_into().ok())
                    })
                })
                .collect()
        };
        dbg!(group_output_count.iter().counts());

        Self {
            translation_table,
            acc,
            state,
            is_xor,
            is_inverted,
            single_packed_outputs,
            update_list,
            cluster_update_list,
            in_update_list,
            group_output_count,
        }
    }
    fn get_state_internal(&self, gate_id: usize) -> bool {
        let index = Self::calc_group_id(gate_id);
        wrapping_bit_get(self.state[index], gate_id)
    }
    fn number_of_gates_external(&self) -> usize {
        self.translation_table.len()
    }
    #[inline(always)] // function used at single call site
    fn update(&mut self) {
        self.update_inner::<false>();
        self.update_inner::<true>();

        //static mut I: usize = 0;

        // inf: 3494, -8.5%
        //
        // 128: 3645, -12.8%
        //
        // 64: 3531,  -14%
        //
        // 32: 3659,  -6.9%
        //
        // 8: 3513,   -11.2%
        //
        // 1/2: 2300

        //unsafe {
        //    I += 1;
        //    I %= 64;
        //    if I == 0 {
        //        self.update_list.get_slice_mut().sort_unstable();
        //    }
        //}
    }
    fn to_internal_id(&self, gate_id: usize) -> usize {
        self.translation_table[gate_id].try_into().unwrap()
    }
    const STRATEGY: UpdateStrategy = UpdateStrategy::BitPack;
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
                    let flags = RunTimeGateType::calc_flags(kind);
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
