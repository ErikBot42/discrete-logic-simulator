//! logic.rs: contains the simulaion engine itself.

#![allow(clippy::inline_always)]
#![allow(dead_code)]

pub mod gate_status;
use core::arch::x86_64::*;
use itertools::{iproduct, Itertools};
use std::collections::HashMap;
use std::mem::transmute;
use std::simd::{LaneCount, Mask, Simd, SimdPartialEq, SupportedLaneCount};
//use std::mem::transmute;

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
        let diff: AccType = inputs as AccTypeInner;
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
        LaneCount<LANES>: SupportedLaneCount,
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

/// Contains gate graph in order to do network optimization
#[derive(Debug, Default, Clone)]
struct Network {
    gates: Vec<Gate>,
    translation_table: Vec<IndexType>,
}
impl Network {
    fn initialized(&self, optimize: bool) -> Self {
        let mut network = self.clone();
        network.translation_table = (0..network.gates.len())
            .into_iter()
            .map(|x| x as IndexType)
            .collect();
        assert_ne!(network.gates.len(), 0, "no gates where added.");
        self.print_info();
        if optimize {
            network = network.optimized();
            self.print_info();
        }
        assert_ne!(network.gates.len(), 0, "optimization removed all gates");
        network
    }
    fn print_info(&self) {
        let counts_iter = self
            .gates
            .iter()
            .map(|x| x.outputs.len())
            .counts()
            .into_iter();
        let mut counts_vec: Vec<(usize, usize)> = counts_iter.collect();
        counts_vec.sort_unstable();
        let total_output_connections = counts_vec.iter().map(|(_, count)| count).sum::<usize>();
        println!("output counts total: {total_output_connections}");
        println!("number of outputs: gates with this number of outputs");
        for (value, count) in counts_vec {
            println!("{value}: {count}");
        }
    }
    /// Create input connections for the new gates, given the old gates.
    /// O(n * k)
    fn create_input_connections(
        new_gates: &mut [Gate],
        old_gates: &[Gate],
        old_to_new_id: &[IndexType],
    ) {
        for (old_gate_id, old_gate) in old_gates.iter().enumerate() {
            let new_id = old_to_new_id[old_gate_id];
            let new_gate: &mut Gate = &mut new_gates[new_id as usize];
            let new_inputs: &mut Vec<IndexType> = &mut old_gate
                .inputs
                .clone()
                .into_iter()
                .map(|x| old_to_new_id[x as usize] as IndexType)
                .collect();
            new_gate.inputs.append(new_inputs);
        }
    }

    /// Remove connections that exist multiple times while
    /// maintaining the circuit behavior.
    /// O(n * k)
    fn remove_redundant_input_connections(new_gates: &mut [Gate]) {
        for new_gate in new_gates.iter_mut() {
            new_gate.inputs.sort_unstable();
            let new_inputs = &new_gate.inputs;
            let deduped_inputs: &mut Vec<IndexType> = &mut Vec::new();
            for new_input in new_inputs {
                if let Some(previous) = deduped_inputs.last() {
                    if *previous == *new_input {
                        if new_gate.kind.can_delete_single_identical_inputs() {
                            continue;
                        } else if new_gate.kind.can_delete_double_identical_inputs() {
                            deduped_inputs.pop();
                            continue;
                        }
                    }
                }
                deduped_inputs.push(*new_input);
            }
            new_gate.inputs.clear();
            new_gate.add_inputs_vec(&mut deduped_inputs.clone());
        }
    }
    /// Create output connections from current input connections
    /// O(n * k)
    fn create_output_connections(new_gates: &mut [Gate]) {
        for gate_id in 0..new_gates.len() {
            let gate = &new_gates[gate_id];
            for i in 0..gate.inputs.len() {
                let gate = &new_gates[gate_id];
                let input_gate_id = gate.inputs[i];
                new_gates[input_gate_id as usize]
                    .outputs
                    .push(gate_id as IndexType);
            }
        }
    }
    /// Create a new merged set of nodes based on the old nodes
    /// and a translation back to the old ids.
    /// O(n)
    fn create_nodes_optimized_from(old_gates: &[Gate]) -> (Vec<Gate>, Vec<IndexType>) {
        let mut new_gates: Vec<Gate> = Vec::new();
        let mut old_to_new_id: Vec<IndexType> = Vec::new();
        let mut gate_key_to_new_id: HashMap<GateKey, usize> = HashMap::new();
        for (old_gate_id, old_gate) in old_gates.iter().enumerate() {
            let key = old_gate.calc_key();
            let new_id = new_gates.len();
            if let Some(existing_new_id) = gate_key_to_new_id.get(&key) {
                // this gate is same as other, so use other's id.
                assert!(old_to_new_id.len() == old_gate_id);
                old_to_new_id.push(*existing_new_id as IndexType);
                assert!(existing_new_id < &new_gates.len());
            } else {
                // this gate is new, so a fresh id is created.
                assert!(old_to_new_id.len() == old_gate_id);
                old_to_new_id.push(new_id as IndexType);
                new_gates.push(Gate::from_gate_type(old_gate.kind));
                gate_key_to_new_id.insert(key, new_id);
                assert!(new_id < new_gates.len(), "new_id: {new_id}");
            }
        }
        assert!(old_gates.len() == old_to_new_id.len());
        (new_gates, old_to_new_id)
    }
    /// Create translation that combines the old and new translation
    /// from outside facing ids to nodes
    /// O(n)
    fn create_translation_table(
        old_translation_table: &[IndexType],
        old_to_new_id: &[IndexType],
    ) -> Vec<IndexType> {
        Self::create_translation_table_min_len(old_translation_table, old_to_new_id, 0)
    }
    fn create_translation_table_min_len(
        old_translation_table: &[IndexType],
        old_to_new_id: &[IndexType],
        min_len: usize,
    ) -> Vec<IndexType> {
        let v: Vec<_> = old_translation_table
            .iter()
            .map(|x| old_to_new_id[*x as usize])
            .chain(old_translation_table.len() as IndexType..min_len as IndexType)
            .collect();
        assert_ge!(v.len(), min_len);
        v
    }
    /// Single network optimization pass. Much like compilers,
    /// some passes make it possible for others or the same
    /// pass to be run again.
    ///
    /// Will completely recreate the network.
    /// O(n * k)
    fn optimization_pass(&self) -> Self {
        // Iterate through all old gates.
        // Add gate if type & original input set is unique.
        let old_gates = &self.gates;
        let (mut new_gates, old_to_new_id) = Self::create_nodes_optimized_from(old_gates);
        Self::create_input_connections(&mut new_gates, old_gates, &old_to_new_id);
        Self::remove_redundant_input_connections(&mut new_gates);
        Self::create_output_connections(&mut new_gates);
        let old_translation_table = &self.translation_table;
        let new_translation_table =
            Self::create_translation_table(old_translation_table, &old_to_new_id);
        Network {
            gates: new_gates,
            translation_table: new_translation_table,
        }
    }
    fn optimized(&self) -> Self {
        let mut prev_network_gate_count = self.gates.len();
        loop {
            let new_network = self.optimization_pass();
            if new_network.gates.len() == prev_network_gate_count {
                return new_network;
            }
            prev_network_gate_count = new_network.gates.len();
        }
    }
    fn optimize_for_scalar(&self) -> Self {
        let sort = |a: &Gate, b: &Gate| {
            a.outputs
                .len()
                .cmp(&b.outputs.len())
                .then(a.kind.cmp(&b.kind))
        };
        // TODO: PERF: reorder outputs to try and fit more outputs in single group
        self.reordered_by(|mut v| {
            v.sort_by(|(_, a), (_, b)| sort(a, b));
            Self::aligned_by_inner(
                v,
                gate_status::PACKED_ELEMENTS,
                Gate::has_overlapping_outputs_at_same_index,
            )
        })
    }

    /// List will have each group of `elements` in such that cmp will return false.
    /// Will also make sure list is a multiple of `elements`
    /// Order is maybe preserved to some extent.
    /// This is just a heuristic, solving it without inserting None is sometimes impossible
    /// Solving it perfectly is probably NP-hard.
    /// `cmp` has no restrictions.
    /// O(n)
    fn aligned_by_inner<F: Fn(&Gate, &Gate) -> bool>(
        mut gates: Vec<(usize, &Gate)>,
        elements: usize,
        cmp: F,
    ) -> Vec<Option<(usize, &Gate)>> {
        let mut current_group: Vec<Option<(usize, &Gate)>> = Vec::new();
        let mut final_list: Vec<Option<(usize, &Gate)>> = Vec::new();
        loop {
            match current_group.len() {
                0 => current_group.push(Some(unwrap_or_else!(gates.pop(), break))),
                n if n == elements => final_list.append(&mut current_group),
                _ => {
                    let mut index = None;
                    'o: for (i, gate) in gates.iter().enumerate().rev() {
                        for cgate in current_group.iter() {
                            if let Some(cgate) = cgate && cmp(gate.1, cgate.1) {
                                continue 'o;
                            }
                        }
                        index = Some(i);
                        break;
                    }
                    current_group.push(index.map(|i| gates.remove(i)));
                },
            }
        }
        assert_eq!(current_group.len(), 0);
        final_list
    }

    /// Change order of gates and update ids afterwards, might be better for cache.
    /// Removing gates is UB, adding None is used to add padding.
    ///
    /// O(n * k) + O(reorder(n, k))
    fn reordered_by<F: FnMut(Vec<(usize, &Gate)>) -> Vec<Option<(usize, &Gate)>>>(
        &self,
        mut reorder: F,
    ) -> Network {
        let gates_with_ids: Vec<(usize, &Gate)> = self.gates.iter().enumerate().collect();

        let gates_with_ids = reorder(gates_with_ids);

        let (inverse_translation_table, gates): (Vec<Option<usize>>, Vec<Option<&Gate>>) =
            gates_with_ids
                .into_iter()
                .map(|o| o.map_or((None, None), |(a, b)| (Some(a), Some(b))))
                .unzip();
        assert_eq_len!(gates, inverse_translation_table);
        let mut translation_table: Vec<IndexType> = (0..inverse_translation_table.len())
            .map(|_| 0 as IndexType)
            .collect();
        assert_eq_len!(gates, translation_table);
        inverse_translation_table
            .iter()
            .enumerate()
            .for_each(|(index, new)| {
                if let Some(new) = new {
                    translation_table[*new] = index as IndexType
                }
            });
        let gates: Vec<Gate> = gates
            .into_iter()
            .map(|gate| {
                if let Some(gate) = gate {
                    let mut gate = gate.clone();
                    gate.outputs.iter_mut().for_each(|output| {
                        *output = translation_table[*output as usize] as IndexType
                    });
                    gate.inputs
                        .iter_mut()
                        .for_each(|input| *input = translation_table[*input as usize] as IndexType);
                    gate
                } else {
                    Gate::default() // filler gate.
                }
            })
            .collect();
        assert_eq_len!(gates, translation_table);
        //assert_le_len!(self.translation_table, translation_table);
        let translation_table =
            Self::create_translation_table(&self.translation_table, &translation_table);
        for t in translation_table.iter() {
            assert_le!(*t as usize, gates.len());
        }
        Self {
            gates,
            translation_table,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct CompiledNetworkInner {
    packed_outputs: Vec<IndexType>,
    packed_output_indexes: Vec<IndexType>,

    //state: Vec<u8>,
    //in_update_list: Vec<bool>,
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

#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum UpdateStrategy {
    #[default]
    Reference = 0,
    ScalarSimd = 1,
    Simd = 2,
}

/// Contains prepared datastructures to run the network.
#[derive(Debug, Default, Clone)]
pub(crate) struct CompiledNetwork<const STRATEGY: u8> {
    pub(crate) i: CompiledNetworkInner,

    update_list: UpdateList,
    cluster_update_list: UpdateList,
}
impl<const STRATEGY_I: u8> CompiledNetwork<STRATEGY_I> {
    const STRATEGY: UpdateStrategy = match STRATEGY_I {
        0 => UpdateStrategy::Reference,
        1 => UpdateStrategy::ScalarSimd,
        2 => UpdateStrategy::Simd,
        _ => panic!(),
    };

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
    fn create(network: &Network, optimize: bool) -> Self {
        let mut network = network.initialized(optimize);
        network = network.optimize_for_scalar();
        //if Self::STRATEGY == UpdateStrategy::ScalarSimd {
        //}

        let number_of_gates = network.gates.len();
        let update_list = UpdateList::collect_size(
            network
                .gates
                .iter_mut()
                .enumerate()
                .filter(|(_, gate)| gate.kind.will_update_at_start())
                .map(|(gate_id, gate)| {
                    gate.in_update_list = true;
                    gate_id as IndexType
                }),
            number_of_gates,
        );
        let gates = &network.gates;
        let (packed_output_indexes, packed_outputs) = Self::pack_outputs(gates);
        let runtime_gate_kind: Vec<RunTimeGateType> = gates
            .iter()
            .map(|gate| RunTimeGateType::new(gate.kind))
            .collect();
        let in_update_list: Vec<bool> = gates.iter().map(|gate| gate.in_update_list).collect();
        let state: Vec<u8> = gates.iter().map(|gate| gate.state as u8).collect();

        let acc: Vec<u8> = gates.iter().map(|gate| gate.acc).collect();
        dbg!(gates.iter().map(|gate| gate.acc as i8).collect::<Vec<i8>>());

        let status: Vec<gate_status::Inner> = in_update_list
            .iter()
            .zip(state.iter())
            .zip(runtime_gate_kind.iter())
            .map(|((i, s), r)| gate_status::new(*i, *s != 0, *r))
            .collect::<Vec<gate_status::Inner>>();

        let status_packed = gate_status::pack(status.iter().copied());
        let acc_packed = gate_status::pack(acc.iter().copied());
        dbg!(acc_packed
            .iter()
            .copied()
            .map(gate_status::unpack_single)
            .collect::<Vec<[u8; gate_status::PACKED_ELEMENTS]>>());

        acc_packed
            .iter()
            .cloned()
            .flat_map(gate_status::unpack_single)
            .zip(acc.iter().copied())
            .enumerate()
            .for_each(|(i, (a, b))| debug_assert_eq!(a, b, "{a}, {b}, {i}"));

        let mut kind: Vec<GateType> = gates.iter().map(|g| g.kind).collect();
        kind.push(GateType::Or);
        kind.push(GateType::Or);
        kind.push(GateType::Or);
        kind.push(GateType::Or);

        Self {
            i: CompiledNetworkInner {
                acc_packed,
                acc,
                packed_outputs,
                packed_output_indexes,
                //state,
                //in_update_list,
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
    fn pack_outputs(gates: &[Gate]) -> (Vec<IndexType>, Vec<IndexType>) {
        // TODO: potentially optimized overlapping outputs/indexes
        // (requires 2 pointers/gate)
        // TODO: pack into single array
        let mut packed_output_indexes: Vec<IndexType> = Vec::new();
        let mut packed_outputs: Vec<IndexType> = Vec::new();
        for gate in gates.iter() {
            packed_output_indexes.push(packed_outputs.len().try_into().unwrap());
            packed_outputs.append(&mut gate.outputs.clone());
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
                Self::update_gates_scalar::<CLUSTER>(&mut self.i);
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
    fn update_gates_scalar<const CLUSTER: bool>(inner: &mut CompiledNetworkInner) {
        // this updates EVERY gate
        // TODO: unchecked reads.
        // TODO: case where entire group of deltas is zero
        for (id_packed, status_p) in inner.status_packed.iter_mut().enumerate() {
            let acc_p = &inner.acc_packed[id_packed];
            let is_cluster = [0, 1, 2, 3, 4, 5, 6, 7].map(|x| {
                inner.kind[x + id_packed * gate_status::PACKED_ELEMENTS] == GateType::Cluster
            });

            //let is_cluster = [CLUSTER; 8];

            let delta_p =
                gate_status::eval_mut_scalar_masked::<CLUSTER>(status_p, *acc_p, is_cluster);
            if delta_p == 0 {
                continue;
            }

            let packed_output_indexes = &inner.packed_output_indexes;
            let packed_outputs = &inner.packed_outputs;
            let acc_packed = &mut inner.acc_packed;
            Self::propagate_delta_to_accs_scalar_simd(
                delta_p,
                id_packed,
                acc_packed,
                packed_output_indexes,
                packed_outputs,
            );
        }
    }

    /// Update all gates in update list.
    /// Appends next update list.
    #[inline(never)]
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
    #[inline(never)]
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
    // This uses direct intrinsics instead.
    #[inline(never)]
    fn propagate_delta_to_accs_scalar_simd(
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

        //TODO: These reads need to be 256-bit (32 byte) aligned to work with _mm256_load_si256
        // _mm256_loadu_si256 has no alignment requirements
        let from_index_mm = unsafe {
            _mm256_loadu_si256(transmute(
                packed_output_indexes
                    .as_ptr()
                    .offset(group_id_offset as isize),
            ))
        };
        let from_index_simd: Simd<u32, 8> = from_index_mm.into();
        let to_index_mm = unsafe {
            _mm256_loadu_si256(transmute(
                packed_output_indexes
                    .as_ptr()
                    .offset(group_id_offset as isize + 1),
            ))
        };
        let to_index_simd: Simd<u32, 8> = to_index_mm.into();

        let mut output_id_index_simd: Simd<u32, _> = from_index_simd;
        let mut not_done_mask: Mask<i32, _> = deltas_simd.simd_ne(Simd::splat(0)).into();
        //let mut not_done_mask: Mask<isize, _> = Mask::splat(true);
        loop {
            not_done_mask &= output_id_index_simd.simd_ne(to_index_simd);

            if not_done_mask == Mask::splat(false) {
                break;
            }

            let output_id_simd = unsafe {
                Self::gather_select_unchecked_u32(
                    packed_outputs,
                    not_done_mask,
                    output_id_index_simd,
                )
            };
            let output_id_simd = output_id_simd.cast();

            let acc_simd = unsafe {
                Simd::gather_select_unchecked(
                    acc,
                    not_done_mask.cast(),
                    output_id_simd,
                    Simd::splat(0),
                )
            } + deltas_simd;
            unsafe { acc_simd.scatter_select_unchecked(acc, not_done_mask.cast(), output_id_simd) };
            output_id_index_simd += Simd::splat(1);
        }
    }

    unsafe fn gather_select_unchecked_u32(
        slice: &[u32],
        enable: Mask<i32, 8>,
        idxs: Simd<u32, 8>,
    ) -> Simd<u32, 8> {
        const SCALE: i32 = std::mem::size_of::<u32>() as i32;
        unsafe {
            _mm256_mask_i32gather_epi32::<SCALE>(
                _mm256_setzero_si256(),
                transmute(slice.as_ptr()),
                transmute::<Simd<u32, _>, Simd<i32, _>>(idxs).into(),
                transmute::<Mask<i32, _>, __m256i>(enable.into()),
            )
            .into()
        }
    }

    /// Reference impl
    #[inline(never)]
    fn propagate_delta_to_accs_scalar(
        delta_p: gate_status::Packed,
        id_packed: usize,
        acc_packed: &mut [gate_status::Packed],
        packed_output_indexes: &[IndexType],
        packed_outputs: &[IndexType],
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
                |_| {},
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
            .into_iter()
            .cloned()
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

#[derive(Debug, Default, Clone)]
pub struct GateNetwork<const STRATEGY: u8> {
    network: Network,
}
impl<const STRATEGY: u8> GateNetwork<STRATEGY> {
    /// Internally creates a vertex.
    /// Returns vertex id
    /// ids of gates are guaranteed to be unique
    /// # Panics
    /// If more than `IndexType::MAX` are added, or after initialized
    pub(crate) fn add_vertex(&mut self, kind: GateType) -> usize {
        let next_id = self.network.gates.len();
        self.network.gates.push(Gate::from_gate_type(kind));
        assert!(self.network.gates.len() < IndexType::MAX as usize);
        next_id
    }

    /// Add inputs to `gate_id` from `inputs`.
    /// Connection must be between cluster and a non cluster gate
    /// and a connection can only be made once for a given pair of gates.
    /// # Panics
    /// if precondition is not held.
    pub(crate) fn add_inputs(&mut self, kind: GateType, gate_id: usize, inputs: Vec<usize>) {
        let gate = &mut self.network.gates[gate_id];
        gate.add_inputs(inputs.len().try_into().unwrap());
        let mut in2 = Vec::new();
        for input in &inputs {
            in2.push((*input).try_into().unwrap());
        }
        gate.inputs.append(&mut in2);
        gate.inputs.sort_unstable();
        gate.inputs.dedup();
        for input_id in inputs {
            assert!(
                input_id < self.network.gates.len(),
                "Invalid input index {input_id}"
            );
            assert_ne!(
                (kind == GateType::Cluster),
                (self.network.gates[input_id].kind == GateType::Cluster),
                "Connection was made between cluster and non cluster for gate {gate_id}"
            );
            // panics if it cannot fit in IndexType
            self.network.gates[input_id]
                .outputs
                .push(gate_id.try_into().unwrap());
            self.network.gates[input_id].outputs.sort_unstable();
            self.network.gates[input_id].outputs.dedup();
        }
    }

    /// Adds all gates to update list and performs initialization
    /// Currently cannot be modified after initialization.
    /// # Panics
    /// Already initialized
    #[must_use]
    pub(crate) fn compiled(&self, optimize: bool) -> CompiledNetwork<{ STRATEGY }> {
        CompiledNetwork::create(&self.network, optimize)
    }
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
