// logic.rs: contains the simulaion engine itself.
#![allow(clippy::inline_always)]
use itertools::Itertools;
use no_panic::no_panic;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
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
    Cluster, // equivalent to OR
}
impl GateType {
    /// guaranteed to activate immediately
    fn will_update_at_start(self) -> bool {
        matches!(self, GateType::Nor | GateType::Nand | GateType::Xnor)
    }

    /// can a pair of identical connections be removed without changing behaviour
    fn can_delete_double_identical_inputs(&self) -> bool {
        match self {
            GateType::Xor | GateType::Xnor => true,
            GateType::And | GateType::Or | GateType::Nor | GateType::Nand | GateType::Cluster => {
                false
            },
        }
    }

    /// can one connection in pair of identical connections be removed without changing behaviour
    fn can_delete_single_identical_inputs(&self) -> bool {
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
type AccTypeInner = u32;
type AccType = AccTypeInner;

// tests don't need that many indexes, but this is obviously a big limitation.
// u16 enough for typical applications (65536), u32
// u32 > u16, u32
type IndexType = u32;

type GateKey = (GateType, Vec<IndexType>);

/// data needed after processing network
#[derive(Debug)]
pub(crate) struct Gate {
    // constant:
    inputs: Vec<IndexType>,  // list of ids
    outputs: Vec<IndexType>, // list of ids
    kind: GateType,

    // variable:
    acc: AccType,
    state: bool,
    in_update_list: bool,
    //TODO: "do not merge" flag
}
impl Gate {
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

    #[inline(always)]
    fn evaluate(acc: AccType, kind: RunTimeGateType) -> bool {
        match kind {
            RunTimeGateType::OrNand => acc != (0),
            RunTimeGateType::AndNor => acc == (0),
            RunTimeGateType::XorXnor => acc & (1) == (1),
        }
    }
    fn evaluate_from_flags(acc: AccType, (is_inverted, is_xor): (bool, bool)) -> bool {
        // inverted from perspective of or gate
        // hopefully this generates branchless code.
        if !is_xor {
            (acc != (0)) != is_inverted
        } else {
            acc % (2) == (1)
        }
    }
    fn evaluate_branchless(acc: AccType, (is_inverted, is_xor): (bool, bool)) -> bool {
        !is_xor && ((acc != (0)) != is_inverted) || is_xor && (acc % (2) == (1))
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

/// A list that is just a raw array that is manipulated directly.
/// Very unsafe but slightly faster than a normal vector
#[derive(Debug, Default)]
struct RawList {
    list: Box<[IndexType]>,
    len: usize,
}
impl RawList {
    #[inline(always)]
    fn clear(&mut self) {
        self.len = 0;
    }
    #[inline(always)]
    fn push(&mut self, el: IndexType) {
        *unsafe { self.list.get_unchecked_mut(self.len) } = el;
        self.len += 1;
    }
    #[inline(always)]
    fn get_slice(&self) -> &[IndexType] {
        // &self.list[0..self.len]
        unsafe { self.list.get_unchecked(..self.len) }
    }
    //#[inline(always)]
    //fn len(&self) -> usize {
    //    self.len
    //}
}

/// Contains gate graph in order to do network optimization
/// without mutation.
#[derive(Debug, Default)]
struct Network {
    gates: Vec<Gate>,
    translation_table: Vec<IndexType>,
}
impl Network {
    fn print_info(&self) {
        let counts_iter = self
            .gates
            .iter()
            .map(|x| x.outputs.len())
            .counts()
            .into_iter();
        let mut counts_vec: Vec<(usize, usize)> = counts_iter.collect();
        counts_vec.sort();
        let total_output_connections = counts_vec.iter().map(|(_, count)| count).sum::<usize>();
        println!("output counts total: {total_output_connections}");
        println!("number of outputs: gates with this number of outputs");
        for (value, count) in counts_vec {
            println!("{value}: {count}");
        }
    }

    /// Create input connections for the new gates, given the old gates.
    fn create_input_connections(
        new_gates: &mut Vec<Gate>,
        old_gates: &Vec<Gate>,
        old_to_new_id: &Vec<IndexType>,
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
    fn remove_redundant_input_connections(new_gates: &mut Vec<Gate>) {
        for new_gate in new_gates.iter_mut() {
            new_gate.inputs.sort();
            let new_inputs = &new_gate.inputs;
            let deduped_inputs: &mut Vec<IndexType> = &mut Vec::new();
            for i in 0..new_inputs.len() {
                if let Some(previous) = deduped_inputs.last() {
                    if *previous == new_inputs[i] {
                        if new_gate.kind.can_delete_single_identical_inputs() {
                            continue;
                        } else if new_gate.kind.can_delete_double_identical_inputs() {
                            deduped_inputs.pop();
                            continue;
                        }
                    }
                }
                deduped_inputs.push(new_inputs[i]);
            }
            new_gate.inputs.clear();
            new_gate.add_inputs_vec(&mut deduped_inputs.clone());
        }
    }

    /// Create output connections from current input connections
    fn create_output_connections(new_gates: &mut Vec<Gate>) {
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
    fn create_nodes_optimized_from(old_gates: &Vec<Gate>) -> (Vec<Gate>, Vec<IndexType>) {
        let mut new_gates: Vec<Gate> = Vec::new();
        let mut old_to_new_id: Vec<IndexType> = Vec::new();
        let mut gate_key_to_new_id: HashMap<GateKey, usize> = HashMap::new();
        for (old_gate_id, old_gate) in old_gates.iter().enumerate() {
            let key = old_gate.calc_key();
            let new_id = new_gates.len();
            match gate_key_to_new_id.get(&key) {
                Some(existing_new_id) => {
                    // this gate is same as other, so use other's id.
                    assert!(old_to_new_id.len() == old_gate_id);
                    old_to_new_id.push(*existing_new_id as IndexType);
                    assert!(existing_new_id < &new_gates.len());
                },
                None => {
                    // this gate is new, so a fresh id is created.
                    assert!(old_to_new_id.len() == old_gate_id);
                    old_to_new_id.push(new_id as IndexType);
                    new_gates.push(Gate::from_gate_type(old_gate.kind));
                    gate_key_to_new_id.insert(key, new_id);
                    assert!(new_id < new_gates.len(), "new_id: {new_id}");
                },
            }
        }
        assert!(old_gates.len() == old_to_new_id.len());
        (new_gates, old_to_new_id)
    }

    /// Create translation that combines the old and new translation
    /// from outside facing ids to nodes
    fn create_translation_table(
        old_translation_table: &Vec<IndexType>,
        old_to_new_id: &Vec<IndexType>,
    ) -> Vec<IndexType> {
        old_translation_table
            .clone()
            .into_iter()
            .map(|x| old_to_new_id[x as usize])
            .collect()
    }

    /// Single network optimization pass. Much like compilers,
    /// some passes make it possible for others or the same
    /// pass to be run again.
    fn optimization_pass(&self) -> Self {
        //TODO: split into separate stages.
        // iterate through all old gates,
        // add gate if type & original input set is unique
        // this does not handle the case when an id becomes another id
        // and in turn another merge becomes valid, therefore,
        // this function should be run several times.
        let old_gates = &self.gates;

        let (mut new_gates, old_to_new_id) = Self::create_nodes_optimized_from(old_gates);
        Self::create_input_connections(&mut new_gates, &old_gates, &old_to_new_id);
        Self::remove_redundant_input_connections(&mut new_gates);
        Self::create_output_connections(&mut new_gates);

        let old_translation_table = &self.translation_table;
        let new_translation_table =
            Self::create_translation_table(&old_translation_table, &old_to_new_id);

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
}

#[derive(Debug, Default)]
pub(crate) struct GateNetwork {
    network: Network,

    packed_outputs: Vec<IndexType>,
    packed_output_indexes: Vec<IndexType>,

    state: Vec<bool>,
    acc: Vec<AccType>,
    in_update_list: Vec<bool>,
    runtime_gate_kind: Vec<RunTimeGateType>,

    gate_flags: Vec<(bool, bool)>,

    initialized: bool, //TODO: typestate pattern
    update_list: RawList,
    cluster_update_list: RawList,
}
impl GateNetwork {
    /// Internally creates a vertex.
    /// Returns vertex id
    /// ids of gates are guaranteed to be unique
    /// # Panics
    /// If more than `IndexType::MAX` are added, or after initialized
    pub(crate) fn add_vertex(&mut self, kind: GateType) -> usize {
        assert!(!self.initialized);
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
        assert!(!self.initialized);
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

    #[must_use]
    /// # Panics
    /// Not initialized, if `gate_id` is out of range
    pub(crate) fn get_state(&self, gate_id: usize) -> bool {
        assert!(self.initialized);
        let gate_id = self.network.translation_table[gate_id];
        self.state[gate_id as usize]
    }
    /// Adds all gates to update list and performs initialization
    /// Currently cannot be modified after initialization.
    /// # Panics
    /// Not initialized
    pub(crate) fn init_network(&mut self) {
        assert!(!self.initialized);
        self.network.translation_table = (0..self.network.gates.len())
            .into_iter()
            .map(|x| x as IndexType)
            .collect();
        self.network = self.network.optimized();
        let number_of_gates = self.network.gates.len();

        self.update_list.list = vec![0; number_of_gates].into_boxed_slice();
        self.cluster_update_list.list = vec![0; number_of_gates].into_boxed_slice();
        self.update_list.len = 0;
        self.cluster_update_list.len = 0;

        // TODO: iterator
        for gate_id in 0..number_of_gates {
            let gate = &mut self.network.gates[gate_id];
            let kind = gate.kind;

            // add gates that will immediately update to the
            // update list
            if kind.will_update_at_start() {
                self.update_list.push(gate_id.try_into().unwrap());
                gate.in_update_list = true;
            }

            // pack gate type, acc, state, outputs, flags
            let runtime_kind = RunTimeGateType::new(gate.kind);
            self.runtime_gate_kind.push(runtime_kind);
            self.gate_flags.push(Gate::calc_flags(runtime_kind));

            self.acc.push(gate.acc);
            self.state.push(gate.state);
            self.in_update_list.push(gate.in_update_list);
            self.packed_output_indexes
                .push(self.packed_outputs.len().try_into().unwrap());
            self.packed_outputs.append(&mut gate.outputs.clone());
        }
        self.packed_output_indexes
            .push(self.packed_outputs.len().try_into().unwrap());
        self.initialized = true;
    }

    /// Updates state of all gates.
    /// # Panics
    /// Not initialized (debug)
    #[inline(always)]
    //#[no_panic]
    pub(crate) fn update(&mut self) {
        // Somehow impacts release performance. This should be replaced by type state
        // Will keep anyways
        debug_assert!(self.initialized);

        // This somehow improves performance, even when update list is non-zero.
        // It should also be very obvious to the compiler...
        if self.update_list.len == 0 {
            return;
        }

        Self::update_gates_in_list::<false>(
            &mut self.update_list,
            &mut self.cluster_update_list,
            &mut self.acc,
            &mut self.state,
            &mut self.in_update_list,
            &self.runtime_gate_kind,
            &self.gate_flags,
            &self.packed_output_indexes,
            &self.packed_outputs,
        );
        Self::update_gates_in_list::<true>(
            &mut self.cluster_update_list,
            &mut self.update_list,
            &mut self.acc,
            &mut self.state,
            &mut self.in_update_list,
            &self.runtime_gate_kind,
            &self.gate_flags,
            &self.packed_output_indexes,
            &self.packed_outputs,
        );
    }

    /// Update all gates in update list. Clears current update list and produces a new update list
    #[inline(always)]
    fn update_gates_in_list<const ASSUME_CLUSTER: bool>(
        update_list: &mut RawList,
        next_update_list: &mut RawList,
        acc: &mut [AccType],
        state: &mut [bool],
        in_update_list: &mut [bool],

        gate_kinds: &[RunTimeGateType],
        gate_flags: &[(bool, bool)],
        packed_output_indexes: &[IndexType],
        packed_outputs: &[IndexType],
    ) {
        if update_list.len == 0 {
            return;
        }
        //for id in update_list.get_slice() {
        for id in update_list.get_slice().iter().map(|id| *id as usize) {
            let kind;
            let flags;
            if ASSUME_CLUSTER {
                kind = RunTimeGateType::OrNand;
                flags = (false, false);
            } else {
                kind = *unsafe { gate_kinds.get_unchecked(id) };
                flags = *unsafe { gate_flags.get_unchecked(id) };
            };

            //debug_assert!(in_update_list[*id as usize], "{id:?}");
            let next_state = Gate::evaluate(*unsafe { acc.get_unchecked(id) }, kind);
            //let next_state =
            //    Gate::evaluate_from_flags(*unsafe { acc.get_unchecked(id) }, flags);
            //let next_state = Gate::evaluate_branchless(*unsafe { acc.get_unchecked(id) }, flags);
            if *unsafe { state.get_unchecked(id) } != next_state {
                let delta: AccType = if next_state {
                    1 as AccTypeInner
                } else {
                    (0 as AccTypeInner).wrapping_sub(1 as AccTypeInner)
                };
                let from_index = *unsafe { packed_output_indexes.get_unchecked(id) };
                let to_index = *unsafe { packed_output_indexes.get_unchecked(id + 1) };
                // all, any, find, find map, position
                // map, zip, for_each, enumerate

                for output_id in
                    unsafe { packed_outputs.get_unchecked(from_index as usize..to_index as usize) }
                        .iter()
                {
                    let in_update_list =
                        unsafe { in_update_list.get_unchecked_mut(*output_id as usize) };
                    let other_acc = unsafe { acc.get_unchecked_mut(*output_id as usize) };
                    *other_acc += delta;
                    if !*in_update_list {
                        *in_update_list = true;
                        next_update_list.push(*output_id);
                    }
                }

                *unsafe { state.get_unchecked_mut(id) } = next_state;
            }
            // this gate should be ready to be re-added to the update list.
            *unsafe { in_update_list.get_unchecked_mut(id) } = false;
        }
        update_list.clear();
    }
}
