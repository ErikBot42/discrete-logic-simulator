// logic.rs: contains the simulaion engine itself.
#![allow(clippy::inline_always)]
use itertools::Itertools;
use std::collections::HashMap;
use std::convert::From;
use std::simd::*;

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone, PartialOrd, Ord)]
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
    //OrNand = 0,
    //AndNor = 1,
    //XorXnor = 2,
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

type SimdLogicType = u8;

// tests don't need that many indexes, but this is obviously a big limitation.
// u16 enough for typical applications (65536), u32
// u32 > u16, u32
type IndexType = u32; //AccTypeInner;
type UpdateList = crate::raw_list::RawList<IndexType>;

type GateKey = (GateType, Vec<IndexType>);

//TODO: this only uses 4 bits, 2 adjacent gates could share their
//      in_update_list flag and be updated at the same time.
#[derive(Debug, Clone, Copy)]
struct GateStatus {
    inner: u8,
}
impl GateStatus {
    // bit locations
    const STATE: u8 = 1;
    const IN_UPDATE_LIST: u8 = 0;
    const IS_INVERTED: u8 = 2;
    const IS_XOR: u8 = 3;
    const FLAGS_MASK: u8 = (1 << Self::IS_XOR) | (1 << Self::IS_INVERTED);

    fn new(in_update_list: bool, state: bool, kind: RunTimeGateType) -> Self {
        //let in_update_list = in_update_list as u8;
        //let state = state as u8;
        let (is_inverted, is_xor) = Gate::calc_flags(kind);

        Self {
            inner: ((state as u8) << Self::STATE)
                | ((in_update_list as u8) << Self::IN_UPDATE_LIST)
                | ((is_inverted as u8) << Self::IS_INVERTED)
                | ((is_xor as u8) << Self::IS_XOR),
        }
    }

    fn flags(&self) -> (bool, bool) {
        (
            (self.inner >> Self::IS_INVERTED) & 1 != 0,
            (self.inner >> Self::IS_XOR) & 1 != 0,
        )
    }

    /// Evaluate and update internal state.
    /// # Returns
    /// Delta (+-1) if state changed (0 = no change)
    /// TODO: assumptions for CLUSTER
    #[inline(always)]
    fn eval_mut<const CLUSTER: bool>(&mut self, acc: AccType) -> AccType {
        // <- high, low ->
        //(     3,           2,              1,     0)
        //(is_xor, is_inverted, in_update_list, state)
        // variables are valid for their *first* bit
        debug_assert!(self.in_update_list());
        //let expected_new_state = Gate::evaluate_from_flags(acc, self.flags());

        let inner = self.inner; // 0000XX1X
        let state = inner >> Self::STATE;
        let acc = acc as u8; // XXXXXXXX
        let new_state = if CLUSTER {
            (acc != 0) as u8
        } else {
            let is_xor = inner >> Self::IS_XOR; // 0|1
            let acc_parity = acc; // XXXXXXXX
            let xor_term = is_xor & acc_parity; // 0|1

            let acc_not_zero = (acc != 0) as u8; // 0|1
            let is_inverted = inner >> Self::IS_INVERTED; // XX
            let not_xor = !is_xor; // 0|11111111
            let acc_term = not_xor & (is_inverted ^ acc_not_zero); // XXXXXXXX
            xor_term | acc_term
        };

        let state_changed = new_state ^ state;
        let new_state_1 = new_state & 1;

        // automatically sets "in_update_list" bit to zero
        self.inner = (new_state_1 << Self::STATE) | (self.inner & Self::FLAGS_MASK);

        debug_assert!(!self.in_update_list());
        //debug_assert_eq!(expected_new_state, new_state_1 != 0);

        if state_changed & 1 != 0 {
            (new_state_1 << 1).wrapping_sub(1)
        } else {
            0
        }
    }
    #[inline(always)]
    fn mark_in_update_list(&mut self) {
        self.inner |= 1 << Self::IN_UPDATE_LIST;
    }
    #[inline(always)]
    fn in_update_list(&self) -> bool {
        self.inner & (1 << Self::IN_UPDATE_LIST) != 0
    }
    #[inline(always)]
    fn state(&self) -> bool {
        (self.inner & (1 << Self::STATE)) != 0
    }
}

/// data needed after processing network
#[derive(Debug, Clone)]
pub(crate) struct Gate {
    // constant:
    inputs: Vec<IndexType>,  // list of ids
    outputs: Vec<IndexType>, // list of ids
    kind: GateType,

    // variable:
    acc: AccType,
    state: bool,
    in_update_list: bool,
    //TODO: "do not merge" flag for gates that are "volatile", i.e doing something with IO
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
        !is_xor && ((acc != 0) != is_inverted) || is_xor && (acc % 2 == 1)
    }
    #[inline(always)] // inline always required to keep SIMD in registers.
    #[must_use]
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
        if optimize {
            network = network.optimized();
        }
        assert_ne!(network.gates.len(), 0, "optimization removed all gates");
        return network;
    }
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
    /// O(n)
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
    /// O(n)
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
    /// O(n)
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
    /// O(n)
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
        // Iterate through all old gates.
        // Add gate if type & original input set is unique.
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
                //return std::hint::black_box(new_network.sorted());
                //return new_network.sorted();
                return new_network;
            }
            prev_network_gate_count = new_network.gates.len();
        }
    }

    /// Change order of gates, might be better for cache.
    /// TODO: currently seems to have negative effect
    fn sorted(&self) -> Self {
        //use rand::prelude::*;
        let mut gates_with_ids: Vec<(usize, &Gate)> = self.gates.iter().enumerate().collect();
        //let mut rng = rand::thread_rng();
        //gates_with_ids.shuffle(&mut rng);
        gates_with_ids.sort_by(|(_, a), (_, b)| a.kind.cmp(&b.kind));
        //gates_with_ids.sort_by(|(i, _), (j, _)| i.cmp(&j));
        //gates_with_ids.sort_by(|(i, _), (j, _)| j.cmp(&i));
        let (inverse_translation_table, gates): (Vec<usize>, Vec<&Gate>) =
            gates_with_ids.into_iter().unzip();
        let mut translation_table: Vec<IndexType> = (0..inverse_translation_table.len())
            .map(|_| 0 as IndexType)
            .collect();
        inverse_translation_table
            .iter()
            .enumerate()
            .for_each(|(index, new)| translation_table[*new] = index as IndexType);
        let gates: Vec<Gate> = gates
            .into_iter()
            .cloned()
            .map(|mut gate| {
                gate.outputs
                    .iter_mut()
                    .for_each(|output| *output = translation_table[*output as usize] as IndexType);
                gate.inputs
                    .iter_mut()
                    .for_each(|input| *input = translation_table[*input as usize] as IndexType);
                gate
            })
            .collect();

        Self {
            gates,
            translation_table: Self::create_translation_table(
                &self.translation_table,
                &translation_table,
            ),
        }
    }
}

/// Contains prepared datastructures to run the network.
#[derive(Debug, Default, Clone)]
pub(crate) struct CompiledNetwork {
    packed_outputs: Vec<IndexType>,
    packed_output_indexes: Vec<IndexType>,

    //state: Vec<u8>,
    //in_update_list: Vec<bool>,
    //runtime_gate_kind: Vec<RunTimeGateType>,
    acc: Vec<AccType>,

    gate_status: Vec<GateStatus>,
    update_list: UpdateList,
    cluster_update_list: UpdateList,
    translation_table: Vec<IndexType>,
    pub iterations: usize,
}
impl CompiledNetwork {
    fn create(network: &Network, optimize: bool) -> Self {
        let mut network = network.initialized(optimize);

        let number_of_gates = network.gates.len();
        let update_list = UpdateList::collect(
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
        let gate_status: Vec<GateStatus> = in_update_list
            .iter()
            .zip(state.iter())
            .zip(runtime_gate_kind.iter())
            .map(|((i, s), r)| GateStatus::new(*i, *s != 0, *r))
            .collect::<Vec<GateStatus>>();

        Self {
            packed_outputs,
            packed_output_indexes,
            //state,
            acc: gates.iter().map(|gate| gate.acc).collect(),
            //in_update_list,
            //runtime_gate_kind,
            gate_status,
            update_list,
            cluster_update_list: UpdateList::new(number_of_gates),
            iterations: 0,
            translation_table: network.translation_table,
        } //.clone()
    }
    fn pack_outputs(gates: &Vec<Gate>) -> (Vec<IndexType>, Vec<IndexType>) {
        //TODO: optimized overlapping outputs/indexes
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
        let gate_id = self.translation_table[gate_id];
        //self.state[gate_id as usize] != 0
        self.gate_status[gate_id as usize].state()
    }
    #[inline(always)]
    pub(crate) fn update_simd(&mut self) {
        self.update_internal::<true>();
    }
    /// Updates state of all gates.
    /// # Panics
    /// Not initialized (debug)
    #[inline(always)]
    pub(crate) fn update(&mut self) {
        self.update_internal::<false>();
    }
    fn update_internal<const USE_SIMD: bool>(&mut self) {
        self.iterations += 1;
        // This somehow improves performance, even when update list is non-zero.
        // It should also be very obvious to the compiler...
        if self.update_list.len() == 0 {
            return;
        }
        //TODO: move clear into update functions.
        //TODO: move update list into separate struct to borrow them
        //      separately
        self.update_gates::<false, USE_SIMD>();
        self.update_list.clear();
        self.update_gates::<true, USE_SIMD>();
        self.cluster_update_list.clear();
    }
    fn update_gates<const CLUSTER: bool, const USE_SIMD: bool>(&mut self) {
        if USE_SIMD {
            self.update_gates_in_list_simd::<CLUSTER>();
        } else {
            self.update_gates_in_list::<CLUSTER>();
        }
    }
    /// Update all gates in update list.
    /// Appends next update list.
    #[inline(always)]
    fn update_gates_in_list<const CLUSTER: bool>(&mut self) {
        let (update_list, next_update_list) = if CLUSTER {
            (self.cluster_update_list.get_slice(), &mut self.update_list)
        } else {
            (self.update_list.get_slice(), &mut self.cluster_update_list)
        };
        if update_list.len() == 0 {
            return;
        }
        for id in update_list.iter().map(|id| *id as usize) {
            //debug_assert!(self.in_update_list[id], "{id:?}");
            //let (kind /*flags*/,) = if CLUSTER {
            //    (RunTimeGateType::OrNand /*(false, false)*/,)
            //} else {
            //    (
            //        unsafe { *self.runtime_gate_kind.get_unchecked(id) },
            //        //unsafe { *self.gate_flags.get_unchecked(id) },
            //    )
            //};
            let delta = unsafe {
                self.gate_status
                    .get_unchecked_mut(id)
                    .eval_mut::<CLUSTER>(*self.acc.get_unchecked(id))
            };
            //let next_state_expected = Gate::evaluate(*unsafe { self.acc.get_unchecked(id) }, kind);
            //let state_changed_expected = next_state_expected != (self.state[id] != 0);

            //*unsafe { self.state.get_unchecked_mut(id) } = next_state_expected as u8;
            //debug_assert_eq!(
            //    delta != 0,
            //    state_changed_expected,
            //    "id: {id} status {} acc {} kind {:?} flags1 {:?} flags2 {:?}",
            //    self.gate_status[id].inner,
            //    self.acc[id],
            //    kind,
            //    Gate::calc_flags(kind),
            //    self.gate_status[id].flags(),
            //);
            //let next_state =
            //    Gate::evaluate_from_flags(*unsafe { self.acc.get_unchecked(id) }, flags);
            //let next_state = Gate::evaluate_branchless(*unsafe { acc.get_unchecked(id) }, flags);
            //if (*unsafe { self.state.get_unchecked(id) } != 0) != next_state {
            if delta != 0 {
                //let delta: AccType = ((next_state as AccType) << 1).wrapping_sub(1) as AccType;
                //let delta_expected: AccType = if next_state_expected {
                //    1 as AccTypeInner
                //} else {
                //    (0 as AccTypeInner).wrapping_sub(1 as AccTypeInner)
                //};
                //debug_assert_eq!(delta, delta_expected, "{}", id);
                let from_index = *unsafe { self.packed_output_indexes.get_unchecked(id) };
                let to_index = *unsafe { self.packed_output_indexes.get_unchecked(id + 1) };
                for output_id in unsafe {
                    self.packed_outputs
                        .get_unchecked(from_index as usize..to_index as usize)
                }
                .iter()
                {
                    //let in_update_list =
                    //    unsafe { self.in_update_list.get_unchecked_mut(*output_id as usize) };
                    let other_acc = unsafe { self.acc.get_unchecked_mut(*output_id as usize) };
                    *other_acc = other_acc.wrapping_add(delta);
                    let other_status =
                        unsafe { self.gate_status.get_unchecked_mut(*output_id as usize) };
                    //debug_assert_eq!(*in_update_list, other_status.in_update_list());
                    if !other_status.in_update_list() {
                        //*in_update_list = true;
                        next_update_list.push(*output_id);
                        other_status.mark_in_update_list();
                    }
                }
            }
            // this gate should be ready to be re-added to the update list.
            //*unsafe { self.in_update_list.get_unchecked_mut(id) } = false;
        }
    }
    #[inline(always)]
    //#[inline(never)]
    fn update_gates_in_list_simd<const ASSUME_CLUSTER: bool>(
        &mut self,
        //update_list: &[IndexType],
        //next_update_list: &mut UpdateList,
        //acc: &mut [AccType],
        //state: &mut [u8],
        //in_update_list: &mut [bool],

        //gate_kinds: &[RunTimeGateType],
        //gate_flags: &[(bool, bool)],
        //gate_flag_xor: &[u8],
        //gate_flag_inverted: &[u8],
        //packed_output_indexes: &[IndexType],
        //packed_outputs: &[IndexType],
    ) {
        self.update_gates_in_list::<ASSUME_CLUSTER>();

        //TODO: SIMD: make assumptions when cluster.
        //if update_list.len() == 0 {
        //    return;
        //}
        //const LANES: usize = 16; //16; //16; //16; //TODO: optimize

        //let (packed_pre, packed_simd, packed_suf): (
        //    &[IndexType],
        //    &[Simd<IndexType, LANES>],
        //    &[IndexType],
        //) = update_list.as_simd::<LANES>();
        //Self::update_gates_in_list::<ASSUME_CLUSTER>(
        //    packed_pre,
        //    next_update_list,
        //    acc,
        //    state,
        //    in_update_list,
        //    gate_kinds,
        //    gate_flags,
        //    gate_flag_xor,
        //    gate_flag_inverted,
        //    packed_output_indexes,
        //    packed_outputs,
        //);
        //Self::update_gates_in_list::<ASSUME_CLUSTER>(
        //    packed_suf,
        //    next_update_list,
        //    acc,
        //    state,
        //    in_update_list,
        //    gate_kinds,
        //    gate_flags,
        //    gate_flag_xor,
        //    gate_flag_inverted,
        //    packed_output_indexes,
        //    packed_outputs,
        //);
        //for id_simd in packed_simd {
        //    Self::update_gates_in_list::<ASSUME_CLUSTER>(
        //        id_simd.as_array(),
        //        next_update_list,
        //        acc,
        //        state,
        //        in_update_list,
        //        gate_kinds,
        //        gate_flags,
        //        gate_flag_xor,
        //        gate_flag_inverted,
        //        packed_output_indexes,
        //        packed_outputs,
        //    );
        //}
        //return;

        //for id_simd in packed_simd {
        //    let id_simd_c = id_simd.cast();
        //    let (gate_inverted_simd, gate_xor_simd) = if ASSUME_CLUSTER {
        //        (Simd::splat(0), Simd::splat(0))
        //    } else {
        //        (
        //            Simd::gather_select(
        //                gate_flag_inverted,
        //                Mask::splat(true),
        //                id_simd_c,
        //                Simd::splat(0),
        //            ),
        //            Simd::gather_select(
        //                gate_flag_xor,
        //                Mask::splat(true),
        //                id_simd_c,
        //                Simd::splat(0),
        //            ),
        //        )
        //    };

        //    let acc_simd = Simd::gather_select(acc, Mask::splat(true), id_simd_c, Simd::splat(0));
        //    let state_simd =
        //        Simd::gather_select(state, Mask::splat(true), id_simd_c, Simd::splat(0));
        //    let (new_state_simd, state_changed_simd) = {
        //        Gate::evaluate_simd::<LANES>(
        //            acc_simd,
        //            gate_inverted_simd,
        //            gate_xor_simd,
        //            state_simd,
        //        )
        //    };

        //    // 0 -> -1, 1 -> 1
        //    let delta_simd = (new_state_simd + new_state_simd) - Simd::splat(1);

        //    // new state can be written immediately with simd
        //    // this will write unconditionally, but maybe it would be better
        //    // to only write if state actually changed.
        //    new_state_simd
        //        .cast()
        //        .scatter_select(state, Mask::splat(true), id_simd_c);

        //    // handle outputs (SIMD does not work well here)
        //    for (((id, next_state), delta), state_changed) in id_simd
        //        .to_array()
        //        .into_iter()
        //        .map(|id| id as usize)
        //        .zip(new_state_simd.to_array().into_iter())
        //        .zip(delta_simd.to_array().into_iter())
        //        .zip(state_changed_simd.to_array().into_iter())
        //    {
        //        if state_changed != 0 {
        //            let from_index = packed_output_indexes[id];
        //            let to_index = packed_output_indexes[id + 1];
        //            //let from_index = *unsafe { packed_output_indexes.get_unchecked(id) };
        //            //let to_index = *unsafe { packed_output_indexes.get_unchecked(id + 1) };

        //            //let delta: AccType = if next_state != 0 {
        //            //    1 as AccTypeInner
        //            //} else {
        //            //    (0 as AccTypeInner).wrapping_sub(1 as AccTypeInner)
        //            //};

        //            for output_id in packed_outputs[from_index as usize..to_index as usize]
        //            //unsafe {
        //            //    packed_outputs.get_unchecked(from_index as usize..to_index as usize)
        //            //}
        //            .iter()
        //            {
        //                let in_update_list = in_update_list.get_mut(*output_id as usize).unwrap();
        //                //let in_update_list = unsafe { in_update_list.get_unchecked_mut(*output_id as usize) };
        //                //let other_acc = unsafe { acc.get_unchecked_mut(*output_id as usize) };
        //                let other_acc = acc.get_mut(*output_id as usize).unwrap();
        //                *other_acc = other_acc.wrapping_add(delta as AccType);
        //                if !*in_update_list {
        //                    *in_update_list = true;
        //                    next_update_list.push(*output_id);
        //                }
        //            }
        //        }
        //        *unsafe { in_update_list.get_unchecked_mut(id) } = false; //TODO SIMD
        //    }
        //}
    }
}

#[derive(Debug, Default, Clone)]
pub struct GateNetwork {
    network: Network,
}
impl GateNetwork {
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
    pub(crate) fn compiled(&mut self, optimize: bool) -> CompiledNetwork {
        return CompiledNetwork::create(&self.network, optimize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gate_evaluation_regression() {
        for kind in [
            RunTimeGateType::OrNand,
            RunTimeGateType::AndNor,
            RunTimeGateType::XorXnor,
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
                    let mut status = GateStatus::new(in_update_list, state, kind);
                    let status_delta = status.eval_mut::<false>(acc);
                    let res = [
                        Gate::evaluate_from_flags(acc, flags),
                        Gate::evaluate_branchless(acc, flags),
                        Gate::evaluate(acc, kind),
                        status.state(),
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
                }
            }
        }
        //fn eval_mut<const CLUSTER: bool>(&mut self, acc: AccType) -> AccType {
        //fn evaluate_simd<const LANES: usize>(
    }
}
