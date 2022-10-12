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
type AccTypeInner = i32;
type AccType = AccTypeInner;

type SimdLogicType = AccTypeInner;

// tests don't need that many indexes, but this is obviously a big limitation.
// u16 enough for typical applications (65536), u32
// u32 > u16, u32
type IndexType = i32; //AccTypeInner;
type UpdateList = crate::raw_list::RawList<IndexType>;

type GateKey = (GateType, Vec<IndexType>);

//TODO: this only uses 4 bits, 2 adjacent gates could share their
//      in_update_list flag and be updated at the same time.
//#[derive(Debug, Clone, Copy)]

//struct GateStatus {
//    inner: u8,
//}

mod gate_status {
    use super::*;
    pub(crate) type Inner = i32;
    pub(crate) type GateStatus = Inner;
    pub(crate) type InnerSigned = i32;
    // bit locations
    const STATE: Inner = 0;
    const IN_UPDATE_LIST: Inner = 1;
    const IS_INVERTED: Inner = 2;
    const IS_XOR: Inner = 3;

    const FLAG_STATE: Inner = 1 << STATE;
    const FLAG_IN_UPDATE_LIST: Inner = 1 << IN_UPDATE_LIST;
    const FLAG_IS_INVERTED: Inner = 1 << IS_INVERTED;
    const FLAG_IS_XOR: Inner = 1 << IS_XOR;

    const FLAGS_MASK: Inner = FLAG_IS_INVERTED | FLAG_IS_XOR;

    //TODO: pub super?
    pub(crate) fn new(in_update_list: bool, state: bool, kind: RunTimeGateType) -> Inner {
        //let in_update_list = in_update_list as u8;
        //let state = state as u8;
        let (is_inverted, is_xor) = Gate::calc_flags(kind);

        ((state as Inner) << STATE)
            | ((in_update_list as Inner) << IN_UPDATE_LIST)
            | ((is_inverted as Inner) << IS_INVERTED)
            | ((is_xor as Inner) << IS_XOR)
    }

    pub(crate) fn flags(inner: &Inner) -> (bool, bool) {
        ((inner >> IS_INVERTED) & 1 != 0, (inner >> IS_XOR) & 1 != 0)
    }

    /// Evaluate and update internal state.
    /// # Returns
    /// Delta (+-1) if state changed (0 = no change)
    /// TODO: assumptions for CLUSTER
    #[inline(always)]
    pub(crate) fn eval_mut<const CLUSTER: bool>(inner_mut: &mut Inner, acc: AccType) -> AccType {
        // <- high, low ->
        //(     3,           2,              1,     0)
        //(is_xor, is_inverted, in_update_list, state)
        // variables are valid for their *first* bit
        let inner = *inner_mut;
        debug_assert!(in_update_list(inner));

        //let inner = self.inner; // 0000XX1X

        let flag_bits = inner & FLAGS_MASK;

        let state_1 = (inner >> STATE) & 1;
        let acc = acc as Inner; // XXXXXXXX
        let new_state_1 = if CLUSTER {
            (acc != 0) as Inner
        } else {
            match flag_bits {
                0 => (acc != 0) as Inner,
                FLAG_IS_INVERTED => (acc == 0) as Inner,
                FLAG_IS_XOR => acc & 1,
                //_ => 0,
                _ => unsafe {
                    debug_assert!(false);
                    std::hint::unreachable_unchecked()
                },
            }
        };
        debug_assert_eq!(new_state_1, {
            let is_xor = inner >> IS_XOR; // 0|1
            debug_assert_eq!(is_xor & 1, is_xor);
            let acc_parity = acc; // XXXXXXXX
            let xor_term = is_xor & acc_parity; // 0|1
            debug_assert_eq!(xor_term & 1, xor_term);
            let acc_not_zero = (acc != 0) as Inner; // 0|1
            let is_inverted = inner >> IS_INVERTED; // XX
            let not_xor = !is_xor; // 0|11111111
            let acc_term = not_xor & (is_inverted ^ acc_not_zero); // XXXXXXXX
            xor_term | (acc_term & 1)
        });

        let state_changed_1 = new_state_1 ^ state_1;

        // automatically sets "in_update_list" bit to zero
        *inner_mut = (new_state_1 << STATE) | flag_bits;

        debug_assert!(!in_update_list(*inner_mut));
        //debug_assert_eq!(expected_new_state, new_state_1 != 0);
        //super::debug_assert_assume(true);
        debug_assert!(state_changed_1 == 0 || state_changed_1 == 1);
        debug_assert!(state_changed_1 < 2);
        //unsafe {
        //    std::intrinsics::assume(state_changed_1 == 0 || state_changed_1 == 1);
        //}

        if state_changed_1 != 0 {
            (new_state_1 << 1).wrapping_sub(1) as AccType
        } else {
            0
        }
    }

    #[inline(always)]
    pub(crate) fn eval_mut_simd<const CLUSTER: bool, const LANES: usize>(
        inner_mut: &mut Simd<Inner, LANES>,
        acc: Simd<AccType, LANES>,
    ) -> Simd<AccType, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let inner = *inner_mut;
        // cannot assert this in simd
        //debug_assert!(in_update_list(inner));

        //let inner = self.inner; // 0000XX1X

        let flag_bits = inner & Simd::splat(FLAGS_MASK);

        let state_1 = (inner >> Simd::splat(STATE)) & Simd::splat(1);
        let acc = acc.cast(); // XXXXXXXX
        let new_state_1: Simd<SimdLogicType, LANES> = if CLUSTER {
            //(acc != 0) as u8
            acc.simd_ne(Simd::splat(0 as SimdLogicType))
                .select(Simd::splat(1), Simd::splat(0))
        } else {
            //match flag_bits {
            //    0 => (acc != 0) as u8,
            //    FLAG_IS_INVERTED => (acc == 0) as u8,
            //    FLAG_IS_XOR => acc & 1,
            //    //_ => 0,
            //    _ => unsafe {
            //        debug_assert!(false);
            //        std::hint::unreachable_unchecked()
            //    },
            //}

            let is_xor = inner >> Simd::splat(IS_XOR); // 0|1
                                                       //debug_assert_eq!(is_xor & Simd::splat(1), is_xor);
            let acc_parity = acc; // XXXXXXXX
            let xor_term = is_xor & acc_parity; // 0|1
                                                //debug_assert_eq!(xor_term & Simd::splat(1), xor_term);
            let acc_not_zero = acc
                .simd_ne(Simd::splat(0))
                .select(Simd::splat(1), Simd::splat(0)); // 0|1
            let is_inverted = inner >> Simd::splat(IS_INVERTED); // XX
            let not_xor = !is_xor; // 0|11111111
            let acc_term = not_xor & (is_inverted ^ acc_not_zero); // XXXXXXXX
            xor_term | (acc_term & Simd::splat(1))
        };

        let state_changed_1 = new_state_1 ^ state_1;

        // TODO: optimize
        let state_changed_1_signed: Simd<InnerSigned, LANES> = state_changed_1.cast();

        //TODO: invert
        let state_changed_mask =
            unsafe { Mask::from_int_unchecked(state_changed_1_signed - Simd::splat(1)) };

        // automatically sets "in_update_list" bit to zero
        *inner_mut = (new_state_1 << Simd::splat(STATE)) | flag_bits;

        // cannot assert this in simd
        //debug_assert!(!in_update_list(*inner_mut));
        //debug_assert_eq!(expected_new_state, new_state_1 != 0);
        //super::debug_assert_assume(true);
        //debug_assert!(state_changed_1 == 0 || state_changed_1 == 1);
        //debug_assert!(state_changed_1 < 2);
        //unsafe {
        //    std::intrinsics::assume(state_changed_1 == 0 || state_changed_1 == 1);
        //}
        state_changed_mask.select(
            Simd::splat(0),
            (new_state_1 << Simd::splat(1)) - Simd::splat(1),
        )

        //if state_changed_1 != 0 {
        //    (new_state_1 << 1).wrapping_sub(1)
        //} else {
        //    0
        //}
        //todo!()
    }

    #[inline(always)]
    pub(crate) fn mark_in_update_list(inner: &mut Inner) {
        *inner |= FLAG_IN_UPDATE_LIST;
    }
    #[inline(always)]
    pub(crate) fn in_update_list(inner: Inner) -> bool {
        inner & FLAG_IN_UPDATE_LIST != 0
    }
    #[inline(always)]
    pub(crate) fn state(inner: Inner) -> bool {
        inner & FLAG_STATE != 0
    }

    #[inline(always)]
    pub(crate) fn state_simd<const LANES: usize>(
        inner: Simd<Inner, LANES>,
    ) -> Mask<InnerSigned, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        (inner & Simd::splat(FLAG_STATE)).simd_ne(Simd::splat(0))
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
    #[inline(always)]
    fn evaluate_from_flags(acc: AccType, (is_inverted, is_xor): (bool, bool)) -> bool {
        // inverted from perspective of or gate
        // hopefully this generates branchless code.
        if !is_xor {
            (acc != 0) != is_inverted
        } else {
            acc & 1 == 1
        }
    }
    #[inline(always)]
    fn evaluate_branchless(acc: AccType, (is_inverted, is_xor): (bool, bool)) -> bool {
        !is_xor && ((acc != 0) != is_inverted) || is_xor && (acc & 1 == 1)
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

#[derive(Debug, Default, Clone)]
struct CompiledNetworkInner {
    packed_outputs: Vec<IndexType>,
    packed_output_indexes: Vec<IndexType>,

    //state: Vec<u8>,
    //in_update_list: Vec<bool>,
    //runtime_gate_kind: Vec<RunTimeGateType>,
    acc: Vec<AccType>,
    kind: Vec<GateType>,

    status: Vec<gate_status::Inner>,
    translation_table: Vec<IndexType>,
    pub iterations: usize,
    number_of_gates: usize,
}

/// Contains prepared datastructures to run the network.
#[derive(Debug, Default, Clone)]
pub(crate) struct CompiledNetwork {
    i: CompiledNetworkInner,

    update_list: UpdateList,
    cluster_update_list: UpdateList,
}
impl CompiledNetwork {
    /// Adds all non-cluster gates to update list
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
        let gate_status: Vec<gate_status::Inner> = in_update_list
            .iter()
            .zip(state.iter())
            .zip(runtime_gate_kind.iter())
            .map(|((i, s), r)| gate_status::new(*i, *s != 0, *r))
            .collect::<Vec<gate_status::Inner>>();
        let kind = gates.iter().map(|g| g.kind).collect();

        Self {
            i: CompiledNetworkInner {
                packed_outputs,
                packed_output_indexes,
                //state,
                acc: gates.iter().map(|gate| gate.acc).collect(),
                //in_update_list,
                //runtime_gate_kind,
                status: gate_status,

                iterations: 0,
                translation_table: network.translation_table,
                number_of_gates,
                kind,
            },
            update_list,
            cluster_update_list: UpdateList::new(number_of_gates),
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
        let gate_id = self.i.translation_table[gate_id];
        //self.state[gate_id as usize] != 0
        gate_status::state(self.i.status[gate_id as usize])
    }
    //#[inline(always)]
    pub(crate) fn update_simd(&mut self) {
        self.update_internal::<true>();
    }
    /// Updates state of all gates.
    /// # Panics
    /// Not initialized (debug)
    //#[inline(always)] //<- results in slight regression
    pub(crate) fn update(&mut self) {
        self.update_internal::<false>();
    }
    //#[inline(always)]
    #[inline(always)]
    fn update_internal<const USE_SIMD: bool>(&mut self) {
        self.i.iterations += 1;
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
    //#[inline(always)]
    #[inline(always)]
    fn update_gates<const CLUSTER: bool, const USE_SIMD: bool>(&mut self) {
        if USE_SIMD {
            Self::update_gates_in_list_simd_wrapper::<CLUSTER>(
                &mut self.i,
                &mut self.update_list,
                &mut self.cluster_update_list,
            );
            //self.update_gates_in_list_simd::<CLUSTER>();
        } else {
            Self::update_gates_in_list_wrapper::<CLUSTER>(
                &mut self.i,
                &mut self.update_list,
                &mut self.cluster_update_list,
            );
        }
    }
    #[inline(always)]
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
    /// Update all gates in update list.
    /// Appends next update list.
    #[inline(always)]
    fn update_gates_in_list<const CLUSTER: bool>(
        inner: &mut CompiledNetworkInner,
        update_list: &[IndexType],
        next_update_list: &mut UpdateList,
    ) {
        //let (update_list, next_update_list) = if CLUSTER {
        //    (unsafe { cluster_update_list.get_slice() }, gate_update_list)
        //} else {
        //    (unsafe { gate_update_list.get_slice() }, cluster_update_list)
        //};
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
                gate_status::eval_mut::<CLUSTER>(
                    inner.status.get_unchecked_mut(id),
                    *inner.acc.get_unchecked(id),
                )
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
                let from_index = *unsafe { inner.packed_output_indexes.get_unchecked(id) };
                let to_index = *unsafe { inner.packed_output_indexes.get_unchecked(id + 1) };
                debug_assert!(from_index <= to_index);
                for output_id in unsafe {
                    inner
                        .packed_outputs
                        .get_unchecked(from_index as usize..to_index as usize)
                }
                .iter()
                {
                    //let in_update_list =
                    //    unsafe { innern_update_list.get_unchecked_mut(*output_id as usize) };
                    let other_acc = unsafe { inner.acc.get_unchecked_mut(*output_id as usize) };
                    *other_acc = other_acc.wrapping_add(delta);
                    let other_status =
                        unsafe { inner.status.get_unchecked_mut(*output_id as usize) };
                    //debug_assert_eq!(*in_update_list, other_status.in_update_list());
                    if !gate_status::in_update_list(*other_status) {
                        //*in_update_list = true;
                        unsafe { next_update_list.push(*output_id) };
                        gate_status::mark_in_update_list(other_status);
                    }
                }
            }
            // this gate should be ready to be re-added to the update list.
            //*unsafe { innern_update_list.get_unchecked_mut(id) } = false;
        }
    }

    #[inline(always)]
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

    #[inline(always)]
    //#[inline(never)]
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
        packed_simd.into_iter().for_each(|packed| {
            Self::update_gates_in_list::<CLUSTER>(inner, packed.as_array(), next_update_list);
        });

        for id_simd in packed_simd {
            debug_assert_eq!(LANES, 8);
            let id_simd = *id_simd;
            let id_simd_c = id_simd.cast();

            //let id_simd_i = unsafe { transmute::<u32x8, i32x8>(id_simd) };
            //let id_simd_m: __m256i = __m256i::from(id_simd_i);

            //let acc_simd = unsafe {
            //    transmute::<i32x8, u32x8>(Simd::from(gather_32x8_to_32x8(&inner.acc, id_simd_m)))
            //};

            let acc_simd = unsafe {
                Simd::gather_select_unchecked(
                    &inner.acc,
                    Mask::splat(true),
                    id_simd_c,
                    Simd::splat(0),
                )
            };
            //let mut status_simd = unsafe {
            //    transmute::<i32x8, u32x8>(Simd::from(gather_32x8_to_32x8(&inner.status, id_simd_m)))
            //};
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
                )
            };
            let all_zeroes = delta_simd == Simd::splat(0);
            if all_zeroes {
                continue;
            }
            //assert!(!all_zeroes);

            let from_index_simd = unsafe {
                Simd::gather_select_unchecked(
                    &inner.packed_output_indexes,
                    Mask::splat(true),
                    id_simd_c,
                    Simd::splat(0),
                )
            };
            let to_index_simd = unsafe {
                Simd::gather_select_unchecked(
                    &inner.packed_output_indexes,
                    Mask::splat(true),
                    id_simd_c + Simd::splat(1),
                    Simd::splat(0),
                )
            };
            for ((delta, from_index), to_index) in delta_simd
                .as_array()
                .into_iter()
                .zip(from_index_simd.as_array().into_iter())
                .zip(to_index_simd.as_array().into_iter())
                .filter(|((delta, _), _)| **delta != 0)
            {
                //let from_index = *unsafe { inner.packed_output_indexes.get_unchecked(id) };
                //let to_index =
                //    *unsafe { inner.packed_output_indexes.get_unchecked(id + 1) };
                for output_id in unsafe {
                    inner
                        .packed_outputs
                        .get_unchecked(*from_index as usize..*to_index as usize)
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

use core::arch::x86_64::__m256i;
use std::mem::transmute;
#[inline(always)]
fn gather_32x8_to_32x8(data: &[u32], indexes: __m256i) -> __m256i {
    assert!(is_x86_feature_detected!("avx2"));
    const SCALE: i32 = 4; // 4 bytes in 32 bit?
    unsafe {
        let ptr = transmute::<*const u32, *const i32>(data.as_ptr());
        core::arch::x86_64::_mm256_i32gather_epi32(ptr, indexes, SCALE)
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
                    let mut status = gate_status::new(in_update_list, state, kind);
                    let status_delta = gate_status::eval_mut::<false>(&mut status, acc);

                    const LANES: usize = 64;
                    let mut status_simd: Simd<gate_status::Inner, LANES> =
                        Simd::splat(gate_status::new(in_update_list, state, kind));
                    let status_delta_simd = gate_status::eval_mut_simd::<false, LANES>(
                        &mut status_simd,
                        Simd::splat(acc),
                    );

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
                }
            }
        }
        //fn eval_mut<const CLUSTER: bool>(&mut self, acc: AccType) -> AccType {
        //fn evaluate_simd<const LANES: usize>(
    }
}
