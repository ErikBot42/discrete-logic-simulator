// logic.rs: contains the simulaion engine itself.
// use std::collections::BTreeSet;
// use std::collections::HashSet;
// use std::collections::HashMap;
#![allow(clippy::inline_always)]
use std::collections::HashMap;

//use bitvec::prelude::*;


#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub(crate) enum GateType {
    And,
    Or,
    Nor,
    Nand,
    Xor,
    Xnor,
    Cluster, // equivalent to OR
}

impl GateType {
    /// guaranteed to activate immediately
    fn is_inverted(self) -> bool {
        matches!(self, GateType::Nor | GateType::Nand | GateType::Xnor)
    }
}


/// the only cases that matter at the hot code sections
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) enum RunTimeGateType {
    OrNand,
    AndNor,
    XorXnor,
}
impl RunTimeGateType {
    fn new(kind: GateType) -> Self {
        match kind {
            GateType::And | GateType::Nor                      => RunTimeGateType::AndNor,
            GateType::Or  | GateType::Nand | GateType::Cluster => RunTimeGateType::OrNand,
            GateType::Xor | GateType::Xnor                     => RunTimeGateType::XorXnor,
        }
    }
}

// speed: u8 < u16 = u32
// Gates only care about this equalling 0 or parity, ergo signed/unsigned is irrelevant.
// let n = 1 << bits_in_value(AccType)
// Or, Nand: max active: n
// Nor, And: max inactive: n
// Xor, Xnor: no limitation
type AccType = u16;

// tests don't need that many indexes, but this is obviously a big limitation.
// u16 enough for typical applications (65536), u32
// u32 < u16
type IndexType = u32;

/// data needed after processing network
#[derive(Debug)]
pub(crate) struct Gate {
    // constant:
    inputs: Vec<IndexType>, // list of ids
    outputs: Vec<IndexType>, // list of ids
    kind: GateType,

    // variable:
    acc: AccType, 
    state: bool,
    in_update_list: bool,
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
        let diff: AccType = inputs.try_into().unwrap();
        match self.kind {
            GateType::And | GateType::Nand 
                => self.acc = self.acc.wrapping_sub(diff),
                GateType::Or | GateType::Nor | GateType::Xor | GateType::Xnor | GateType::Cluster
                    => (),
        }
    }

    #[inline(always)]
    fn evaluate_from_runtime_static(acc: AccType, kind:RunTimeGateType) -> bool {
        match kind {
            RunTimeGateType::OrNand  => acc != 0,
            RunTimeGateType::AndNor  => acc == 0,
            RunTimeGateType::XorXnor => acc & 1 == 1,
        } 
    }


    fn generate_gate_key(&self) -> (GateType, Vec<IndexType>) {
        let mut kind = self.kind;
        let inputs = self.inputs.clone();
        let input_count = self.inputs.len();

        let buffer_gate = GateType::Or;
        let nor_gate = GateType::Nor;

        if input_count < 2 {kind = match kind {
            GateType::And => buffer_gate,
                GateType::Or => buffer_gate,
                GateType::Nor => nor_gate,
                GateType::Nand => nor_gate,
                GateType::Xor => buffer_gate,
                GateType::Xnor => nor_gate,
                GateType::Cluster => buffer_gate}
        }

        (kind, inputs)
    }
}

// a list that is just a raw array that is manipulated directly.
#[derive(Debug, Default)]
struct RawList {
    list: Box<[IndexType]>,
    len: usize,
}
impl RawList {
    #[inline(always)]
    fn clear(&mut self) {self.len = 0;}
    #[inline(always)]
    fn push(&mut self, el: IndexType) {
        //self.list[self.len] = el;
        unsafe{*self.list.get_unchecked_mut(self.len) = el;}
        self.len += 1;
    }
    // (hopefully) branchless push
    #[inline(always)]
    fn push_if(&mut self, el: IndexType, push: bool) {
        unsafe{*self.list.get_unchecked_mut(self.len) = el;}
        self.len += if push {1} else {0};
    }
    #[inline(always)]
    fn get_slice(&self) -> &[IndexType] {
        &self.list[0..self.len]
    }
}

#[derive(Debug, Default)]
pub(crate) struct GateNetwork {
    //TODO: bitvec
    gates: Vec<Gate>,
    //clusters: Vec<Gate>,

    packed_outputs: Vec<IndexType>,
    packed_output_indexes: Vec<IndexType>,
    state: Vec<bool>,
    acc: Vec<AccType>,
    in_update_list: Vec<bool>,
    runtime_gate_kind: Vec<RunTimeGateType>,
    initialized: bool,
    
    update_list: RawList,
    cluster_update_list: RawList,
}
// TODO: only add to update list if state will change?
// TODO: add layer after cluster directly?
// TODO: atomics do not seem to significantly impact performance, therefore, they could be used.

impl GateNetwork {
    /// Internally creates a vertex.
    /// Returns vertex id
    /// ids of gates are guaranteed to be unique
    /// # Panics
    /// If more than `IndexType::MAX` are added, or after initialized 
    pub(crate) fn add_vertex(&mut self, kind: GateType) -> usize {
        assert!(!self.initialized);
        let next_id = self.gates.len();
        self.gates.push(Gate::from_gate_type(kind));
        assert!(self.gates.len() < IndexType::MAX as usize);
        next_id
    }

    /// Add inputs to `gate_id` from `inputs`.
    /// Connection must be between cluster and a non cluster gate 
    /// and a connection can only be made once for a given pair of gates.
    /// # Panics
    /// if precondition is not held.
    pub(crate) fn add_inputs(&mut self, kind: GateType, gate_id: usize, inputs: Vec<usize>) {
        assert!(!self.initialized);
        //debug_assert!(inputs.len()!=0);//TODO: remove me
        let gate = &mut self.gates[gate_id];
        gate.add_inputs(inputs.len().try_into().unwrap());
        let mut in2 = Vec::new();
        for input in &inputs {
            in2.push((*input).try_into().unwrap());
        }
        gate.inputs.append(&mut in2);
        gate.inputs.sort_unstable();
        gate.inputs.dedup();
        for input_id in inputs {
            assert!(input_id<self.gates.len(), "Invalid input index {input_id}");
            assert_ne!((kind == GateType::Cluster),(self.gates[input_id].kind == GateType::Cluster), "Connection was made between cluster and non cluster for gate {gate_id}");
            // panics if it cannot fit in IndexType
            self.gates[input_id].outputs.push(gate_id.try_into().unwrap());
            self.gates[input_id].outputs.sort_unstable();
            // TODO: add this back.
            //for output in &self.gates[input_id].outputs {
            //    assert_ne!(*output,gate_id as IndexType, "Connection was made multiple times for gate {gate_id} to gate {output}");
            //}
        }
    }

    #[must_use]
    /// # Panics
    /// Not initialized, if `gate_id` is out of range
    pub(crate) fn get_state(&self, gate_id: usize) -> bool {
        assert!(self.initialized);
        //self.gates[gate_id].state
        self.state[gate_id]
            //self.gate_flags[gate_id].state()
    }
    #[inline(never)]
    /// # Panics
    /// Not initialized
    /// pre: on first update, the list only contains gates that will change.
    pub(crate) fn update(&mut self) {
        assert!(self.initialized); // assert because cheap
                                   // TODO: allow gate to add to "wrong" update list
                                   // after network optimization
        for gate_id in self.update_list.get_slice() {
            GateNetwork::update_kind(
                *gate_id,
                unsafe{ *self.runtime_gate_kind.get_unchecked(*gate_id as usize)},
                &mut self.cluster_update_list,
                &self.packed_outputs,
                &self.packed_output_indexes,
                &mut self.acc,
                &mut self.state,
                &mut self.in_update_list,
                //& self.runtime_gate_kind,
                //Some(RunTimeGateType::OrNand),
                );
        }
        self.update_list.clear();
        for cluster_id in self.cluster_update_list.get_slice() {
            GateNetwork::update_kind(
                *cluster_id,
                RunTimeGateType::OrNand,
                &mut self.update_list,
                &self.packed_outputs,
                &self.packed_output_indexes,
                &mut self.acc,
                &mut self.state,
                &mut self.in_update_list,
                //& self.runtime_gate_kind,
                //None,
                );
        }
        //for cluster_id in &self.cluster_update_list {
        //}
        self.cluster_update_list.clear();
    }
    /// Adds all gates to update list and performs initialization
    /// and TODO: network optimizaton.
    /// Currently cannot be modified after initialization.
    /// # Panics
    /// Not initialized
    pub(crate) fn init_network(&mut self) {
        assert!(!self.initialized);

        let number_of_gates = self.gates.len();

        //self.update_list         = Vec::with_capacity(number_of_gates);
        //self.cluster_update_list = Vec::with_capacity(number_of_gates);
        //
        
        self.        update_list.list = vec![0; number_of_gates].into_boxed_slice();
        self.cluster_update_list.list = vec![0; number_of_gates].into_boxed_slice();
        self.        update_list.len = 0;
        self.cluster_update_list.len = 0;


        for gate_id in 0..number_of_gates {
            let gate = &mut self.gates[gate_id];
            let kind = gate.kind;

            // add gates that will immediately update to the
            // update list
            if kind.is_inverted() {
                self.update_list.push(gate_id.try_into().unwrap()); 
                gate.in_update_list = true;
            } 

            // pack gatetype, acc, state, outputs
            self.runtime_gate_kind.push(RunTimeGateType::new(gate.kind));
            self.acc.push(gate.acc);
            self.state.push(gate.state); 
            self.in_update_list.push(gate.in_update_list);
            self.packed_output_indexes.push(self.packed_outputs.len().try_into().unwrap());
            self.packed_outputs.append(&mut gate.outputs.clone());
        }
        self.packed_output_indexes.push(self.packed_outputs.len().try_into().unwrap());

        let mut gate_set: HashMap<(GateType, Vec<IndexType>), IndexType> = HashMap::new();
        for gate_id in 0..self.gates.len() {
            let gate = &self.gates[gate_id];
            let key = gate.generate_gate_key();
            match gate_set.get(&key) {
                Some(other_id) => /*println!("{gate_id} is {other_id}")*/{},
                None => {gate_set.insert(key, gate_id as IndexType); ()},
            }
            //let mut gate_set: HashMap<(GateType, Vec<IndexType>), IndexType> = HashMap::new();
            //for gate_id in 0..self.gates.len() {
            //    let gate = &self.gates[gate_id];
            //    let key = GateNetwork::generate_gate_key(gate);
            //    match gate_set.get(&key) {
            //        Some(other_id) => println!("{gate_id} is {other_id}"),
            //        None => {gate_set.insert(key, gate_id as IndexType); ()},
            //    }
            //}
            //let a = gate_set.len();
            //let b = self.gates.len();

            //println!("needed gates/full set = {}/{} = {}%",a,b,a*100/b);

            self.initialized = true;
        }
        //let b = self.gates.len();
        //let a = b-gate_set.len();

    }

    #[inline(always)]
    fn update_kind(
        id: IndexType,
        kind: RunTimeGateType,
        update_list: &mut RawList,
        packed_outputs: &[IndexType],
        packed_output_indexes: &[IndexType],
        acc: &mut Vec<AccType>,
        state: &mut Vec<bool>,
        in_update_list: &mut Vec<bool>,
        //kinds: &[RunTimeGateType],
        //other_kind: Option<RunTimeGateType>,
        ) {

        // if this assert fails, the system will recover anyways
        // but that would probably have been caused by a bug.
        debug_assert!(in_update_list[id as usize], "{id:?}"); 

        // memory reads:
        //
        // id:        acc, 2*state, packed_output_indexes, in_update_list
        // id+1:                    packed_output_indexes
        //
        // in loop (*iterations):
        // i:         packed_outputs
        // output_id: in_update_list, acc
        //
        // update_list push output_id
        //
        // acc - in_update_list = (8,8)
        // state - packed_output_indexes = (8,16)
        // packed_outputs = (16)
        // update_list = (16)
        //
        // (acc, in_update_list/state, packed_output_indexes) = (8,8/8,16)
        //
        // packed_outputs = (16)
        // update_list = (16)
        //
        //
        // (acc, in_update_list, state) = (16,8,8)
        //
        // TODO: inline abstract over reading/writing these
        //
        //
        //

        unsafe {
            let next = Gate::evaluate_from_runtime_static(*acc.get_unchecked(id as usize), kind);
            if *state.get_unchecked(id as usize) != next {
                let delta = if next {1} else {(0 as AccType).wrapping_sub(1)};
                let from_index = *packed_output_indexes.get_unchecked(id as usize  );
                let   to_index = *packed_output_indexes.get_unchecked(id as usize+1);
                for i in from_index..to_index {
                    let output_id = packed_outputs.get_unchecked(i as usize);
                    let other_acc = acc.get_unchecked_mut(*output_id as usize);
                    *other_acc = other_acc.wrapping_add(delta);
                    //*other_acc += delta;
                    let in_update_list = in_update_list.get_unchecked_mut(*output_id as usize);
                    if !*in_update_list {
                        *in_update_list = true;
                        update_list.push(*output_id);
                    }
                }
                *state.get_unchecked_mut(id as usize) = next;
            }
            // this gate should be ready to be readded to the update list.
            *in_update_list.get_unchecked_mut(id as usize) = false; 
        }
    }
}
