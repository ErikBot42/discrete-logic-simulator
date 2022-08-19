// logic.rs: contains the simulaion engine itself.
use std::collections::BTreeSet;



#[derive(Debug, PartialEq, Copy, Clone)]
pub enum GateType {
    AND,
    OR,
    NOR,
    NAND,
    XOR,
    XNOR,
    CLUSTER, // equivilent to OR
}

// encode gate information & state in flags
struct GateFlags {
    inner: u8,
}
impl GateFlags {
    const STATE_MASK:   u8 = 0b0000_0001; // must be in this spot
    const AND_NOR_MASK: u8 = 0b0000_0010; // can change
    const XOR_MASK:     u8 = 0b0000_0100; // can change
    fn new(kind: GateType) -> Self {
        GateFlags{inner: match RunTimeGateType::new(kind) {
            RunTimeGateType::OrNand => 0,
            RunTimeGateType::AndNor => Self::AND_NOR_MASK,
            RunTimeGateType::XorXnor => Self::XOR_MASK,
        }}
    }

    fn state(&self) -> bool {
        // equivilent to inner & mask as bool, but would 
        // need a panic branch
        self.inner & Self::STATE_MASK == Self::STATE_MASK
    }

    #[inline(always)]
    fn set_state(&mut self, new_state: bool) {
        self.inner = (!Self::STATE_MASK)&self.inner|new_state as u8;
    }

    /// update gate state, if state changed, return count delta
    #[inline(always)]
    fn eval_set(&mut self, acc: AccType) -> Option<AccType> {
        // hopefully only a single 0 will need
        // to be put in a register.
        // TODO: keep in u8 longer to aid compiler?
        let new_state: u8 = 
        if self.inner & Self::XOR_MASK == 0 {
            if self.inner & Self::AND_NOR_MASK == 0 {
                // or, nand
                (acc != 0) as u8
            }
            else {
                // and, nor 
                (acc == 0) as u8
            }
        }
        else {
            // xor, xnor
            unsafe { std::mem::transmute::<i8,u8>(acc & 1) }
        };

        let state: u8 = self.inner & Self::STATE_MASK;


        if new_state 



        
        unimplemented!()
    }
    //fn 
}

/// the only cases that matter at the hot code sections
#[derive(Debug, Copy, Clone, PartialEq)]
enum RunTimeGateType {
    OrNand,
    AndNor,
    XorXnor,
}
impl RunTimeGateType {
    fn new(kind: GateType) -> Self {
        match kind {
            GateType::AND => RunTimeGateType::AndNor,
            GateType::OR => RunTimeGateType::OrNand,
            GateType::NOR => RunTimeGateType::AndNor,
            GateType::NAND => RunTimeGateType::OrNand,
            GateType::XOR => RunTimeGateType::XorXnor,
            GateType::XNOR => RunTimeGateType::XorXnor,
            GateType::CLUSTER => RunTimeGateType::OrNand, // equivilent to OR
        }
    }
}

// will only support about 128 inputs/outputs (or about 255 if wrapped add)
type AccType = i8;

// tests don't need that many indexes, but this is obviusly a big limitation.
type IndexType = u8;

/// data needed after processing network
#[derive(Debug)]
pub struct Gate {
    // constant:
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
            GateType::XNOR => 1,
            _ => 0,
        };
        Gate {
            outputs,
            acc: start_acc,
            kind,
            state: false, // all gates/clusters init to off
            in_update_list: false,
        }
    }
    fn from_gate_type(kind: GateType) -> Self {
        Self::new(kind, Vec::new())
    }
    /// Change number of inputs to handle logic correctly
    /// Can be called multiple times for *diffrent* inputs
    fn add_inputs(&mut self, inputs: i32) {
        match self.kind {
            GateType::AND | GateType::NAND 
                => self.acc -= inputs as AccType,
            GateType::OR | GateType::NOR | GateType::XOR | GateType::XNOR | GateType::CLUSTER
                => (),
        }
    }
    //#[inline(always)]
    //#[inline(always)]
    //fn evaluate(&self) -> bool {
    //    match self.kind {
    //        GateType::NAND | GateType::OR | GateType::CLUSTER 
    //            => self.acc != 0,
    //        GateType::AND | GateType::NOR
    //            => self.acc == 0,
    //        GateType::XOR | GateType::XNOR
    //            => self.acc & 1 == 1,
    //    } 
    //}
    //#[inline(always)]
    //fn evaluate_from_kind(&self, kind:GateType) -> bool {
    //    match kind {
    //        GateType::NAND | GateType::OR | GateType::CLUSTER 
    //            => self.acc != 0,
    //        GateType::AND | GateType::NOR
    //            => self.acc == 0,
    //        GateType::XOR | GateType::XNOR
    //            => self.acc & 1 == 1,
    //    } 
    //}
    //#[inline(always)]
    //fn evaluate_from_runtime(&self, kind:RunTimeGateType) -> bool {
    //    match kind {
    //        RunTimeGateType::OrNand  => self.acc != 0,
    //        RunTimeGateType::AndNor  => self.acc == 0,
    //        RunTimeGateType::XorXnor => self.acc & 1 == 1,
    //    } 
    //}

    #[inline(always)]
    fn evaluate_from_runtime_static(acc: AccType, kind:RunTimeGateType) -> bool {
        match kind {
            RunTimeGateType::OrNand  => acc != 0,
            RunTimeGateType::AndNor  => acc == 0,
            RunTimeGateType::XorXnor => acc & 1 == 1,
        } 
    }


    //#[inline(always)]
    //fn evaluate_from_runtime_static(acc: AccType, kind:RunTimeGateType) -> bool {
    //    kind == RunTimeGateType::OrNand  && (acc != 0)||
    //    kind == RunTimeGateType::AndNor  && (acc == 0)||
    //    kind == RunTimeGateType::XorXnor && (acc & 1 == 1)
    //}
    // xor_xnor is a bit mask
    //fn evaluate(acc: AccType, or_nand: bool, xor_xnor: u8) -> bool {
    //    acc == 0 
    //}
}

#[derive(Debug, Default)]
pub struct GateNetwork {
    //TODO: bitvec
    gates: Vec<Gate>,
    //clusters: Vec<Gate>,
    
    update_list: Vec<IndexType>,
    packed_outputs: Vec<IndexType>,
    packed_output_indexes: Vec<IndexType>,
    state: Vec<bool>,
    acc: Vec<AccType>,
    in_update_list: Vec<bool>,


    runtime_gate_kind: Vec<RunTimeGateType>,

    initialized: bool,
    //acc: Vec<AccType>,


    // outputs: Vec<BTreeSet<IndexType>>,
    // inputs: Vec<BTreeSet<IndexType>>,
    //TODO: packed outputs representation
    // just storing start of indexes is enough, but a slice is safer.
}
// TODO: only add to update list if state will change?
// TODO: add layer after cluster directly?
// TODO: atomics do not seem to significantly impact performance, therefore, they could be used.

impl GateNetwork {
    /// Internally creates a vertex.
    /// Returns vertex id
    /// ids of gates are guaranteed to be unique
    pub fn add_vertex(&mut self, kind: GateType) -> usize {
        assert!(!self.initialized);
        let next_id = self.gates.len();
        self.gates.push(Gate::from_gate_type(kind));
        next_id
    }

    /// Add inputs to gate_id fron inputs.
    /// Connection must be between cluster and a non cluster gate. 
    /// Panics if connection is added more than once.
    /// Above assertion will guarantee the shape of the network.
    pub fn add_inputs(&mut self, kind: GateType, gate_id: usize, inputs: Vec<usize>) {
        assert!(!self.initialized);
        let gate = &mut self.gates[gate_id];
        gate.add_inputs(inputs.len() as i32);
        for input_id in inputs {
            assert!(input_id<self.gates.len());
            for output in &self.gates[input_id].outputs {
                assert_ne!(*output,gate_id as IndexType);
            }
            // panics if it cannot fit in IndexType
            self.gates[input_id].outputs.push(gate_id as IndexType);
            self.gates[input_id].outputs.sort();
        }
    }

    pub fn get_state(&self, gate_id: usize) -> bool {
        assert!(self.initialized);
        //self.gates[gate_id].state
        self.state[gate_id]
    }
    #[inline(always)]
    pub fn update(&mut self) {
        assert!(self.initialized);
        // TODO: swap buffers instead of 2 lists.
        let mut cluster_update_list = Vec::new();
        //println!("update_list: {:?}", self.update_list);
        for gate_id in &self.update_list {
            //Gate::update(*gate_id, &mut cluster_update_list, &mut self.gates);
            //GateNetwork::update_kind(*gate_id, self.gates[*gate_id as usize].kind, &mut cluster_update_list, &mut self.gates, &self.packed_outputs, &self.packed_output_indexes);
            unsafe {
                GateNetwork::update_kind(
                    *gate_id,
                    //self.gates.get_unchecked(*gate_id as usize).kind,
                    *self.runtime_gate_kind.get_unchecked(*gate_id as usize),
                    &mut cluster_update_list,
                    &self.packed_outputs,
                    &self.packed_output_indexes,
                    &mut self.acc,
                    &mut self.state,
                    &mut self.in_update_list);
            }
            //Gate::update_kind2(*gate_id, self.gates[*gate_id].kind, &mut cluster_update_list, &mut self.gates);
        }
        //println!("cluster_update_list: {:?}", cluster_update_list);
        self.update_list.clear();
        // TODO: call diffrent update function that makes more assumptions here.
        // this will be guaranteed safe since shape of network is known.
        for cluster_id in &cluster_update_list {
            //Gate::update(*cluster_id, &mut self.update_list, &mut self.gates);
            //Gate::update_assume_or(*cluster_id, &mut self.update_list, &mut self.gates);
            //Gate::update_kind(*cluster_id, self.gates[*cluster_id].kind, &mut self.update_list, &mut self.gates);
            GateNetwork::update_kind(
                *cluster_id,
                RunTimeGateType::OrNand,
                &mut self.update_list,
                &self.packed_outputs,
                &self.packed_output_indexes,
                &mut self.acc,
                &mut self.state,
                &mut self.in_update_list);
        }
    }
    /// Adds all gates to update list and performs initialization
    /// and TODO: network optimizaton.
    /// Currently cannot be modified after initialization.
    pub fn init_network(&mut self) {
        assert!(!self.initialized);

        // add all gates to update list.
        for gate_id in 0..self.gates.len() {
            let kind = self.gates[gate_id as usize].kind;
            if kind != GateType::CLUSTER {
                self.update_list.push(gate_id as IndexType); 
                self.gates[gate_id].in_update_list = true;
            }
        } 

        // pack outputs
        for gate_id in 0..self.gates.len() {
            let gate = &self.gates[gate_id];
            self.packed_output_indexes.push(self.packed_outputs.len() as IndexType);
            self.packed_outputs.append(&mut gate.outputs.clone());
        }
        self.packed_output_indexes.push(self.packed_outputs.len() as IndexType);
        
        // pack gatetype, acc, state
        for gate_id in 0..self.gates.len() {
            let gate = &self.gates[gate_id];
            self.runtime_gate_kind.push(RunTimeGateType::new(gate.kind));
            self.acc.push(gate.acc);
            self.state.push(gate.state); 
            self.in_update_list.push(gate.in_update_list);
        }
        self.initialized = true;
    }

    #[inline(always)]
    fn update_kind(
        id: IndexType,
        kind: RunTimeGateType,
        update_list: &mut Vec<IndexType>,
        packed_outputs: &Vec<IndexType>,
        packed_output_indexes: &Vec<IndexType>,
        acc: &mut Vec<AccType>,
        state: &mut Vec<bool>,
        in_update_list: &mut Vec<bool>) {

        // if this assert fails, the system will recover anyways
        // but that would probably have been caused by a bug.
        //debug_assert!(gates[id as usize].in_update_list); 
        debug_assert!(in_update_list[id as usize]); 

        unsafe {
            //let next = Gate::evaluate_from_runtime_static(gates.get_unchecked(id as usize).acc, kind);
            let next = Gate::evaluate_from_runtime_static(*acc.get_unchecked(id as usize), kind);
            if *state.get_unchecked(id as usize) != next {
                let delta = if next {1} else {-1};
                for i in *packed_output_indexes.get_unchecked(id as usize)..*packed_output_indexes.get_unchecked(id as usize+1) {
                    let output_id = packed_outputs.get_unchecked(i as usize);
                    *acc.get_unchecked_mut(*output_id as usize) += delta;
                    if !*in_update_list.get_unchecked_mut(*output_id as usize) {
                        *in_update_list.get_unchecked_mut(*output_id as usize) = true;
                        update_list.push(*output_id);
                    }
                }
                *state.get_unchecked_mut(id as usize) = next;
            }
            //gates.get_unchecked_mut(id as usize).in_update_list = false; // this gate should be ready to be readded to the update list.
            *in_update_list.get_unchecked_mut(id as usize) = false; // this gate should be ready to be readded to the update list.
        }
    }
}
