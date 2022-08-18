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

/// the only cases that matter at the hot code sections
#[derive(Debug)]
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
//
type AccType = i8;
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
    //fn new_cluster() -> Self {
    //    Self::new(GateType::CLUSTER, Vec::new())
    //}
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
    #[inline(always)]
    fn evaluate(&self) -> bool {
        match self.kind {
            GateType::NAND | GateType::OR | GateType::CLUSTER 
                => self.acc != 0,
            GateType::AND | GateType::NOR
                => self.acc == 0,
            GateType::XOR | GateType::XNOR
                => self.acc & 1 == 1,
        } 
    }

    #[inline(always)]
    fn evaluate_from_kind(&self, kind:GateType) -> bool {
        match kind {
            GateType::NAND | GateType::OR | GateType::CLUSTER 
                => self.acc != 0,
            GateType::AND | GateType::NOR
                => self.acc == 0,
            GateType::XOR | GateType::XNOR
                => self.acc & 1 == 1,
        } 
    }

    
    #[inline(always)]
    fn evaluate_from_runtime(&self, kind:RunTimeGateType) -> bool {
        match kind {
            RunTimeGateType::OrNand => self.acc != 0,
            RunTimeGateType::AndNor => self.acc == 0,
            RunTimeGateType::XorXnor => self.acc & 1 == 1,
        } 
    }

}

#[derive(Debug, Default)]
pub struct GateNetwork {
    pub gates: Vec<Gate>,
    //clusters: Vec<Gate>,
    update_list: Vec<IndexType>,
    packed_outputs: Vec<IndexType>,
    packed_output_indexes: Vec<IndexType>,
    runtime_gate_kind: Vec<RunTimeGateType>,


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
        let next_id = self.gates.len();
        self.gates.push(Gate::from_gate_type(kind));
        next_id
    }

    /// Add inputs to gate_id fron inputs.
    /// Connection must be between cluster and a non cluster gate. 
    /// Only add connection once plz (TODO: Enforce with assertion)
    /// Above assertion will guarantee the shape of the network.
    pub fn add_inputs(&mut self, kind: GateType, gate_id: usize, inputs: Vec<usize>) {
        let gate = &mut self.gates[gate_id];
        gate.add_inputs(inputs.len() as i32);
        for input_id in inputs {
            for output in &self.gates[input_id].outputs {
                assert_ne!(*output,gate_id as IndexType);
            }

            // panics if it cannot fit in IndexType
            self.gates[input_id].outputs.push(gate_id as IndexType);
            self.gates[input_id].outputs.sort();
        }
    }

    pub fn get_state(&self, gate_id: usize) -> bool {
        self.gates[gate_id].state
    }
    #[inline(always)]
    pub fn update(&mut self) {
        // TODO: swap buffers instead of 2 lists.
        let mut cluster_update_list = Vec::new();
        //println!("update_list: {:?}", self.update_list);
        for gate_id in &self.update_list {
            //Gate::update(*gate_id, &mut cluster_update_list, &mut self.gates);
            //GateNetwork::update_kind(*gate_id, self.gates[*gate_id as usize].kind, &mut cluster_update_list, &mut self.gates, &self.packed_outputs, &self.packed_output_indexes);
            unsafe {
                GateNetwork::update_kind(
                    *gate_id,
                    self.gates.get_unchecked(*gate_id as usize).kind,
                    &mut cluster_update_list,
                    &mut self.gates,
                    &self.packed_outputs,
                    &self.packed_output_indexes);
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
            GateNetwork::update_kind(*cluster_id, GateType::OR, &mut self.update_list, &mut self.gates, &self.packed_outputs, &self.packed_output_indexes);
        }
    }
    /// Adds all gates to update list and performs initialization
    /// and TODO: network optimizaton.
    pub fn init_network(&mut self) {
        // add all gates to update list.
        for gate_id in 0..self.gates.len() {
            let kind = self.gates[gate_id as usize].kind;
            if kind != GateType::CLUSTER {
                self.update_list.push(gate_id as IndexType); 
                self.gates[gate_id].in_update_list = true;
            }
        } 
        for gate_id in 0..self.gates.len() {
            let gate = &self.gates[gate_id];
            self.packed_output_indexes.push(self.packed_outputs.len() as IndexType);
            self.packed_outputs.append(&mut gate.outputs.clone());
        }
        self.packed_output_indexes.push(self.packed_outputs.len() as IndexType);

        
        for gate_id in 0..self.gates.len() {
            let gate = &self.gates[gate_id];
            self.runtime_gate_kind.push(RunTimeGateType::new(gate.kind));

        }
    }



    #[inline(always)]
    fn update_kind(
        id: IndexType,
        kind: GateType,
        update_list: &mut Vec<IndexType>,
        gates: &mut Vec<Gate>,
        packed_outputs: &Vec<IndexType>,
        packed_output_indexes: &Vec<IndexType>) {
        //TODO: make idiomatic
        //let this = &mut clusters[gate_id];
        // if this assert fails, the system will recover anyways
        // but that would probably have been caused by a bug.
        //let gate = unsafe { gates.get_unchecked(id) };
        debug_assert!(gates[id as usize].in_update_list); 

        //println!("Update:\n{:?}",this);
            // TODO move setting update list to aid compiler optimizer
                //println!("new state!");
                //TODO: move up the chain.
                //for i in 0..gates.get_unchecked(id as usize).outputs.len() {
                //    let output_id = gates.get_unchecked(id as usize).outputs[i];
                    //let cluster = &mut clusters[*output_id];
                    //let cluster = &mut gates[*output_id as usize];
                    //println!("Cluster:\n{:?}",this);
        unsafe {
            let next = gates.get_unchecked(id as usize).evaluate_from_kind(kind);
            if gates.get_unchecked(id as usize).state != next {
                let delta = if next {1} else {-1};
                for i in *packed_output_indexes.get_unchecked(id as usize)..*packed_output_indexes.get_unchecked(id as usize+1) {
                    let output_id = packed_outputs.get_unchecked(i as usize);
                    //assert!(*output_id != id);
                    let cluster = gates.get_unchecked_mut(*output_id as usize);
                    cluster.acc += delta;
                    if !cluster.in_update_list {
                        cluster.in_update_list = true;
                        update_list.push(*output_id);
                    }
                }
                gates.get_unchecked_mut(id as usize).state = next;
            }
            gates.get_unchecked_mut(id as usize).in_update_list = false; // this gate should be ready to be readded to the update list.
        }
    }
}
