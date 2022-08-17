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
enum RunTimeGateType {
    OR_NAND,
    AND_NOR,
    XOR_XNOR,
}
impl RunTimeGateType {
    fn new(kind: GateType) -> Self {
        match kind {
            GateType::AND => RunTimeGateType::AND_NOR,
            GateType::OR => RunTimeGateType::OR_NAND,
            GateType::NOR => RunTimeGateType::AND_NOR,
            GateType::NAND => RunTimeGateType::OR_NAND,
            GateType::XOR => RunTimeGateType::XOR_XNOR,
            GateType::XNOR => RunTimeGateType::XOR_XNOR,
            GateType::CLUSTER => RunTimeGateType::OR_NAND, // equivilent to OR
        }
    }
}

// will only support about 128 inputs/outputs (or about 255 if wrapped add)
//
type AccType = i8;

/// data needed after processing network
#[derive(Debug)]
pub struct Gate {
    // constant:
    outputs: Vec<usize>, // list of ids
    kind: GateType,

    // variable:
    acc: AccType, 
    state: bool,
    in_update_list: bool,
}
impl Gate {
    fn new(kind: GateType, outputs: Vec<usize>) -> Self {
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
    //#[inline(never)]
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
    //#[inline(never)]
    fn update(id: usize, update_list: &mut Vec<usize>, gates: &mut Vec<Gate>) {
        //TODO: make idiomatic
        //let this = &mut clusters[gate_id];
        // if this assert fails, the system will recover anyways
        // but that would probably have been caused by a bug.
        debug_assert!(gates[id].in_update_list); 

        //println!("Update:\n{:?}",this);
        gates[id].in_update_list = false; // this gate should be ready to be readded to the update list.
        let next = gates[id].evaluate();
        if gates[id].state != next {
            //println!("new state!");
            for i in 0..gates[id].outputs.len() {
                let output_id = gates[id].outputs[i];
                //let cluster = &mut clusters[*output_id];
                let cluster = &mut gates[output_id];
                cluster.acc += if next {1} else {-1};
                if !cluster.in_update_list {
                    cluster.in_update_list = true;
                    update_list.push(output_id);
                }
                //println!("Cluster:\n{:?}",this);

            }
            gates[id].state = next;
        }
    }
}

#[derive(Debug, Default)]
pub struct GateNetwork {
    pub gates: Vec<Gate>,
    //clusters: Vec<Gate>,
    update_list: Vec<usize>,
    // outputs: Vec<BTreeSet<usize>>,
    // inputs: Vec<BTreeSet<usize>>,
    //TODO: packed outputs representation
    // just storing start of indexes is enough, but a slice is safer.
}
// TODO: merge gates & cluster lists?
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
            self.gates[input_id].outputs.push(gate_id);
        }
    }
    pub fn get_state(&self, kind: GateType, gate_id: usize) -> bool {
        self.gates[gate_id].state
    }
    //#[inline(never)]
    pub fn update(&mut self) {
        let mut cluster_update_list = Vec::new();
        //println!("update_list: {:?}", self.update_list);
        for gate_id in &self.update_list {
            Gate::update(*gate_id, &mut cluster_update_list, &mut self.gates);
        }
        //println!("cluster_update_list: {:?}", cluster_update_list);
        self.update_list.clear();
        // TODO: call diffrent update function that makes more assumptions here.
        // this will be guaranteed safe since shape of network is known.
        for cluster_id in &cluster_update_list {
            Gate::update(*cluster_id, &mut self.update_list, &mut self.gates);
        }
    }
    /// Adds all gates to update list and performs initialization
    /// and TODO: network optimizatoin.
    pub fn init_network(&mut self) {
        // add all gates to update list.
        for gate_id in 0..self.gates.len() {
            let kind = self.gates[gate_id].kind;
            if kind != GateType::CLUSTER {
                self.update_list.push(gate_id); 
                self.gates[gate_id].in_update_list = true;
            }
        } 
    }
}
