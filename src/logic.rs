// logic.rs: contains the simulaion engine itself.


#[derive(Debug)]
pub enum GateType {
    AND,
    OR,
    NOR,
    NAND,
    XOR,
    XNOR,
    CLUSTER, // equivilent to OR
}

// data needed after processing network
type AccType = i8;
#[derive(Debug)]
pub struct Gate {
    // constant:
    outputs: Vec<usize>, // list of ids
    kind: GateType,

    // variable:
    acc: AccType,            // TODO i8
    state: bool,
    in_update_list: bool,
}
impl Gate {
    pub fn new(kind: GateType, outputs: Vec<usize>) -> Self {
        let start_acc = match kind {
            GateType::XNOR 
                => 1,
            GateType::AND | GateType::OR | GateType::NOR | GateType::NAND | GateType::XOR | GateType::CLUSTER 
                => 0
        };
        Gate {
            outputs,
            acc: start_acc,
            kind,
            state: false, // all gates/clusters init to off
            in_update_list: false,
        }
    }
    fn new_cluster() -> Self {
        Self::new(GateType::CLUSTER, Vec::new())
    }
    // change number of inputs to handle logic correctly
    fn add_inputs(&mut self, inputs: AccType) {
        match self.kind {
            GateType::AND | GateType::NAND 
                => self.acc -= inputs,
            GateType::OR | GateType::NOR | GateType::XOR | GateType::XNOR | GateType::CLUSTER
                => (),
        }
    }
    #[inline(never)]
    fn evaluate(&self) -> bool {
        match self.kind {
            GateType::NAND | GateType::OR | GateType::CLUSTER 
                => self.acc != 0,
            GateType::AND | GateType::NOR
                => self.acc == 0,
            GateType::XOR | GateType::XNOR
                => self.acc & 1 == 0,
        } 
    }
    #[inline(never)]
    fn update(&mut self, update_list: &mut Vec<usize>, clusters: &mut Vec<Gate>) {
        // if this assert fails, the system will recover anyways
        // but that would probably have been caused by a bug.
        debug_assert!(self.in_update_list); 

        //println!("Update:\n{:?}",self);
        self.in_update_list = false; // this gate should be ready to be readded to the update list.
        let next = self.evaluate();
        if self.state != next {
            //println!("new state!");
            for output_id in &self.outputs {
                let cluster = &mut clusters[*output_id];
                cluster.acc += if next {1} else {-1};
                if !cluster.in_update_list {
                    cluster.in_update_list = true;
                    update_list.push(*output_id);
                }
                //println!("Cluster:\n{:?}",self);

            }
            self.state = next;
        }
    }
}

#[derive(Debug, Default)]
pub struct GateNetwork {
    gates: Vec<Gate>,
    clusters: Vec<Gate>,
    update_list: Vec<usize>,
}
// TODO: merge gates & cluster lists?
// TODO: only add to update list if state will change?
// TODO: add layer after cluster directly?
// TODO: atomics do not seem to significantly impact performance, therefore 

impl GateNetwork {
    pub fn add_gate(&mut self, gate: Gate, inputs: Vec<usize>) -> usize {
        GateNetwork::internal_add_circuit(gate, &mut self.gates, &mut self.clusters, inputs)
    }
    pub fn add_cluster(&mut self) -> usize {
        GateNetwork::internal_add_circuit(Gate::new_cluster(), &mut self.clusters, &mut self.gates, Vec::new())
    }
    fn internal_add_circuit(mut gate: Gate, gates: &mut Vec<Gate>, clusters: &mut Vec<Gate>, inputs: Vec<usize>) -> usize {
        gate.add_inputs(inputs.len().try_into().unwrap());
        let next_id = gates.len();
        gates.push(gate);
        for input_id in inputs {
            clusters[input_id].outputs.push(next_id);
        }
        next_id
    }
    #[inline(never)]
    pub fn update(&mut self) {
        let mut cluster_update_list = Vec::new();
        //println!("update_list: {:?}", self.update_list);
        for gate_id in &self.update_list {
            self.gates[*gate_id].update(&mut cluster_update_list, &mut self.clusters);
        }
        //println!("cluster_update_list: {:?}", cluster_update_list);
        self.update_list.clear();
        for cluster_id in cluster_update_list {
            self.clusters[cluster_id].update(&mut self.update_list, &mut self.gates);
        }
    }
    pub fn add_all_gates_to_update_list(&mut self) {
        for gate_id in 0..self.gates.len() {
            self.update_list.push(gate_id); 
            self.gates[gate_id].in_update_list = true;
        } 
    }
}
