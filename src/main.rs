#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
enum GateType {
    AND,
    OR,
    NOR,
    NAND,
    XOR,
    XNOR,
    CLUSTER, // equivilent to OR
}

// data needed after processing network
#[derive(Debug)]
struct Gate {
    // constant:
    outputs: Vec<usize>, // list of ids
    kind: GateType,

    // variable:
    acc: i32,            // TODO i8
    state: bool,
    in_update_list: bool,
}
impl Gate {
    fn new(kind: GateType, outputs: Vec<usize>) -> Self {
        let start_acc = match kind {
            GateType::XNOR 
                => 0,
            GateType::AND | GateType::OR | GateType::NOR | GateType::NAND | GateType::XOR | GateType::CLUSTER 
                => 0
        };
        Gate {
            outputs: Vec::new(),
            acc: start_acc,
            kind: GateType::CLUSTER,
            state: false, // all gates/clusters init to off
            in_update_list: false,
        }
    }
    fn new_cluster() -> Self {
        Self::new(GateType::CLUSTER, Vec::new())
    }
    // change number of inputs to handle logic correctly
    fn add_inputs(&mut self, inputs: i32) {
        match self.kind {
            GateType::AND | GateType::NAND 
                => self.acc -= inputs,
            GateType::OR | GateType::NOR | GateType::XOR | GateType::XNOR | GateType::CLUSTER
                => (),
        }
    }
    //#[inline(always)]
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
    //#[inline(always)]
    fn update(&mut self, update_list: &mut Vec<usize>, clusters: &mut Vec<Gate>) {
        // if this assert fails, the system will recover anyways
        // but that would probably have been caused by a bug.
        assert!(self.in_update_list); 

        self.in_update_list = false; // this gate should be ready to be readded to the update list.
        let next = self.evaluate();
        if self.state != next {
            for output_id in &self.outputs {
                let cluster = &mut clusters[*output_id];
                cluster.acc += if next {1} else {-1};
                if !cluster.in_update_list {
                    cluster.in_update_list = true;
                    update_list.push(*output_id);
                }
            }
            self.state = next;
        }
    }
}

#[derive(Debug)]
struct GateNetwork {
    gates: Vec<Gate>,
    clusters: Vec<Gate>,
    update_list: Vec<usize>,
}

// only add to update list if state will change?

impl GateNetwork {
    fn add_gate(&mut self, gate: Gate, inputs: Vec<usize>) -> usize {
        GateNetwork::internal_add_circuit(gate, &mut self.gates, &mut self.clusters, inputs)
    }
    fn add_cluster(&mut self) -> usize {
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
    fn update(&mut self, mut update_list: Vec<usize>) {
        let mut cluster_update_list = Vec::new();
        for gate_id in &update_list {
            self.gates[*gate_id].update(&mut cluster_update_list, &mut self.clusters);
        }
        update_list.clear();
        for cluster_id in cluster_update_list {
            self.gates[cluster_id].update(&mut update_list, &mut self.clusters);
        }
    }
}

fn main() {
    let mut network = GateNetwork{gates: Vec::new(), clusters: Vec::new(), update_list: Vec::new()};
    let c1 = network.add_cluster();
    let c2 = network.add_cluster();
    let g1 = network.add_gate(Gate::new(GateType::NOR, [c1].to_vec()), [c1,c2].to_vec());
    let update_list = [g1].to_vec();
    println!("{:#?}",network);
    network.update(update_list);
    println!("{:#?}",network);

    //let g1 = 
}
