#[allow(clippy::upper_case_acronyms)]
use std::time::Instant;
//use std::sync::atomic::*;

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
type acc_type = i8;
#[derive(Debug)]
struct Gate {
    // constant:
    outputs: Vec<usize>, // list of ids
    kind: GateType,

    // variable:
    acc: acc_type,            // TODO i8
    state: bool,
    in_update_list: bool,
}
impl Gate {
    fn new(kind: GateType, outputs: Vec<usize>) -> Self {
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
    fn add_inputs(&mut self, inputs: acc_type) {
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
    #[inline(never)]
    fn update(&mut self) {
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
    fn add_all_gates_to_update_list(&mut self) {
        for gate_id in 0..self.gates.len() {
            self.update_list.push(gate_id); 
            self.gates[gate_id].in_update_list = true;
        } 
    }
}

fn main() {
    let mut network = GateNetwork{gates: Vec::new(), clusters: Vec::new(), update_list: Vec::new()};
    let c1 = network.add_cluster();
    //let c2 = network.add_cluster();
    let g1 = network.add_gate(Gate::new(GateType::NOR, [c1].to_vec()), [c1/*,c2*/].to_vec());

    network.add_all_gates_to_update_list();
    println!("{:#?}",network);
    network.update();
    println!("{:#?}",network);
    
    network.update();
    println!("{:#?}",network);
    let iterations = 1_000_000_0;
    println!("running {} iterations",iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        network.update();
    }
    let elapsed_time = start.elapsed().as_millis();
    println!("{:#?}",network);
    println!("running {} iterations took {} ms",iterations, elapsed_time);
    println!("done");


    //let g1 = 
}
