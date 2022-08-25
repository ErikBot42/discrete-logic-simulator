// logic.rs: contains the simulaion engine itself.
// use std::collections::BTreeSet;
// use std::collections::HashSet;
// use std::collections::HashMap;




#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
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
/*#[derive(Debug)]
struct GateFlags {
    inner: u8,
}
impl GateFlags {
    const STATE_MASK:       u8 = 0b0000_0001; // must be in this spot
    const AND_NOR_MASK:     u8 = 0b0000_0010; // can change
    const XOR_MASK:         u8 = 0b0000_0100; // can change
    const UPDATE_LIST_MASK: u8 = 0b0000_1000; // can change
    fn new(kind: GateType) -> Self {
        GateFlags{inner: match RunTimeGateType::new(kind) {
            RunTimeGateType::OrNand => 0,
            RunTimeGateType::AndNor => Self::AND_NOR_MASK,
            RunTimeGateType::XorXnor => Self::XOR_MASK,
        }}
    }
    //fn in_update_list(self) -> bool {
    //    self.inner & Self::UPDATE_LIST_MASK == Self::UPDATE_LIST_MASK
    //}
    //fn set_in_update_list(&mut self, set: bool) {
    //    if set {
    //        self.inner |= Self::UPDATE_LIST_MASK;
    //    }
    //    else {
    //        self.inner &= !Self::UPDATE_LIST_MASK;
    //    }
    //}

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
                } else {
                    // and, nor 
                    (acc == 0) as u8
                }
            } else {
                // xor, xnor
                unsafe { std::mem::transmute::<i8,u8>(acc & 1) }
            };

        let state_curr: u8 = self.inner & Self::STATE_MASK;
        let rval = if new_state == state_curr {
            None
        } else {
            Some(if new_state == 1 {1} else {-1})
        };
        self.set_state(new_state == 1);


        rval
    }
}*/

/// the only cases that matter at the hot code sections
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum RunTimeGateType {
    OrNand,
    AndNor,
    XorXnor,
}
impl RunTimeGateType {
    fn new(kind: GateType) -> Self {
        match kind {
            GateType::AND | GateType::NOR                      => RunTimeGateType::AndNor,
            GateType::OR  | GateType::NAND | GateType::CLUSTER => RunTimeGateType::OrNand,
            GateType::XOR | GateType::XNOR                     => RunTimeGateType::XorXnor,
        }
    }
}

// will only support about 128 inputs/outputs (or about 255 if wrapped add)
type AccType = i8;

// tests don't need that many indexes, but this is obviusly a big limitation.
type IndexType = u16;

/// data needed after processing network
#[derive(Debug)]
pub struct Gate {
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
            GateType::XNOR => 1,
            _ => 0,
        };
        Gate {
            inputs: Vec::new(),
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

    #[inline(always)]
    fn evaluate_from_runtime_static(acc: AccType, kind:RunTimeGateType) -> bool {
        match kind {
            RunTimeGateType::OrNand  => acc != 0,
            RunTimeGateType::AndNor  => acc == 0,
            RunTimeGateType::XorXnor => acc & 1 == 1,
        } 
    }
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
    //gate_flags: Vec<GateFlags>,


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
        assert!(self.gates.len() < IndexType::MAX as usize);
        next_id
    }

    /// Add inputs to `gate_id` from `inputs`.
    /// Connection must be between cluster and a non cluster gate 
    /// and a connection can only be made once for a given pair of gates.
    /// Panics if precondition is not held.
    pub fn add_inputs(&mut self, kind: GateType, gate_id: usize, inputs: Vec<usize>) {
        assert!(!self.initialized);
        //debug_assert!(inputs.len()!=0);//TODO: remove me
        let gate = &mut self.gates[gate_id];
        gate.add_inputs(inputs.len() as i32);
        let mut in2 = Vec::new();
        for input in &inputs {
            in2.push(*input as IndexType);
        }
        gate.inputs.append(&mut in2);
        gate.inputs.sort_unstable();
        gate.inputs.dedup();
        for input_id in inputs {
            assert!(input_id<self.gates.len(), "Invalid input index {input_id}");
            assert_ne!((kind == GateType::CLUSTER),(self.gates[input_id].kind == GateType::CLUSTER), "Connection was made between cluster and non cluster for gate {gate_id}");
            // panics if it cannot fit in IndexType
            self.gates[input_id].outputs.push(gate_id.try_into().unwrap());
            self.gates[input_id].outputs.sort_unstable();
            // TODO: add this back.
            //for output in &self.gates[input_id].outputs {
            //    assert_ne!(*output,gate_id as IndexType, "Connection was made multiple times for gate {gate_id} to gate {output}");
            //}
        }
    }

    pub fn get_state(&self, gate_id: usize) -> bool {
        assert!(self.initialized);
        //self.gates[gate_id].state
        self.state[gate_id]
        //self.gate_flags[gate_id].state()
    }
    #[inline(always)]
    pub fn update(&mut self) {
        assert!(self.initialized);
        // TODO: swap buffers instead of 2 lists.
        // allow gate to add to "wrong" update list
        // after network optimization
        let mut cluster_update_list = Vec::new();
        for gate_id in &self.update_list {
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
                    &mut self.in_update_list,
                    //&mut self.gate_flags,
                    //&self.runtime_gate_kind
                    );
            }
        }
        self.update_list.clear();
        // TODO: call diffrent update function that makes more assumptions here.
        // this will be guaranteed safe since shape of network is known.
        for cluster_id in &cluster_update_list {
            GateNetwork::update_kind(
                *cluster_id,
                RunTimeGateType::OrNand,
                &mut self.update_list,
                &self.packed_outputs,
                &self.packed_output_indexes,
                &mut self.acc,
                &mut self.state,
                &mut self.in_update_list,
                //&mut self.gate_flags,
                //&self.runtime_gate_kind
                );
        }
    }
    /// Adds all gates to update list and performs initialization
    /// and TODO: network optimizaton.
    /// Currently cannot be modified after initialization.
    pub fn init_network(&mut self) {
        assert!(!self.initialized);

        // add all gates to update list.
        for gate_id in 0..self.gates.len() {
            let kind = self.gates[gate_id].kind;
            if kind != GateType::CLUSTER {
                self.update_list.push(gate_id.try_into().unwrap()); 
                self.gates[gate_id].in_update_list = true;
            }
        } 

        // pack outputs
        for gate_id in 0..self.gates.len() {
            let gate = &self.gates[gate_id];
            self.packed_output_indexes.push(self.packed_outputs.len().try_into().unwrap());
            self.packed_outputs.append(&mut gate.outputs.clone());
        }
        self.packed_output_indexes.push(self.packed_outputs.len().try_into().unwrap());

        // pack gatetype, acc, state
        for gate_id in 0..self.gates.len() {
            let gate = &self.gates[gate_id];
            self.runtime_gate_kind.push(RunTimeGateType::new(gate.kind));
            self.acc.push(gate.acc);
            self.state.push(gate.state); 
            self.in_update_list.push(gate.in_update_list);
            //self.gate_flags.push(GateFlags::new(gate.kind));
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

    /*fn generate_gate_key(gate: &Gate) -> (GateType, Vec<IndexType>) {
        let mut kind = gate.kind;
        let inputs = gate.inputs.clone();
        let input_count = gate.inputs.len();

        let buffer_gate = GateType::OR;
        let nor_gate = GateType::NOR;

        if input_count < 2 {kind = match kind {
            GateType::AND => buffer_gate,
            GateType::OR => buffer_gate,
            GateType::NOR => nor_gate,
            GateType::NAND => nor_gate,
            GateType::XOR => buffer_gate,
            GateType::XNOR => nor_gate,
            GateType::CLUSTER => buffer_gate}
        }

        (kind, inputs)
    }*/

    #[inline(always)]
    fn update_kind(
        id: IndexType,
        kind: RunTimeGateType,
        update_list: &mut Vec<IndexType>,
        packed_outputs: &[IndexType],
        packed_output_indexes: &[IndexType],
        acc: &mut Vec<AccType>,
        state: &mut Vec<bool>,
        in_update_list: &mut Vec<bool>,
        //gate_flags: &mut Vec<GateFlags>,
        //kinds: & Vec<RunTimeGateType>
        ) {

        // if this assert fails, the system will recover anyways
        // but that would probably have been caused by a bug.
        assert!(in_update_list[id as usize], "{id:?}"); 


        unsafe {
            //let next = Gate::evaluate_from_runtime_static(gates.get_unchecked(id as usize).acc, kind);
            let next = Gate::evaluate_from_runtime_static(*acc.get_unchecked(id as usize), kind);
            if *state.get_unchecked(id as usize) != next {
                let delta = if next {1} else {-1};
                for i in *packed_output_indexes.get_unchecked(id as usize)..*packed_output_indexes.get_unchecked(id as usize+1) {
                    let output_id = packed_outputs.get_unchecked(i as usize);
                    *acc.get_unchecked_mut(*output_id as usize) += delta;
                    //TODO: only add if state will likley change.
                    //if Gate::evaluate_from_runtime_static(
                    //    *acc.get_unchecked(*output_id as usize),
                    //    *kinds.get_unchecked(*output_id as usize)) 
                    //== *state.get_unchecked(*output_id as usize){continue}

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
