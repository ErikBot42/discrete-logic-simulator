//! network.rs: Manage and optimize the network while preserving behaviour.
use crate::logic::{
    gate_status, CompiledNetwork, Gate, GateKey, GateType, IndexType, UpdateStrategy,
};
use itertools::Itertools;
use std::collections::HashMap;

/// Iterate through all gates, skipping any
/// placeholder gates.
trait NetworkInfo {
    fn output_counts(&self) -> Vec<usize>;
    fn print_info(&self) {
        let mut counts_vec: Vec<(usize, usize)> = self
            .output_counts()
            .into_iter()
            .counts()
            .into_iter()
            .collect();
        counts_vec.sort_unstable();
        let total_output_connections: usize = counts_vec.iter().map(|(_, count)| count).sum();
        println!("-----");
        println!("Network info: ");
        println!("Number of gates: {}", self.output_counts().len());
        println!(
            "Number of connections: {}",
            self.output_counts().into_iter().sum::<usize>()
        );
        println!("Output counts total: {total_output_connections}");
        println!("Number of outputs: gates with this number of outputs");
        for (value, count) in counts_vec {
            println!("{value}: {count}");
        }
        println!("-----");
    }
}
impl NetworkInfo for EditableNetwork {
    fn output_counts(&self) -> Vec<usize> {
        self.gates.iter().map(|x| x.outputs.len()).collect()
    }
}
impl NetworkInfo for InitializedNetwork {
    fn output_counts(&self) -> Vec<usize> {
        self.gates.iter().map(|x| x.outputs.len()).collect()
    }
}
// TODO: review visibility.

/// Network that contains empty gate slots used for alignment
/// Needed to separate cluster and non cluster in packed forms.
pub struct NetworkWithGaps {
    pub(crate) gates: Vec<Option<Gate>>,
    pub(crate) translation_table: Vec<IndexType>,
}
impl NetworkWithGaps {
    fn create_from(network: InitializedNetwork) -> Self {
        Self {
            gates: network.gates.into_iter().map(Some).collect(),
            translation_table: network.translation_table,
        }
    }
}

/// Contains translation table and can no longer be edited by client.
/// Can be edited for optimizations.
pub(crate) struct InitializedNetwork {
    gates: Vec<Gate>,
    translation_table: Vec<IndexType>,
}
impl InitializedNetwork {
    pub(crate) fn with_gaps(self, strategy: UpdateStrategy) -> NetworkWithGaps {
        match strategy {
            UpdateStrategy::ScalarSimd => Self::prepare_for_scalar_packing(&self),
            _ => NetworkWithGaps::create_from(self),
        }
    }
    fn create_from(network: EditableNetwork, optimize: bool) -> Self {
        assert_ne!(network.gates.len(), 0, "no gates where added.");
        let new_network = InitializedNetwork {
            translation_table: (0..network.gates.len())
                .into_iter()
                .map(|x| x.try_into().unwrap())
                .collect(),
            gates: network.gates,
        };
        if optimize {
            new_network.optimized()
        } else {
            new_network
        }
    }

    /// Create input connections for the new gates, given the old gates.
    /// O(n * k)
    fn create_input_connections(
        new_gates: &mut [Gate],
        old_gates: &[Gate],
        old_to_new_id: &[IndexType],
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
    /// O(n * k)
    fn remove_redundant_input_connections(new_gates: &mut [Gate]) {
        for new_gate in new_gates.iter_mut() {
            new_gate.inputs.sort_unstable();
            let new_inputs = &new_gate.inputs;
            let deduped_inputs: &mut Vec<IndexType> = &mut Vec::new();
            for new_input in new_inputs {
                if let Some(previous) = deduped_inputs.last() {
                    if *previous == *new_input {
                        if new_gate.kind.can_delete_single_identical_inputs() {
                            continue;
                        } else if new_gate.kind.can_delete_double_identical_inputs() {
                            deduped_inputs.pop();
                            continue;
                        }
                    }
                }
                deduped_inputs.push(*new_input);
            }
            new_gate.inputs.clear();
            new_gate.add_inputs_vec(&mut deduped_inputs.clone());
        }
    }

    /// Create output connections from current input connections
    /// O(n * k)
    fn create_output_connections(new_gates: &mut [Gate]) {
        for gate_id in 0..new_gates.len() {
            for i in 0..new_gates[gate_id].inputs.len() {
                new_gates[new_gates[gate_id].inputs[i] as usize]
                    .outputs
                    .push(gate_id.try_into().unwrap());
            }
        }
    }

    /// Create a new merged set of nodes based on the old nodes
    /// and a translation back to the old ids.
    /// O(n)
    fn create_nodes_optimized_from(old_gates: &[Gate]) -> (Vec<Gate>, Vec<IndexType>) {
        //{
        //    let mut key_arr: Vec<_> = old_gates.iter().map(|g| g.calc_key()).collect();
        //    key_arr.sort();
        //    println!("before dedup: {}", key_arr.len());
        //    key_arr.dedup();
        //    println!("after dedup: {}", key_arr.len());
        //}
        let estimate_gates = old_gates.len();
        let mut new_gates: Vec<Gate> = Vec::with_capacity(estimate_gates);
        let mut old_to_new_id: Vec<IndexType> = Vec::with_capacity(estimate_gates);
        let mut gate_key_to_new_id: HashMap<GateKey, usize> =
            HashMap::with_capacity(estimate_gates);
        for (old_gate_id, old_gate) in old_gates.iter().enumerate() {
            let key = old_gate.calc_key();
            let new_id = new_gates.len();
            if let Some(existing_new_id) = gate_key_to_new_id.get(&key) {
                // this gate is same as other, so use other's id.
                assert!(old_to_new_id.len() == old_gate_id);
                old_to_new_id.push((*existing_new_id).try_into().unwrap());
                assert!(existing_new_id < &new_gates.len());
            } else {
                // this gate is new, so a fresh id is created.
                assert!(old_to_new_id.len() == old_gate_id);
                old_to_new_id.push(new_id.try_into().unwrap());
                new_gates.push(Gate::from_gate_type(old_gate.kind));
                gate_key_to_new_id.insert(key, new_id);
                assert!(new_id < new_gates.len(), "new_id: {new_id}");
            }
        }
        assert!(old_gates.len() == old_to_new_id.len());
        (new_gates, old_to_new_id)
    }

    /// Create translation that combines the old and new translation
    /// from outside facing ids to nodes
    /// O(n)
    fn create_translation_table(
        old_translation_table: &[IndexType],
        old_to_new_id: &[IndexType],
    ) -> Vec<IndexType> {
        {
            let v: Vec<_> = old_translation_table
                .iter()
                .map(|x| old_to_new_id[*x as usize])
                .collect();
            v
        }
    }

    /// Single network optimization pass. Much like compilers,
    /// some passes make it possible for others or the same
    /// pass to be run again.
    ///
    /// Will completely recreate the network.
    /// O(n * k)
    fn optimization_pass(&self) -> Self {
        // Iterate through all old gates.
        // Add gate if type & original input set is unique.
        let old_gates = &self.gates;
        let (mut new_gates, old_to_new_id) = Self::create_nodes_optimized_from(old_gates);
        Self::create_input_connections(&mut new_gates, old_gates, &old_to_new_id);
        Self::remove_redundant_input_connections(&mut new_gates);
        Self::create_output_connections(&mut new_gates);
        let old_translation_table = &self.translation_table;
        let new_translation_table =
            Self::create_translation_table(old_translation_table, &old_to_new_id);
        Self {
            gates: new_gates,
            translation_table: new_translation_table,
        }
    }

    /// Perform repeated optimization passes.
    fn optimized(&self) -> Self {
        timed!(
            {
                let mut prev_network_gate_count = self.gates.len();
                let mut new_network = self.optimization_pass();
                //self.print_info();
                loop {
                    //new_network.print_info();
                    if new_network.gates.len() == prev_network_gate_count {
                        break new_network;
                    }
                    prev_network_gate_count = new_network.gates.len();
                    new_network = new_network.optimization_pass();
                }
            },
            "optimized network in: {:?}"
        )
    }

    /// In order for scalar packing optimizations to be sound,
    /// cluster and non cluster cannot be mixed
    fn prepare_for_scalar_packing(&self) -> NetworkWithGaps {
        println!("SCALAR PACKING RUNNING!!!!!!!!!!!");
        self.reordered_by(|v| {
            Self::aligned_by_inner(v, gate_status::PACKED_ELEMENTS, |a, b| {
                Gate::is_cluster_a_xor_is_cluster_b(a, b)
            })
        })
    }

    /// List will have each group of `elements` in such that cmp will return false.
    /// Will also make sure list is a multiple of `elements`
    /// Order is maybe preserved to some extent.
    /// This is just a heuristic, solving it without inserting None is sometimes impossible
    /// Solving it perfectly is probably NP-hard.
    /// `cmp` has no restrictions.
    /// O(n)
    fn aligned_by_inner<F: Fn(&Gate, &Gate) -> bool>(
        mut gates: Vec<(usize, &Gate)>,
        elements: usize,
        cmp: F,
    ) -> Vec<Option<(usize, &Gate)>> {
        let mut current_group: Vec<Option<(usize, &Gate)>> = Vec::new();
        let mut final_list: Vec<Option<(usize, &Gate)>> = Vec::new();
        loop {
            match current_group.len() {
                0 => current_group.push(Some(unwrap_or_else!(gates.pop(), break))),
                n if n == elements => final_list.append(&mut current_group),
                _ => {
                    let mut index = None;
                    'o: for (i, gate) in gates.iter().enumerate().rev() {
                        for cgate in &current_group {
                            if let Some(cgate) = cgate && cmp(gate.1, cgate.1) {
                                continue 'o;
                            }
                        }
                        index = Some(i);
                        break;
                    }
                    current_group.push(index.map(|i| gates.remove(i)));
                },
            }
        }
        assert_eq!(current_group.len(), 0);
        final_list
    }

    /// Change order of gates and update ids afterwards, might be better for cache.
    /// Removing gates is UB, adding None is used to add padding.
    ///
    /// O(n * k) + O(reorder(n, k))
    fn reordered_by<F: FnMut(Vec<(usize, &Gate)>) -> Vec<Option<(usize, &Gate)>>>(
        &self,
        mut reorder: F,
    ) -> NetworkWithGaps {
        let gates_with_ids: Vec<(usize, &Gate)> = self.gates.iter().enumerate().collect();

        let gates_with_ids = reorder(gates_with_ids);

        let (inverse_translation_table, gates): (Vec<Option<usize>>, Vec<Option<&Gate>>) =
            gates_with_ids
                .into_iter()
                .map(|o| o.map_or((None, None), |(a, b)| (Some(a), Some(b))))
                .unzip();
        assert_eq_len!(gates, inverse_translation_table);
        let mut translation_table: Vec<IndexType> = (0..inverse_translation_table.len())
            .map(|_| 0 as IndexType)
            .collect();
        assert_eq_len!(gates, translation_table);
        inverse_translation_table
            .iter()
            .enumerate()
            .for_each(|(index, new)| {
                if let Some(new) = new {
                    translation_table[*new] = index.try_into().unwrap();
                }
            });
        let gates: Vec<Option<Gate>> = gates
            .into_iter()
            .map(|gate| {
                gate.map(|gate| {
                    let mut gate = gate.clone();
                    gate.outputs.iter_mut().for_each(|output| {
                        *output = translation_table[*output as usize] as IndexType;
                    });
                    gate.inputs
                        .iter_mut()
                        .for_each(|input| *input = translation_table[*input as usize] as IndexType);
                    gate
                })
            })
            .collect();
        assert_eq_len!(gates, translation_table);
        //assert_le_len!(self.translation_table, translation_table);
        let translation_table =
            Self::create_translation_table(&self.translation_table, &translation_table);
        for t in &translation_table {
            assert_le!(*t as usize, gates.len());
        }
        NetworkWithGaps {
            gates,
            translation_table,
        }
    }
}

/// Contains gate graph in order to do network optimization
/// This network has no gaps in it's layout.
/// This network can be edited from client code.
/// Nodes cannot be removed.
#[derive(Debug, Default)]
pub(crate) struct EditableNetwork {
    pub(crate) gates: Vec<Gate>,
}
impl EditableNetwork {
    pub(crate) fn initialized(self, optimize: bool) -> InitializedNetwork {
        InitializedNetwork::create_from(self, optimize)
    }
}

/// The API for creating a gate network.
#[derive(Debug, Default)]
pub(crate) struct GateNetwork {
    network: EditableNetwork,
}
impl GateNetwork {
    /// Internally creates a vertex.
    /// Returns vertex id
    /// ids of gates are guaranteed to be unique
    /// # Panics
    /// If more than `IndexType::MAX` are added, or after initialized
    pub(crate) fn add_vertex(&mut self, kind: GateType) -> usize {
        let next_id: IndexType = self.network.gates.len().try_into().unwrap();
        self.network.gates.push(Gate::from_gate_type(kind));
        next_id.try_into().unwrap()
    }

    /// Add inputs to `gate_id` from `inputs`.
    /// Connection must be between cluster and a non cluster gate
    /// and a connection can only be made once for a given pair of gates.
    /// # Panics
    /// If preconditions are not held.
    pub(crate) fn add_inputs(&mut self, kind: GateType, gate_id: usize, inputs: Vec<usize>) {
        let gate = &mut self.network.gates[gate_id];
        gate.add_inputs(inputs.len().try_into().unwrap());
        let mut in2 = Vec::new();
        for input in &inputs {
            in2.push((*input).try_into().unwrap());
        }
        gate.inputs.append(&mut in2);
        gate.inputs.sort_unstable();
        let len_before_dedup = gate.inputs.len();
        gate.inputs.dedup();
        assert_eq!(len_before_dedup, gate.inputs.len());
        for input_id in inputs {
            assert!(
                input_id < self.network.gates.len(),
                "Invalid input index {input_id}"
            );
            assert_ne!(
                kind == GateType::Cluster,
                self.network.gates[input_id].kind == GateType::Cluster,
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
    /// Should not panic.
    #[must_use]
    pub(crate) fn compiled<T: crate::logic::LogicSim>(self, optimize: bool) -> T {
        T::create(
            self.network
                .initialized(optimize)
                .with_gaps(UpdateStrategy::from(T::strategy())),
        )
    }
}
