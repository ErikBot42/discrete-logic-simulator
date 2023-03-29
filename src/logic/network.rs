//! network.rs: Manage and optimize the network while preserving behaviour.
use crate::logic::{gate_status, Gate, GateKey, GateType, IndexType, UpdateStrategy};
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::repeat;
use std::ops::Index;

fn reorder_by_indices<T>(data: &mut [T], indices: Vec<usize>) {
    reorder_by_indices_with(|a, b| data.swap(a, b), indices)
}
fn reorder_by_indices_with<F: FnMut(usize, usize)>(mut swap_indices: F, mut indices: Vec<usize>) {
    for i in 0..indices.len() {
        if indices[i] != i {
            let mut current = i;
            loop {
                let target = indices[current];
                indices[current] = current;
                if indices[target] == target {
                    break;
                }
                swap_indices(current, target);
                current = target;
            }
        }
    }
}
pub(crate) trait CsrIndex:
    Copy + Default + Hash + Ord + Debug + 'static + TryInto<usize> + TryFrom<usize>
{
}
impl<T> CsrIndex for T where
    T: Copy + Default + Hash + Ord + Debug + 'static + TryInto<usize> + TryFrom<usize>
{
}
#[derive(Clone)]
pub(crate) struct Csr<T: CsrIndex> {
    pub(crate) indexes: Vec<T>,
    pub(crate) outputs: Vec<T>,
}
impl<T: CsrIndex> Csr<T>
where
    <T as TryFrom<usize>>::Error: Debug,
    <usize as TryFrom<T>>::Error: Debug,
    usize: TryFrom<T>,
    Csr<T>: Index<usize, Output = [T]>,
{
    pub(crate) fn new(outputs_iter: impl IntoIterator<Item = impl IntoIterator<Item = T>>) -> Self {
        let mut this = Self::default();
        for gate_outputs in outputs_iter {
            this.push(gate_outputs);
        }
        this
    }
    fn push(&mut self, new_outputs: impl IntoIterator<Item = T>) {
        self.outputs.extend(new_outputs);
        self.indexes.push(self.outputs.len().try_into().unwrap());
    }
    pub(crate) fn len(&self) -> usize {
        self.indexes.len() - 1
    }
    pub(crate) fn adjacency_iter(&self) -> impl Iterator<Item = (T, T)> + '_ {
        (0..self.len())
            .map(|i| std::iter::repeat(i.try_into().unwrap()).zip(self.index(i).iter().cloned()))
            .flatten()
    }
    pub(crate) fn from_adjacency(mut adjacency: Vec<(T, T)>, len: usize) -> Self {
        use either::Either::{Left, Right};
        use std::mem::replace;
        adjacency.sort();
        let outputs_grouped = adjacency.into_iter().dedup().group_by(|a| a.0);
        let mut outputs_iter = outputs_grouped
            .into_iter()
            .map(|(from, outputs)| (from, outputs.map(|(_, output)| output)));

        let mut curr_output = outputs_iter.next();
        let outputs_iter = (0..len).map(|id| {
            if (curr_output.as_ref().map(|(from, _)| id == usize::try_from(*from).unwrap()) == Some(true)) &&
                let Some(iter) = replace(&mut curr_output, outputs_iter.next()).map(|a| a.1) {
                Left(iter)
            } else {
                Right([T::try_from(0_usize).unwrap(); 0].into_iter())
            }
        });
        Self::new(outputs_iter)
    }
    pub(crate) fn as_csc(&self) -> Self {
        Self::from_adjacency(
            self.adjacency_iter().map(|(from, to)| (to, from)).collect(),
            self.len(),
        )
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = &[T]> {
        (0..self.len()).map(|i| &self[i])
    }
}
impl<T: CsrIndex> Index<usize> for Csr<T>
where
    <T as TryInto<usize>>::Error: Debug,
{
    type Output = [T];
    fn index(&self, i: usize) -> &Self::Output {
        let from: usize = self.indexes[i].try_into().unwrap();
        let to: usize = self.indexes[i + 1].try_into().unwrap();
        &self.outputs[from..to]
    }
}
impl<T: CsrIndex> Default for Csr<T>
where
    T: std::convert::TryFrom<usize>,
    <T as TryFrom<usize>>::Error: Debug,
{
    fn default() -> Self {
        Self {
            indexes: vec![0_usize.try_into().unwrap()],
            outputs: Vec::new(),
        }
    }
}

#[derive(Hash, Clone, Eq, PartialEq)]
struct GateNode {
    kind: GateType, // add constant on/off nodes?
    initial_state: bool,
}
//struct GateNodeFinal {
//    kind: GateType,
//    initial_state: bool,
//    acc_offset: usize,
//    constant: Option<bool>
//}
struct CsrGraph<T: CsrIndex> {
    /// mapping from original nodes to optimized nodes
    /// |n| -> |k|
    table: Vec<T>,
    /// |k|
    csr: Csr<T>,
    /// |k|
    nodes: Vec<GateNode>,
}
impl<T: CsrIndex> CsrGraph<T>
where
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    fn new(csr: Csr<T>, nodes: Vec<GateNode>) -> Self {
        Self {
            table: (0..nodes.len()).map(|i| i.try_into().unwrap()).collect(),
            csr,
            nodes,
        }
    }
}
mod passes {
    use super::*;
    // https://faultlore.com/blah/oops-that-was-important/
    //
    // TODO: keep a "futures" csc that invalidates itself? put feature in Csr?

    /// Normalize gatetype
    /// (node, inputs) -> node, stable
    fn node_normalization_pass<T: CsrIndex>(graph: &mut CsrGraph<T> /*&mut Option<Csc>*/)
    where
        <T as TryFrom<usize>>::Error: Debug,
        <usize as TryFrom<T>>::Error: Debug,
        usize: TryFrom<T>,
        Csr<T>: Index<usize, Output = [T]>,
    {
        let csc = graph.csr.as_csc();
        for (node, input_cardinality) in graph.nodes.iter_mut().zip(csc.iter().map(|i| i.len())) {
            use GateType::*;
            node.kind = match input_cardinality {
                0 | 1 => match node.kind {
                    And | Or | Xor => Or,
                    Nor | Nand | Xnor => Nor,
                    Latch => Latch,
                    Interface(s) => Interface(s),
                    Cluster => Cluster,
                },
                _ => node.kind,
            };
        }
    }
    /// ASSUME: csr outputs sorted (why?)
    /// ASSUME: input is Csc
    fn node_merge_pass<T: CsrIndex>(
        csc: &mut Csr<T>,
        nodes: Vec<GateNode>, /*&mut Option<Csc>*/
    ) where
        <T as TryFrom<usize>>::Error: Debug,
        <usize as TryFrom<T>>::Error: Debug,
        usize: TryFrom<T>,
        Csr<T>: Index<usize, Output = [T]>,
    {
        // gate node + inputs -> new id
        let mut map: HashMap<(&GateNode, &[T]), usize> = HashMap::new();
        let mut table = Vec::new();
        for (node, inputs) in nodes.iter().zip(csc.iter()) {
            table.push(if let Some(id) = map.get(&(node, inputs)) {
                *id
            } else {
                let id = table.len();
                map.insert((node, inputs), id);
                id
            })
        }
    }

    //TODO: * OPTIM: node normalization (1 or 0 input case)
    //TODO: * OPTIM: merge identical nodes (-> extra outputs)
    //TODO: * OPTIM: remove redundant connections
    //TODO: * OPTIM: constant propagation (remove/disconnect outputs of gates with zero inputs)
    //TODO: * OPTIM: detect what gates will be constants (by generating acc ranges?)
    //TODO: * OPTIM: logical optimizations, preserving external view (very hard)
}
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
#[derive(Debug, Clone)]
pub struct InitializedNetwork {
    pub(crate) gates: Vec<Gate>,
    pub(crate) translation_table: Vec<IndexType>,
}
impl InitializedNetwork {
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
        let mut explored: Vec<_> = (0..old_gates.len()).map(|_| false).collect();
        for (old_gate_id, old_gate) in old_gates.iter().enumerate() {
            let new_id = old_to_new_id[old_gate_id];
            let new_id_u = new_id as usize;
            if explored[new_id_u] {
                continue;
            }
            explored[new_id_u] = true;
            let new_gate: &mut Gate = &mut new_gates[new_id_u];
            let new_inputs: &mut Vec<IndexType> = &mut old_gate
                .inputs
                .clone()
                .into_iter()
                .map(|x| old_to_new_id[x as usize] as IndexType)
                .collect();
            new_gate.inputs.append(new_inputs);
            new_gate.inputs.sort_unstable();
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
                new_gates.push(Gate::new(old_gate.kind, old_gate.initial_state));
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
    fn optimization_pass_remove_redundant(&self) -> Self {
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
    fn optimize_remove_redundant(&self) -> InitializedNetwork {
        let mut prev_network_gate_count = self.gates.len();
        let mut new_network = self.optimization_pass_remove_redundant();
        loop {
            if new_network.gates.len() == prev_network_gate_count {
                break new_network;
            }
            prev_network_gate_count = new_network.gates.len();
            new_network = new_network.optimization_pass_remove_redundant();
        }
    }

    /// Tries to reorder in a way that is better for the cache.
    fn optimize_reorder_cache(&self) -> InitializedNetwork {
        self.reordered_by(|v| {
            //TODO: partition in place

            //return v;
            // sorting by input ids implicitly sorts by cluster/non cluster

            let mut active_set: Vec<usize> = Vec::new();
            let mut next_active_set: Vec<usize> = Vec::new();
            let mut visited: Vec<bool> = Vec::new();
            let mut constness_level: Vec<Option<usize>> = v
                .iter()
                .enumerate()
                .map(|(i2, (i, g))| {
                    assert_eq!(i2, *i);
                    if g.is_propably_constant() {
                        active_set.push(*i);
                        visited.push(true);
                        Some(0)
                    } else {
                        visited.push(false);
                        None
                    }
                })
                .collect();
            dbg!(&active_set);
            while !active_set.is_empty() {
                for active_id in active_set.iter().copied() {
                    assert!(constness_level[active_id].is_some());
                    //dbg!(active_id);
                    'foo: for output_id in v[active_id].1.outputs.iter().map(|&i| i as usize) {
                        if visited[output_id] {
                            println!("already visited {output_id}");
                            continue;
                        }
                        //dbg!(output_id);
                        let mut max = None;
                        for input_id in v[output_id].1.inputs.iter().map(|&i| i as usize) {
                            let c = constness_level[input_id];
                            //dbg!(input_id, c);
                            if c.is_none() {
                                continue 'foo;
                            }
                            max = max.max(c);
                        }
                        assert!(max.is_some(), "{max:?}");
                        visited[output_id] = true;
                        constness_level[output_id] = max.map(|x| x + 1);
                        next_active_set.push(output_id);
                        println!("ADDING CONST: {active_id} -> {output_id}");
                    }
                }
                std::mem::swap(&mut active_set, &mut next_active_set);
                next_active_set.clear();
            }
            //let (_, input_count_without_const): (Vec<_>, Vec<_>) = v
            //    .iter()
            //    .map(|(_, g)| {
            //        (
            //            g.inputs.len(),
            //            g.inputs
            //                .iter()
            //                .filter(|&&i| constness_level[i as usize].is_none())
            //                .count(),
            //        )
            //    })
            //    .unzip();

            dbg!(&active_set, &next_active_set);

            let (mut constant, mut dynamic): (Vec<_>, Vec<_>) =
                v.iter().partition(|(i, _)| constness_level[*i].is_some());
            let is_dynamic = |id: usize| constness_level[id].is_none();

            // PROP:
            // add gate, add their outputs
            // what order should outputs have?

            // TODO: make nearby have overlapping outputs
            // TODO: recursive sibling ids

            /*let sibling_ids_not_const2 = |id: usize| {
                assert_eq!(v[id].0, id);
                let mut a: Vec<_> = v[id]
                    .1
                    .inputs
                    .iter()
                    .filter(|id| is_dynamic(**id as usize))
                    .map(|id| {
                        v[*id as usize]
                            .1
                            .inputs
                            .iter()
                            .cloned()
                            .filter(|id| is_dynamic(*id as usize))
                    })
                    .flatten()
                    .map(|id| v[id as usize].1.outputs.iter().cloned())
                    .flatten()
                    .map(|id| v[id as usize].1.outputs.iter().cloned())
                    .flatten()
                    .collect();
                a.sort();
                a.dedup();
                a
            };*/
            let sibling_ids_not_const = |id: usize| {
                assert_eq!(v[id].0, id);
                let mut a: Vec<_> = v[id]
                    .1
                    .inputs
                    .iter()
                    .filter(|id| is_dynamic(**id as usize))
                    .map(|id| v[*id as usize].1.outputs.iter().cloned())
                    .flatten()
                    .collect();
                a.sort();
                a.dedup();
                a
            };
            //let sibling_ids = sibling_ids(*ia);
            //let sibling_ids = if sibling_ids.contains(&(*ib as u32)) {
            //    ia.cmp(&ib)
            //} else {
            //    std::cmp::Ordering::Equal
            //};

            dynamic.sort_by(|(ia, a), (ib, b)| {
                //let input_degree = a.inputs.len().cmp(&b.inputs.len()).reverse();
                //let output_degree = a.outputs.len().cmp(&b.outputs.len());
                //let input_degree_exclude_const = input_count_without_const[*ia]
                //    .cmp(&input_count_without_const[*ib])
                //    .reverse();

                let is_cluster = a.kind.is_cluster().cmp(&b.kind.is_cluster());
                let input_ids = a.inputs.cmp(&b.inputs);
                let output_ids = a.outputs.cmp(&b.outputs).reverse();
                //by_is_cluster.then(by_input_degree_exclude_const)
                //by_is_cluster.then(by_input_degree).then(by_output_degree)
                //.then(by_input_degree)

                //is_cluster.then(input_ids).then(output_ids)
                //is_cluster
                //    .then(sibling_ids)
                is_cluster.then(input_ids.then(output_ids))
            });

            //dynamic.iter().for_each(|(id, _)| {
            //    dbg!(sibling_ids_not_const(*id).len());
            //});
            {
                let mut counts_vec: Vec<(usize, usize)> = dynamic
                    .iter()
                    .map(|(id, _)| sibling_ids_not_const(*id).len())
                    .counts()
                    .into_iter()
                    .collect();
                counts_vec.sort_unstable();
                for (value, count) in counts_vec {
                    println!("{value}: {count}");
                }
            }

            //panic!();
            dynamic.sort_by_key(|(ia, a)| sibling_ids_not_const(*ia).len());
            dynamic.reverse();
            let mut dynamic = {
                let mut not_added: Vec<_> = v.iter().map(|(id, _)| is_dynamic(*id)).collect();
                let mut acc: Vec<(usize, &Gate)> = Vec::new();
                for c in dynamic.iter().cloned() {
                    let sibling_ids_not_added: Vec<_> = sibling_ids_not_const(c.0)
                        .iter()
                        .map(|id| *id as usize)
                        .filter(|id| not_added[*id])
                        .collect();
                    for id in sibling_ids_not_added.iter() {
                        not_added[*id] = false
                    }
                    acc.extend(sibling_ids_not_added.iter().map(|id| v[*id]));
                }
                acc
            };

            //panic!("{:?}, {:?}", dynamic[0], dynamic[dynamic.len() - 1]);

            dynamic.append(&mut constant);

            //panic!();

            dynamic

            /*let mut out = Vec::new();
            out.push(dynamic.pop().unwrap());

            struct Score {
                foo: u8,
            }

            // Score is number of overlapping inputs
            // on tie, use first

            fn count_overlapping_inputs(a: &Gate, b: &Gate) -> usize {
                let a: HashSet<_> = a.inputs.iter().copied().collect();
                let b: HashSet<_> = b.inputs.iter().copied().collect();
                a.intersection(&b).count()
            }

            let limit = 256;

            loop {
                // score, index
                let mut curr_best: Option<(usize, usize)> = None;
                let compare_with = out.last().unwrap().1;
                for (i, (_, gate)) in dynamic.iter().enumerate().take(limit) {
                    let score = count_overlapping_inputs(compare_with, gate);
                    let new_entry = (score, i);
                    curr_best = Some(match curr_best {
                        None => new_entry,
                        Some(curr_best) => {
                            if curr_best.0 < score {
                                new_entry
                            } else {
                                curr_best
                            }
                        },
                    });
                }
                let (_, index) = unwrap_or_else!(curr_best, break);
                out.push(dynamic.swap_remove(index));
            }
            out*/
        })
    }

    /// Change order of gates and update ids afterwards, might be better for cache.
    /// Removing gates is UB, adding None is used to add padding.
    ///
    /// O(n * k) + O(reorder(n, k))
    fn reordered_by<F: FnMut(Vec<(usize, &Gate)>) -> Vec<(usize, &Gate)>>(
        &self,
        mut reorder: F,
    ) -> Self {
        let gates_with_ids: Vec<(usize, &Gate)> = self.gates.iter().enumerate().collect();

        let gates_with_ids = reorder(gates_with_ids);

        let (inverse_translation_table, gates): (Vec<usize>, Vec<&Gate>) =
            gates_with_ids.into_iter().unzip();
        assert_eq_len!(gates, inverse_translation_table);
        let mut translation_table: Vec<IndexType> = (0..inverse_translation_table.len())
            .map(|_| 0 as IndexType)
            .collect();
        assert_eq_len!(gates, translation_table);
        inverse_translation_table
            .iter()
            .enumerate()
            .for_each(|(index, new)| {
                //if let Some(new) = new {
                translation_table[*new] = index.try_into().unwrap();
                //}
            });
        //TODO: avoid allocations here
        let gates: Vec<Gate> = gates
            .into_iter()
            .map(|gate| {
                //gate.map(|gate| {
                let mut gate = gate.clone();
                gate.outputs.iter_mut().for_each(|output| {
                    *output = translation_table[*output as usize] as IndexType;
                });
                gate.inputs
                    .iter_mut()
                    .for_each(|input| *input = translation_table[*input as usize] as IndexType);
                gate.outputs.sort_unstable();
                gate.inputs.sort_unstable();
                gate
                //})
            })
            .collect();
        assert_eq_len!(gates, translation_table);
        //assert_le_len!(self.translation_table, translation_table);
        let translation_table =
            Self::create_translation_table(&self.translation_table, &translation_table);
        for t in &translation_table {
            assert_le!(*t as usize, gates.len());
        }
        Self {
            gates,
            translation_table,
        }
    }

    /// Perform repeated optimization passes.
    fn optimized(&self) -> Self {
        timed!(
            {
                self.print_info();
                //let network = self.clone();
                let network = self.optimize_remove_redundant().optimize_reorder_cache();
                //network.clone()._fgo_connections_grouping();
                network.print_info();
                network
            },
            "optimized network in: {:?}"
        )
    }

    /// In order for scalar packing optimizations to be sound,
    /// cluster and non cluster cannot be mixed
    fn prepare_for_scalar_packing(&self) -> NetworkWithGaps {
        self.reordered_by_gaps(|v| {
            Self::aligned_by_inner(
                v,
                gate_status::PACKED_ELEMENTS,
                Gate::is_cluster_a_xor_is_cluster_b,
            )
        })
    }
    pub(crate) fn prepare_for_bitpack_packing_no_type_overlap_equal_cardinality(
        &self,
        bits: usize,
    ) -> NetworkWithGaps {
        self.reordered_by_gaps(|v| {
            Self::aligned_by_inner(
                v,
                bits,
                Gate::is_cluster_a_xor_is_cluster_b_and_no_type_overlap_equal_cardinality,
            )
        })
    }
    pub(crate) fn prepare_for_bitpack_packing_no_type_overlap(
        &self,
        bits: usize,
    ) -> NetworkWithGaps {
        self.reordered_by_gaps(|v| {
            Self::aligned_by_inner(
                v,
                bits,
                Gate::is_cluster_a_xor_is_cluster_b_and_no_type_overlap,
            )
        })
    }
    pub(crate) fn prepare_for_bitpack_packing(&self, bits: usize) -> NetworkWithGaps {
        self.reordered_by_gaps(|v| {
            Self::aligned_by_inner(v, bits, Gate::is_cluster_a_xor_is_cluster_b)
        })
    }

    /// List will have each group of `elements` in such that cmp will return false.
    /// Will also make sure list is a multiple of `elements`
    /// Order is maybe preserved to some extent.
    /// This is just a heuristic, solving it without inserting None is sometimes impossible
    /// Solving it perfectly is probably NP-hard.
    /// `cmp` has no restrictions.
    /// O(n^2), ~O(n) in practice
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
    fn reordered_by_gaps<F: FnMut(Vec<(usize, &Gate)>) -> Vec<Option<(usize, &Gate)>>>(
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
                    gate.outputs.sort_unstable();
                    gate.inputs.sort_unstable();
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

    fn _fgo_connections_grouping(self) {
        let gates = self.gates;

        #[derive(Eq, PartialEq, Hash, Ord, PartialOrd, Debug, Copy, Clone)]
        enum FgoGateTargetBad {
            Other,
            Cluster,
        }
        let gate_kind_mapping_bad = |g: GateType| match g {
            GateType::Cluster => FgoGateTargetBad::Cluster,
            _ => FgoGateTargetBad::Other,
        };

        #[derive(Eq, PartialEq, Hash, Ord, PartialOrd, Debug, Copy, Clone)]
        enum FgoGateTarget {
            AndNor,
            OrNand,
            XorXnor,
            Latch,
            Interface(Option<u8>),
            Cluster,
        }
        let gate_kind_mapping = |g: GateType| match g {
            GateType::And | GateType::Nor => FgoGateTarget::AndNor,
            GateType::Or | GateType::Nand => FgoGateTarget::OrNand,
            GateType::Cluster => FgoGateTarget::Cluster,
            GateType::Xor | GateType::Xnor => FgoGateTarget::XorXnor,
            GateType::Latch => FgoGateTarget::Latch,
            GateType::Interface(s) => FgoGateTarget::Interface(s),
        };
        fgo::fgo_connections_grouping(gates, gate_kind_mapping);
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
    pub(crate) fn add_vertex(&mut self, kind: GateType, initial_state: bool) -> usize {
        let next_id: IndexType = self.network.gates.len().try_into().unwrap();
        self.network.gates.push(Gate::new(kind, initial_state));
        next_id.try_into().unwrap()
    }

    /// Add inputs to `gate_id` from `inputs`.
    /// Connection must be between cluster and a non cluster gate
    /// and a connection can only be made once for a given pair of gates.
    /// # Panics
    /// If preconditions are not held.
    pub(crate) fn add_inputs(&mut self, kind: GateType, gate_id: usize, inputs: Vec<usize>) {
        let gate = &mut self.network.gates[gate_id];

        gate.add_inputs_vec(&mut inputs.iter().map(|&i| i.try_into().unwrap()).collect());

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
                "Connection was made between cluster and non cluster for gate {gate_id}: {kind:?} {:?}",
                self.network.gates[input_id].kind
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
    pub(crate) fn compiled<T: crate::logic::LogicSim>(self, optimize: bool) -> (Vec<IndexType>, T) {
        T::create(self.network.initialized(optimize))
    }

    pub(crate) fn initialized(self, optimize: bool) -> InitializedNetwork {
        self.network.initialized(optimize)
    }
}

mod fgo {

    //! TODO:
    //! first: make pass with optimistic oid_recursive for each group, save remaining things
    //! then: collect remaining and repeat until level is 2 (oid + kind)
    //!
    //! while: explore outputs and try add them, if unsucessfull add group exploration candidates to list for next iteration. Grouping outputs is done unconditionally.
    //!
    //! finally: put remaining trash into groups.
    //!
    //!
    //! (A,B) -> (C,C) will not work, C must be in 2 places at once.
    //! (A,B) -> (C,D) -> (E, E) will not work for (C,D) because E must be in 2 places at once
    //! (A,B) -> [(C,D), (E,F)] -> (G, H)  will work, but recursive id search will fail here.
    //!
    //! We know for sure that:
    //! * immediate outputs within a group must have disjoint ids in order to become an afgo
    //! * graph is bipartite
    //! * cycles can and do exist. <- can we do analysis without cycles?
    //!
    //! What happens in non-synchronus circuits?
    //!
    //! A -> (B,C)
    //! D -> (E,F)
    //!
    //! B -> C
    //! E -> F
    //!
    //! => (A,D) -> [(B,E), (C,F)], (B,E) -> (C,F)
    //!
    //! => all graphs for each pos have to be entirely disjoint for this to work...
    //!

    use super::*;
    use nohash_hasher::{IntMap, IntSet};

    // output kind key/id, MUST match
    type Oid<G> = Vec<G>;

    // fgo output key/id
    // (fgo group, position in group)
    type Fid = Vec<Option<usize>>;

    type Fgo<const SIZE: usize> = [usize; SIZE];
    type Fg<const SIZE: usize> = [usize; SIZE];

    fn fg_validate<G: Index<usize, Output = G> + Copy + Ord + Eq, const SIZE: usize>(
        maybe_fg: Fg<SIZE>,
        kind: G,
    ) {
        assert_eq!(maybe_fg.into_iter().sorted().dedup().count(), SIZE);
        assert!(maybe_fg.into_iter().map(|id| kind[id]).all_equal());
    }
    fn fgo_validate<G: Index<usize, Output = G> + Copy + Ord + Eq, const SIZE: usize>(
        maybe_fg: Fgo<SIZE>,
        kind: G,
    ) {
        fg_validate(maybe_fg, kind)
    }

    fn afgo_validate<G: Index<usize, Output = G> + Copy + Ord + Eq, const SIZE: usize>(
        maybe_fg: Fgo<SIZE>,
        outputs: &[Vec<usize>],
        kind: G,
    ) -> bool {
        fg_validate(maybe_fg, kind);
        let fg = maybe_fg;
        let output_count = match fg.iter().map(|&i| outputs[i].len()).dedup().exactly_one() {
            Ok(c) => c,
            Err(_) => return false,
        };

        let maybe_fgos = (0..output_count)
            .map(|i| fg.map(|id| outputs[id][i]))
            .collect::<Vec<_>>();
        for &maybe_fgo in &maybe_fgos {
            fgo_validate(maybe_fgo, kind)
        }

        assert_eq!(
            fg.iter()
                .copied()
                .chain(maybe_fgos.iter().copied().flatten())
                .sorted()
                .dedup()
                .count(),
            SIZE * (output_count + 1)
        );
        let fgos = maybe_fgos;

        todo!()
    }
    /// # GG (Gate Group)
    /// * Set of gate/cluster ids.
    /// # FG (Fused Group)
    /// * \|FG\| = `SIZE`
    /// * \|Set(FG)\| = `SIZE`
    /// * \|Set(Map(FG, kind))\| = 1
    /// # FGO (Fused Group Output)
    /// * \|FGO\| = `SIZE`
    /// * \|Set(FGO)\| = `SIZE`
    /// * \|Set(Map(FGO, kind))\| = 1
    /// # AFGO (Array FGO)
    /// * FG, where all outputs are FGO, everything disjoint.
    /// * Disjoint check
    pub(super) fn fgo_connections_grouping<F, G>(gates: Vec<Gate>, gate_kind_mapping: F)
    where
        F: Fn(GateType) -> G,
        G: Eq + PartialEq + Hash + Ord + Copy + Clone + Debug,
    {
        // NOTE: 2x32 output groups viable because of how SIMD is done.
        const SIZE: usize = 2048;

        let ids = (0..gates.len()).collect::<Vec<_>>();
        let (kind, mut outputs, _inputs): (Vec<_>, Vec<_>, Vec<_>) = gates
            .into_iter()
            .map(|g| {
                (
                    gate_kind_mapping(g.kind),
                    g.outputs
                        .into_iter()
                        .map(|i| i as usize)
                        .collect::<Vec<_>>(),
                    g.inputs.into_iter().map(|i| i as usize).collect::<Vec<_>>(),
                )
            })
            .multiunzip();

        // sort gate outputs by kind
        for &i in ids.iter() {
            outputs[i].sort_by_key(|&i| kind[i]);
        }

        let static_candidate_ggs = timed! {static_candidate_groups(&ids, &kind, &outputs), "made static candidates in {:?}"};
        let static_candidate_map: IntMap<usize, &StaticCandidateGroup<G>> = static_candidate_ggs
            .iter()
            .flat_map(|scg| scg.group.iter().copied().zip(repeat(scg)))
            .collect();

        let mut fg_infos: Vec<Option<FgoInfo<SIZE>>> = ids.iter().map(|_| None).collect();
        let mut fgs: Vec<Fgo<SIZE>> = Vec::new();
        let mut afgs: Vec<Fgo<SIZE>> = Vec::new();
        for scg in static_candidate_ggs.iter() {
            // ADD CONSTRAINT: kind
            // ADD CONSTRAINT: oid
            // ARBITRARY: scg exploration order.
            let mut group = scg.group.clone();
            dbg!(scg.kind, scg.group.len());
            'static_group: while {
                group.retain(|&i| {
                    fg_infos[i].is_none()
                        && outputs[i]
                            .iter()
                            .filter_map(|&output| fg_infos[output].map(|f| f.pos()))
                            .all_equal()
                });
                group.len() >= SIZE
            } {
                'fid: for (_fid, group) in group
                    .iter()
                    .map(|&id| {
                        (
                            calc_fgo_id_normalize(id, &mut outputs, &fg_infos, &kind),
                            id,
                        )
                    })
                    .into_group_map()
                    .into_iter() // for determinism
                    .filter(|(_, group)| group.len() >= SIZE)
                    .sorted_by_key(|(_,group)| group.len())
                {
                    // ADD CONSTRAINT: fid
                    let (remaining, pos_constraints) =
                        get_pos_constraints(group, &outputs, &fg_infos);
                    // ADD CONSTRAINT: pos
                    let this_fgo: [usize; SIZE] = unwrap_or_else!(
                        try_make_fgo(pos_constraints, &outputs, remaining),
                        continue 'fid
                    );

                    assert!(add_fgo(
                        &mut fgs,
                        this_fgo,
                        &mut fg_infos,
                        &kind,
                        &mut outputs,
                        true
                    ));

                    afgs.push(this_fgo);
                    {
                        let oid = &static_candidate_map[&this_fgo[0]].oid; // oid already checked
                        let output_fgos: Vec<_> = (0..oid.len())
                            .map(|i| this_fgo.map(|this_id| outputs[this_id][i]))
                            .collect();
                        // FGO is new
                        // CURRENT CONSTRAINTS: input(oid) => kind, input(pos) => pos, fid
                        // ADD CONSTRAINT: scg => oid
                        for fgo in output_fgos.iter().copied() {
                            add_fgo(&mut fgs, fgo, &mut fg_infos, &kind, &mut outputs, false);
                        }
                    }
                    continue 'static_group;
                }
                break 'static_group;
            }
        }

        {
            let mut remaining = ids.iter().copied().collect::<IntSet<_>>();
            for f in fgs.iter().flat_map(|f| f.iter()) {
                remaining.remove(f);
            }
            println!(
                "amount ungrouped: {}",
                remaining.len() as f64 / ids.len() as f64
            );
        }
        {
            let mut remaining = ids.iter().copied().collect::<IntSet<_>>();
            for f in afgs.iter().flat_map(|f| f.iter()) {
                remaining.remove(f);
            }
            println!(
                "amount non full: {}",
                remaining.len() as f64 / ids.len() as f64
            );
        }
    }

    fn static_candidate_groups<G>(
        ids: &[usize],
        kind: &[G],
        outputs: &[Vec<usize>],
    ) -> Vec<StaticCandidateGroup<G>>
    where
        G: Eq + Ord + Hash + Clone + Copy,
    {
        const LEN: usize = 20;
        let a: OidMatrix<LEN> = oid_recursive(kind, outputs);

        // Disjoint sets of gates, with MINIMAL requirements
        let hgg_better: HashMap<(G, Vec<G>, u64), Vec<usize>> =
            ids.iter().cloned().into_group_map_by(|&i| {
                (
                    kind[i],
                    outputs[i].iter().map(|&i| kind[i]).collect::<Vec<_>>(),
                    a[LEN - 1][i],
                )
            });
        let hgg_int = ids.iter().cloned().into_group_map_by(|&i| a[0][i]);

        //let foo: IntMap<_, _> = hgg_int.iter().map(|f| (f.0.clone(), f.1.clone())).collect();

        let static_candidate_groups: Vec<_> = hgg_better
            .into_iter()
            .map(|((kind, oid, extra), group)| StaticCandidateGroup {
                kind,
                oid,
                group,
                extra,
            })
            .sorted()
            .collect();
        static_candidate_groups
    }

    /// FGO data associated with single gate
    #[derive(Debug, Copy, Clone)]
    struct FgoInfo<const SIZE: usize> {
        id: usize, // what fgo group
        //id_graph: usize, // what fgo network (for merging disconnected networks)
        pos: usize, // what internal fgo position
    }
    impl<const SIZE: usize> FgoInfo<SIZE> {
        fn new(id: usize, pos: usize) -> Self {
            assert!(pos < SIZE, "{pos}");
            Self { id, pos }
        }
        fn pos(&self) -> usize {
            let pos = self.pos;
            assert!(pos < SIZE, "{pos}");
            pos
        }
    }
    /// calc fid, sort outputs to normalize it.
    fn calc_fgo_id_normalize<const SIZE: usize, G: Ord>(
        id: usize,
        outputs: &mut [Vec<usize>],
        fgo_infos: &[Option<FgoInfo<SIZE>>],
        kind: &[G],
    ) -> Fid {
        outputs[id].sort_by(|&a, &b| {
            let by_kind = kind[a].cmp(&kind[b]);
            let by_fid = fgo_infos[a].map(|f| f.id).cmp(&fgo_infos[b].map(|f| f.id));
            by_kind.then(by_fid)
        });
        outputs[id]
            .iter()
            .map(|&id| fgo_infos[id].map(|f| f.id))
            .collect()
    }
    #[derive(Ord, PartialOrd, Hash, Eq, PartialEq)]
    struct StaticCandidateGroup<G> {
        kind: G, // needed for hash
        oid: Oid<G>,
        group: Vec<usize>,
        extra: u64,
    }

    fn get_pos_constraints<const SIZE: usize>(
        group: Vec<usize>,
        outputs: &[Vec<usize>],
        fgo_infos: &[Option<FgoInfo<SIZE>>],
    ) -> (Vec<usize>, HashMap<usize, Vec<usize>>) {
        let mut remaining = Vec::new();
        let pos_constraints = group
            .iter()
            .filter_map(|&i| {
                Some((
                    match outputs[i]
                        .iter()
                        .find_map(|&output| fgo_infos[output].map(|f| f.pos()))
                    {
                        Some(pos) => pos,
                        None => {
                            remaining.push(i);
                            return None;
                        },
                    },
                    i,
                ))
            })
            .into_group_map();
        (remaining, pos_constraints)
    }

    fn try_make_fgo<const SIZE: usize>(
        pos_constraints: HashMap<usize, Vec<usize>>,
        outputs: &[Vec<usize>],
        remaining: Vec<usize>,
    ) -> Option<[usize; SIZE]> {
        let mut pos_choices: [Option<usize>; SIZE] = std::array::from_fn(|_| None);
        let mut out_set: HashSet<usize> = HashSet::new();
        for (pos, ids) in pos_constraints {
            // Using first disjoint
            for id in ids {
                let is_disjoint = outputs[id]
                    .iter()
                    .all(|output| out_set.get(output).is_none());
                if is_disjoint {
                    out_set.extend(outputs[id].iter());
                    pos_choices[pos] = Some(id);
                    break;
                }
            }
        }
        // ARBITRARY: remaining iteration order.
        let mut remaining_iter = remaining.iter().filter_map(|&id| {
            let is_disjoint = outputs[id]
                .iter()
                .all(|output| out_set.get(output).is_none());
            out_set.extend(outputs[id].iter());
            is_disjoint.then_some(id)
        });
        let this_fgo =
            pos_choices.try_map(|pos_choice| pos_choice.or_else(|| remaining_iter.next()));
        this_fgo
    }

    /// Add fgo, set fgo info
    /// Returns `true` if fgo is new
    fn add_fgo<const SIZE: usize, G: Ord>(
        fgos: &mut Vec<[usize; SIZE]>,
        new_fgo: [usize; SIZE],
        fgo_infos: &mut [Option<FgoInfo<SIZE>>],
        kind: &[G],
        outputs: &mut [Vec<usize>],
        perform_asserts: bool,
    ) -> bool {
        assert_eq!(
            new_fgo.into_iter().sorted().dedup().count(),
            SIZE,
            "fgo elements not unique: {new_fgo:?}"
        );
        if perform_asserts {
            assert!(
                new_fgo.iter().map(|&id| outputs[id].len()).all_equal(),
                "{:?}",
                new_fgo
                    .iter()
                    .map(|&id| outputs[id].len())
                    .collect::<Vec<_>>()
            );

            assert!(
                new_fgo
                    .iter()
                    .map(|&id| { calc_fgo_id_normalize(id, outputs, &fgo_infos, &kind) })
                    .all_equal(),
                "{:?} {:?}",
                new_fgo
                    .iter()
                    .map(|&id| calc_fgo_id_normalize(id, outputs, &fgo_infos, &kind))
                    .collect::<Vec<_>>(),
                &new_fgo
            );
        }

        let is_empty = new_fgo.iter().all(|&f| fgo_infos[f].is_none());
        assert_ne!(
            new_fgo.iter().all(|&f| fgo_infos[f].is_some()),
            is_empty,
            "{:?}",
            new_fgo.iter().map(|&f| fgo_infos[f]).collect::<Vec<_>>()
        );
        if is_empty {
            let next_fgo_id = fgos.len();
            fgos.push(new_fgo);

            let f_pre = new_fgo.map(|f| fgo_infos[f]);
            for (pos, &id) in new_fgo.iter().enumerate() {
                assert!(pos < SIZE);
                assert!(
                    fgo_infos[id]
                        .replace(FgoInfo::new(next_fgo_id, pos))
                        .is_none(),
                    "\ninfos: {:?}\ninfos_pre {f_pre:?}\nfgo: {new_fgo:?}\npos: {pos}\nid:{id}",
                    new_fgo.iter().map(|&f| fgo_infos[f]).collect::<Vec<_>>()
                );
            }
        }
        let is_full_after = new_fgo.iter().all(|&f| fgo_infos[f].is_some());
        assert!(is_full_after);
        if perform_asserts {
            assert!(
                new_fgo
                    .iter()
                    .map(|&id| { calc_fgo_id_normalize(id, outputs, &fgo_infos, &kind) })
                    .all_equal(),
                "fids: {:?}\n fgo: {:?}\n outputs: {:?}",
                new_fgo
                    .iter()
                    .map(|&id| calc_fgo_id_normalize(id, outputs, &fgo_infos, &kind))
                    .collect::<Vec<_>>(),
                &new_fgo,
                new_fgo
                    .iter()
                    .map(|&id| outputs[id].clone())
                    .collect::<Vec<_>>(),
            );
        }

        is_empty
    }

    fn assert_fgo<'a, const SIZE: usize, T, G>(
        this_fgo: [usize; SIZE],
        static_candidate_map: &T,
        kind: &[G],
        outputs: &mut [Vec<usize>],
        fgo_infos: &[Option<FgoInfo<SIZE>>],
    ) where
        T: for<'b> Index<&'b usize, Output = &'a StaticCandidateGroup<G>>,
        G: Ord + 'a + Copy + Clone,
    {
        let fgo = this_fgo;
        // check elements unique
        assert_eq!(
            fgo.into_iter().sorted().dedup().count(),
            SIZE,
            "fgo elements not unique: {fgo:?}"
        );

        // check kind
        assert!(fgo.iter().map(|&f| kind[f]).all_equal());
        // check oid
        assert!(fgo.iter().map(|f| &static_candidate_map[f].oid).all_equal());
        // check fid
        assert!(
            fgo.iter()
                .map(|&id| { calc_fgo_id_normalize(id, outputs, &fgo_infos, &kind) })
                .all_equal(),
            "{:?} {:?}",
            fgo.iter()
                .map(|&id| { calc_fgo_id_normalize(id, outputs, &fgo_infos, &kind) },),
            fgo.iter()
        );
        // check id equal
        assert!(fgo.iter().map(|&f| fgo_infos[f].unwrap().id).all_equal());
        // check pos
        assert!(
            fgo.iter()
                .map(|&f| fgo_infos[f].unwrap().pos())
                .eq(0..fgo.len()),
            "{:?}",
            fgo.iter().map(|&f| fgo_infos[f].unwrap().pos())
        );
    }

    type OidMatrix<const LEN: usize> = [Vec<u64>; LEN];
    fn oid_recursive<G: Hash, const LEN: usize>(
        kind: &[G],
        outputs: &[Vec<usize>],
    ) -> OidMatrix<LEN> {
        fn rec_oid_hash<'a, G: Hash>(
            kind: &'a [G],
            outputs: &[Vec<usize>],
            random_state: &RandomState,
        ) -> Vec<u64> {
            outputs
                .iter()
                .map(|outputs| {
                    let mut hasher = RandomState::build_hasher(&random_state);
                    outputs
                        .iter()
                        .for_each(|&output| kind[output].hash(&mut hasher));
                    hasher.finish()
                })
                .collect()
        }
        use std::array;
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let random_state = RandomState::new();
        let oid0 = kind
            .iter()
            .map(|k| random_state.hash_one(k))
            .collect::<Vec<_>>();
        let mut oid_iterator = itertools::iterate(oid0.clone(), |oid_prev| {
            rec_oid_hash(oid_prev, outputs, &random_state)
        });
        array::from_fn(|_| oid_iterator.next().unwrap())
    }
}
