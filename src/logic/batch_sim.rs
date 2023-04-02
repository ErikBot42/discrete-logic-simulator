//! Extra optimizations exploiting repeated updates without observation.
/*
use super::{AccType, Gate, GateType, IndexType, LogicSim, RunTimeGateType};
use itertools::Itertools;
use std::mem::{replace, take};
use std::num::NonZeroUsize;
#[derive(Clone)]
pub struct ReferenceBatchSim {
    update_list: Vec<usize>,
    cluster_update_list: Vec<usize>,
    in_update_list: Vec<bool>,
    state: Vec<bool>,
    kind: Vec<super::GateType>,
    acc: Vec<AccType>,
    acc_prev: Vec<AccType>,
    outputs: Vec<Vec<IndexType>>,
}

fn analyze_network(gates: &[Gate]) {
    fn is_pure_function(kind: GateType) -> bool {
        use GateType::*;
        match kind {
            And | Or | Nor | Nand | Xor | Xnor | Cluster => true,
            Latch | Interface(_) => false,
        }
    }

    // batch level == 0 iff #outputs = 0
    let mut next_active_set = Vec::new();
    let mut active_set = Vec::new();
    let (mut batch_levels, mut visited): (Vec<_>, Vec<_>) = gates
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let is_batch = g.outputs.len() == 0 && is_pure_function(g.kind);
            if is_batch {
                active_set.push(i)
            }
            (is_batch.then_some(NonZeroUsize::new(1).unwrap()), is_batch)
        })
        .unzip();

    while active_set.len() > 0 {
        for i in active_set {
            let g = &gates[i];
            // only keep exploring through pure functions
            if is_pure_function(g.kind) {
                // diff batch_level is ok, just take max of the branches.
                batch_levels[i] = batch_levels[i].or_else(|| {
                    g.outputs
                        .iter()
                        .map(|&i| batch_levels[i as usize])
                        .max_by(|a, b| a.is_none().cmp(&b.is_none()).then(a.cmp(&b)))
                        .flatten()
                        .map(|i| i.checked_add(1).unwrap())
                });
                for input in g.inputs.iter().filter_map(|&i| {
                    (!replace(&mut visited[i as usize], true)).then_some(i as usize)
                }) {
                    next_active_set.push(input);
                }
            }
        }
        active_set = take(&mut next_active_set);
    }
    let max_batch_level = batch_levels.iter().cloned().max().flatten();
    dbg!(batch_levels
        .iter()
        .enumerate()
        .filter_map(|(i, b)| b.is_none().then_some(i))
        .collect::<Vec<_>>());
    dbg!((batch_levels, max_batch_level));
}

impl crate::logic::RenderSim for ReferenceBatchSim {}
impl LogicSim for ReferenceBatchSim {
    fn create(network: super::network::InitializedNetwork) -> (Vec<IndexType>, Self) {
        let gates = network.gates;
        analyze_network(&gates);
        let translation_table = network.translation_table;
        let (cluster_update_list, gate_update_list) =
            gates.iter().enumerate().partition_map(|(i, g)| {
                if g.kind.is_cluster() {
                    itertools::Either::Left(i)
                } else {
                    itertools::Either::Right(i)
                }
            });
        let (in_update_list, state, kind, acc, outputs): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
            gates
                .into_iter()
                .map(|g| (true, g.initial_state, g.kind, g.calc_acc(), g.outputs))
                .multiunzip();
        let mut this = Self {
            update_list: gate_update_list,
            cluster_update_list,
            in_update_list,
            state,
            kind,
            acc: acc.clone(),
            acc_prev: acc,
            outputs,
        };
        for (state, outputs) in this.state.iter().zip(this.outputs.iter()) {
            if *state {
                for i in outputs {
                    let acc = &mut this.acc[*i as usize];
                    *acc = acc.wrapping_add(1);
                }
            }
        }
        (translation_table, this)
    }
    fn get_state_internal(&self, gate_id: usize) -> bool {
        self.state[gate_id]
    }
    fn update(&mut self) {
        self.update_inner(false);
        self.update_inner(true);
    }
    const STRATEGY: super::UpdateStrategy = super::UpdateStrategy::Batch;
    fn num_gates_internal(&self) -> usize {
        self.state.len()
    }
}
impl ReferenceBatchSim {
    fn update_inner(&mut self, cluster: bool) {
        let (update_list, next_update_list) = if cluster {
            (&mut self.cluster_update_list, &mut self.update_list)
        } else {
            (&mut self.update_list, &mut self.cluster_update_list)
        };
        for id in update_list.iter().map(|&i| i) {
            assert!(self.in_update_list[id]);
            let kind = self.kind[id];
            assert_eq!(kind.is_cluster(), cluster);
            let acc = self.acc[id];

            let state = Gate::evaluate(
                acc,
                self.acc_prev[id],
                self.state[id],
                RunTimeGateType::new(kind),
            );

            if state != self.state[id] {
                let delta = if state {
                    1
                } else {
                    (0 as AccType).wrapping_sub(1)
                };
                for id in self.outputs[id].iter().map(|&i| i as usize) {
                    let acc = &mut self.acc[id];
                    *acc = acc.wrapping_add(delta);
                    if !self.in_update_list[id] {
                        self.in_update_list[id] = true;
                        next_update_list.push(id);
                    }
                }
            }
            self.state[id] = state;
            self.acc_prev[id] = acc;
            self.in_update_list[id] = false;
        }
        update_list.clear();
    }
}
*/
