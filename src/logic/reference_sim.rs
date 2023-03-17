//! Reference implementation for logic simulation.
//! As simple as possible, and therefore slow.
use super::{AccType, Gate, IndexType, LogicSim, RunTimeGateType};
use itertools::Itertools;
pub struct ReferenceLogicSim {
    update_list: Vec<usize>,
    cluster_update_list: Vec<usize>,
    in_update_list: Vec<bool>,
    state: Vec<bool>,
    kind: Vec<super::GateType>,
    acc: Vec<AccType>,
    acc_prev: Vec<AccType>,
    outputs: Vec<Vec<IndexType>>,
    translation_table: Vec<IndexType>,
}

impl LogicSim for ReferenceLogicSim {
    fn create(network: super::network::InitializedNetwork) -> (Vec<IndexType>, Self) {
        let gates = network.gates;
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
            acc_prev: acc.clone(),
            outputs,
            translation_table: translation_table.clone(),
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
    fn number_of_gates_external(&self) -> usize {
        self.translation_table.len()
    }
    fn update(&mut self) {
        self.update_inner(false);
        self.update_inner(true);
    }
    const STRATEGY: super::UpdateStrategy = super::UpdateStrategy::Reference;

}
impl ReferenceLogicSim {
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
