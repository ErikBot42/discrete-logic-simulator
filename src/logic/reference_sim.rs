//! Reference implementation for logic simulation.
//! As simple as possible, and therefore slow.
use super::{AccType, Gate, IndexType, LogicSim, RunTimeGateType};
use crate::logic::Csr;
use itertools::Itertools;
#[derive(Clone)]
pub struct ReferenceLogicSim {
    update_list: Vec<usize>,
    cluster_update_list: Vec<usize>,
    in_update_list: Vec<bool>,
    state: Vec<bool>,
    kind: Vec<super::GateType>,
    acc: Vec<AccType>,
    acc_prev: Vec<AccType>,
    outputs: Vec<Vec<IndexType>>,
}
impl crate::logic::RenderSim for ReferenceLogicSim {}
impl LogicSim for ReferenceLogicSim {
    fn create(
        //outputs_iter: impl IntoIterator<Item = impl IntoIterator<Item = usize>>,
        csr: Csr<u32>,
        nodes: Vec<crate::logic::network::GateNode>,
        translation_table: Vec<u32>,
    ) -> (Vec<IndexType>, Self) {
        //let csr = crate::logic::network::Csr::new(outputs_iter);
        let csc = csr.as_csc();
        let (cluster_update_list, gate_update_list): (Vec<_>, Vec<_>) =
            nodes.iter().enumerate().partition_map(|(i, n)| {
                if n.kind.is_cluster() {
                    itertools::Either::Left(i)
                } else {
                    itertools::Either::Right(i)
                }
            });

        let (in_update_list, state, kind, acc): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = nodes
            .into_iter()
            .zip(csc.iter().map(<[IndexType]>::len))
            .map(|(node, inputs)| {
                (
                    true,
                    node.initial_state,
                    node.kind,
                    Gate::calc_acc_i(inputs, node.kind),
                )
            })
            .multiunzip();

        let outputs = csr.iter().map(|i| i.iter().map(|&i| i as IndexType).collect_vec()).collect_vec();

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
                    let i = *i as usize;
                    let acc = &mut this.acc[i];
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
    const STRATEGY: super::UpdateStrategy = super::UpdateStrategy::Reference;
    fn num_gates_internal(&self) -> usize {
        self.state.len()
    }
}
impl ReferenceLogicSim {
    fn update_inner(&mut self, cluster: bool) {
        let (update_list, next_update_list) = if cluster {
            (&mut self.cluster_update_list, &mut self.update_list)
        } else {
            (&mut self.update_list, &mut self.cluster_update_list)
        };
        for id in update_list.iter().copied() {
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
