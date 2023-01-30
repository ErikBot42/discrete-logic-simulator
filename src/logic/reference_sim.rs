//! Reference implementation for logic simulation.
//! As simple as possible, and therefore slow.

use super::{Gate, IndexType, LogicSim, RunTimeGateType};
use itertools::Itertools;

pub(crate) struct ReferenceLogicSim {
    gate_update_list: Vec<usize>,
    cluster_update_list: Vec<usize>,

    in_update_list: Vec<bool>,
    state: Vec<bool>,
    kind: Vec<super::GateType>,
    acc: Vec<super::AccType>,
    outputs: Vec<Vec<IndexType>>,
    translation_table: Vec<IndexType>,
}

impl LogicSim for ReferenceLogicSim {
    fn create(network: super::network::InitializedNetwork) -> Self {
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
        let (in_update_list, state, kind, acc, outputs) = gates
            .into_iter()
            .map(|g| (true, false, g.kind, g.calc_acc(), g.outputs))
            .multiunzip();
        Self {
            gate_update_list,
            cluster_update_list,
            in_update_list,
            state,
            kind,
            acc,
            outputs,
            translation_table,
        }
    }

    fn get_state_internal(&self, gate_id: usize) -> bool {
        self.state[gate_id]
    }

    fn number_of_gates_external(&self) -> usize {
        self.translation_table.len()
    }

    fn update(&mut self) {
        todo!();
        //self.update_inner(false);
        //self.gate_update_list.clear();
        //self.update_inner(true);
        //self.cluster_update_list.clear();
    }

    fn to_internal_id(&self, gate_id: usize) -> usize {
        self.translation_table[gate_id] as usize
    }
    const STRATEGY: super::UpdateStrategy = super::UpdateStrategy::Reference;
}
/*impl ReferenceLogicSim {
    fn update_inner(&mut self, cluster: bool) {
        let (old_update_list, new_update_list) = if cluster {
            (&self.cluster_update_list, &mut self.gate_update_list)
        } else {
            (&self.gate_update_list, &mut self.cluster_update_list)
        };
        assert_eq!(new_update_list.len(), 0);
        for id in old_update_list.iter().map(|x| *x as usize) {
            let acc = self.acc[id];
            let kind = self.kind[id];
            let new_state = Gate::evaluate(acc, RunTimeGateType::new(kind));
            let state_changed = new_state != self.state[id];
            if state_changed {
                let delta: u8 = if new_state {
                    1
                } else {
                    (0 as super::AccType).wrapping_sub(1)
                };
                let outputs = self.gates[id].as_ref().unwrap().outputs.clone();
                for output in outputs.into_iter().map(|x| x as usize) {
                    self.acc[output] += delta;
                    if !self.in_update_list[output] {
                        new_update_list.push(output as super::IndexType);
                        self.in_update_list[output] = true;
                    }
                }
            }

            self.in_update_list[id] = false;
        }
    }
}*/
