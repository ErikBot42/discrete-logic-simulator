//! Reference implementation for logic simulation.
//! As simple as possible, and therefore slow.

use super::{Gate, LogicSim, RunTimeGateType};

struct ReferenceLogicSim {
    gate_update_list: Vec<super::IndexType>,
    cluster_update_list: Vec<super::IndexType>,
    in_update_list: Vec<bool>,
    translation_table: Vec<super::IndexType>,
    state: Vec<bool>,
    kind: Vec<super::GateType>,
    acc: Vec<super::AccType>,
    gates: Vec<Option<Gate>>,
}

impl LogicSim for ReferenceLogicSim {
    fn create(network: super::network::NetworkWithGaps) -> Self {
        let translation_table = network.translation_table;
        let kind: Vec<_> = network
            .gates
            .iter()
            .map(|x| x.as_ref().map(|x| x.kind).unwrap_or_default())
            .collect();
        let in_update_list: Vec<_> = kind.iter().map(|x| !x.is_cluster()).collect();
        let gate_update_list: Vec<_> = in_update_list
            .iter()
            .enumerate()
            .filter_map(|(i, u)| u.then_some(i as super::IndexType))
            .collect();
        let cluster_update_list = Vec::new();
        let state = (0..network.gates.len()).map(|_| false).collect();
        let acc: Vec<_> = network
            .gates
            .iter()
            .map(|x| x.as_ref().map(|x| x.acc).unwrap_or_default())
            .collect();
        Self {
            translation_table,
            state,
            kind,
            acc,
            in_update_list,
            gate_update_list,
            cluster_update_list,
            gates: network.gates,
        }
    }

    fn get_state_internal(&self, gate_id: usize) -> bool {
        self.state[gate_id]
    }

    fn number_of_gates_external(&self) -> usize {
        self.translation_table.len()
    }

    fn update(&mut self) {
        self.update_inner(false);
        self.gate_update_list.clear();
        self.update_inner(true);
        self.cluster_update_list.clear();
    }

    fn to_internal_id(&self, gate_id: usize) -> usize {
        self.translation_table[gate_id] as usize
    }
}
impl ReferenceLogicSim {
    fn update_inner(&mut self, cluster: bool) {
        let (old_update_list, new_update_list) = if cluster {
            (&self.cluster_update_list, &mut self.gate_update_list)
        } else {
            (&self.cluster_update_list, &mut self.gate_update_list)
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
}
