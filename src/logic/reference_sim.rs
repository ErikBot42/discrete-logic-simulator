//! Reference implementation for logic simulation.
//! As simple as possible, and therefore slow.

use super::LogicSim;

struct ReferenceLogicSim {
    translation_table: Vec<super::IndexType>,
    state: Vec<bool>,
    in_update_list: Vec<bool>,
    gate_update_list: Vec<super::IndexType>,
    cluster_update_list: Vec<super::IndexType>,
}

impl LogicSim for ReferenceLogicSim {
    fn create(network: super::network::NetworkWithGaps) -> Self {

        todo!()
    }

    fn get_state_internal(&self, gate_id: usize) -> bool {
        self.state[gate_id]
    }

    fn number_of_gates_external(&self) -> usize {
        self.translation_table.len()
    }

    fn update(&mut self) {
        todo!()
    }

    fn to_internal_id(&self, gate_id: usize) -> usize {
        self.translation_table[gate_id] as usize
    }
}
