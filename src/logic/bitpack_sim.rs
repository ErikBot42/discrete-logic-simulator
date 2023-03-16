//use super::*;

use bytemuck::cast_slice_mut;

use super::bitmanip::{
    bit_acc_pack, bit_get, bit_set, extract_acc_info_simd, pack_bits, wrapping_bit_get, BitAcc,
    BitAccPack, BitInt, BITS,
};
use super::{
    Csr, GateType, IndexType, InitializedNetwork, LogicSim, RunTimeGateType, UpdateList,
    UpdateStrategy, Gate,
};

/// size = 8 (u64), align = 4 (u32) -> 8 (u64)
#[derive(Debug, Copy, Clone)]
#[repr(align(8))]
struct Soap {
    base_offset: u32, // 4
    num_outputs: u16, // 2
                      //run_type: RunTimeGateType, // 1
}

pub struct BitPackSimInner {
    translation_table: Vec<IndexType>,
    acc: Vec<BitAccPack>,
    state: Vec<BitInt>,
    parity: Vec<BitInt>,
    csr_outputs: Vec<IndexType>,
    csr_indexes: Vec<IndexType>,
    group_run_type: Vec<RunTimeGateType>,
    update_list: UpdateList,
    cluster_update_list: UpdateList,
    in_update_list: Vec<bool>,
    soap: Vec<Soap>,
}

impl BitPackSimInner {
    #[inline(always)]
    fn calc_group_id(id: usize) -> usize {
        id / BITS
    }
    #[inline(always)]
    fn calc_inner_id(id: usize) -> usize {
        id % BITS
    }
    // pass by reference intentional to use intrinsics.
    #[inline(always)] // function used at single call site
    fn calc_state_pack_parity<const CLUSTER: bool>(
        acc_p: &BitAccPack,
        parity_prev: &mut BitInt,
        state: &BitInt,
        kind: RunTimeGateType,
    ) -> BitInt {
        let (acc_zero, acc_parity) = extract_acc_info_simd(acc_p);
        if CLUSTER {
            acc_zero // don't care about parity since only latch uses it.
        } else {
            match kind {
                RunTimeGateType::OrNand => acc_zero,
                RunTimeGateType::AndNor => !acc_zero,
                RunTimeGateType::XorXnor => acc_parity,
                RunTimeGateType::Latch => {
                    let new_state = state ^ (acc_parity & !*parity_prev);
                    *parity_prev = acc_parity;
                    new_state
                },
            }
        }
    }

    #[inline(always)] // function used at 2 call sites
    fn update_inner<const CLUSTER: bool>(&mut self) {
        self.update_inner_direct::<CLUSTER>();
    }

    #[inline(always)]
    fn update_inner_direct<const CLUSTER: bool>(&mut self) {
        let (update_list, next_update_list) = if CLUSTER {
            (&mut self.cluster_update_list, &mut self.update_list)
        } else {
            (&mut self.update_list, &mut self.cluster_update_list)
        };
        for group_id in update_list.iter_rev().map(|g| g as usize) {
            *unsafe { self.in_update_list.get_unchecked_mut(group_id) } = false;

            let state_mut = unsafe { self.state.get_unchecked_mut(group_id) };
            let parity_mut = unsafe { self.parity.get_unchecked_mut(group_id) };

            let new_state = Self::calc_state_pack_parity::<CLUSTER>(
                unsafe { self.acc.get_unchecked(group_id) },
                parity_mut,
                state_mut,
                *unsafe { self.group_run_type.get_unchecked(group_id) },
            );
            let changed = *state_mut ^ new_state;

            if changed != 0 {
                *state_mut = new_state;
                let soap = *unsafe { self.soap.get_unchecked(group_id) };
                Self::propagate_acc(
                    changed,
                    new_state,
                    &self.csr_outputs,
                    cast_slice_mut(&mut self.acc),
                    &mut self.in_update_list,
                    next_update_list,
                    soap.base_offset,
                    soap.num_outputs,
                );
            }
        }
        update_list.clear();
    }

    /// SET not reset, only for init
    /// If called twice, things will explode
    fn set_state<const CLUSTER: bool>(&mut self, id: usize) {
        let group_id = Self::calc_group_id(id);
        let inner_id = Self::calc_inner_id(id);
        self.state[group_id] = bit_set(self.state[group_id], inner_id, true);

        for id in self.csr_outputs
            [(self.csr_indexes[id] as usize)..(self.csr_indexes[id + 1] as usize)]
            .iter()
            .map(|id| *id as usize)
        {
            let acc: &mut [u8] = cast_slice_mut(&mut self.acc);
            acc[id] = acc[id].wrapping_add(1);

            let update_list = if !CLUSTER {
                &mut self.cluster_update_list
            } else {
                &mut self.update_list
            };
            let id = Self::calc_group_id(id);
            if !self.in_update_list[id] {
                self.in_update_list[id] = true;
                update_list.push_safe(id.try_into().unwrap());
            }
        }
    }

    /// # SAFETY
    /// Assumes invalid gates never activate and that #outputs is constant within a group
    #[inline(always)]
    fn propagate_acc(
        mut changed: BitInt,
        new_state: BitInt,
        csr_outputs: &[IndexType],
        acc: &mut [u8],
        in_update_list: &mut [bool],
        next_update_list: &mut UpdateList,
        base_offset: IndexType,
        num_outputs: u16,
    ) {
        let base_offset = base_offset as u32;
        let num_outputs = num_outputs as u32;

        if num_outputs == 0 {
            return;
        }
        while changed != 0 {
            let i: u32 = changed.trailing_zeros();

            let outputs_start = base_offset + (i * num_outputs);
            let outputs_end = outputs_start + num_outputs;

            changed &= !(1 << i); // ANY

            let delta = if bit_get(new_state, i) {
                1
            } else {
                (0 as BitAcc).wrapping_sub(1)
            };

            for output in
                unsafe { csr_outputs.get_unchecked(outputs_start as usize..outputs_end as usize) }
                    .iter()
                    .map(
                        |&i| i as usize, /* Truncating cast needed for performance */
                    )
            {
                let output_group_id = BitPackSimInner::calc_group_id(output);

                let acc_mut = unsafe { acc.get_unchecked_mut(output) };
                *acc_mut = acc_mut.wrapping_add(delta);

                let in_update_list_mut =
                    unsafe { in_update_list.get_unchecked_mut(output_group_id) };
                if !*in_update_list_mut {
                    unsafe {
                        next_update_list.push_unchecked(output_group_id as IndexType /* Truncating cast is needed for performance */ )
                    };
                    *in_update_list_mut = true;
                }
            }
        }
    }
    fn init_state(&mut self, gates: Vec<Option<Gate>>) {
        for (id, (cluster, state)) in gates
            .iter()
            .map(|g| {
                g.as_ref()
                    .map_or((false, false), |g| (g.kind.is_cluster(), g.initial_state))
            })
            .enumerate()
        {
            if state {
                if cluster {
                    self.set_state::<true>(id);
                } else {
                    self.set_state::<false>(id);
                }
            }
        }
    }
}

impl LogicSim for BitPackSimInner {
    fn create(network: InitializedNetwork) -> Self {

        let network = network.prepare_for_bitpack_packing_no_type_overlap_equal_cardinality(BITS);
        let translation_table = network.translation_table;

        let num_gates = network.gates.len();
        assert_eq!(num_gates % BITS, 0);
        let num_groups = num_gates / BITS;

        let gates = network.gates;

        let csr = Csr::new(
            gates
                .iter()
                .map(|g| g.as_ref().map_or_else(Vec::new, |g| g.outputs.clone())),
        );
        let csr_indexes = csr.indexes;
        let csr_outputs = csr.outputs;
        let state: Vec<_> = gates
            .iter()
            .map(|_| false)
            .array_chunks()
            .map(pack_bits)
            .collect();
        let parity = (0..num_groups).map(|_| 0).collect();

        let kind: Vec<_> = gates
            .iter()
            .map(|g| g.as_ref().map_or(GateType::Cluster, |g| g.kind))
            .collect();

        let group_run_type: Vec<RunTimeGateType> = kind
            .iter()
            .step_by(BITS)
            .map(|k| RunTimeGateType::new(*k))
            .collect();
        let acc: Vec<_> = gates
            .iter()
            .enumerate()
            .map(|(i, g)| {
                g.as_ref().map_or(
                    group_run_type[Self::calc_group_id(i)].acc_to_never_activate(),
                    |g| g.acc() as BitAcc,
                )
            })
            .array_chunks()
            .map(bit_acc_pack)
            .collect();

        let (update_list, cluster_update_list, in_update_list) = make_update_lists(&kind, num_groups, &gates);

        let (group_csr_indexes, group_num_outputs): (Vec<_>, Vec<_>) = (0..num_groups)
            .map(|i| i * BITS)
            .map(|i| {
                (
                    csr_indexes[i],
                    u16::try_from(csr_indexes[i + 1] - csr_indexes[i]).unwrap(),
                )
            })
            .unzip();

        let soap = group_csr_indexes
            .iter()
            .zip(group_num_outputs.iter())
            .map(|(&base_offset, &num_outputs)| Soap {
                base_offset,
                num_outputs,
            })
            .collect();

        let mut this = Self {
            translation_table,
            acc,
            state,
            parity,
            csr_indexes,
            csr_outputs,
            group_run_type,
            update_list,
            cluster_update_list,
            in_update_list,
            soap,
        };

        this.init_state(gates);

        this
    }
    fn get_state_internal(&self, gate_id: usize) -> bool {
        let index = Self::calc_group_id(gate_id);
        wrapping_bit_get(self.state[index], gate_id)
    }
    fn number_of_gates_external(&self) -> usize {
        self.translation_table.len()
    }
    #[inline(always)] // function used at single call site
    fn update(&mut self) {
        self.update_inner::<false>();
        self.update_inner::<true>();
    }
    fn to_internal_id(&self, gate_id: usize) -> usize {
        self.translation_table[gate_id].try_into().unwrap()
    }
    const STRATEGY: UpdateStrategy = UpdateStrategy::BitPack;
}

fn make_update_lists(
    kind: &[GateType],
    num_groups: usize,
    gates: &Vec<Option<Gate>>,
) -> (
    UpdateList,
    UpdateList,
    Vec<bool>,
) {
    let update_list = UpdateList::collect_size(
        kind.iter()
            .step_by(BITS)
            .map(|k| !k.is_cluster())
            .enumerate()
            .filter_map(|(i, b)| b.then_some(i.try_into().unwrap())),
        num_groups,
    );
    let cluster_update_list = UpdateList::collect_size(
        kind.iter()
            .step_by(BITS)
            .map(|k| k.is_cluster())
            .enumerate()
            .filter_map(|(i, b)| b.then_some(i.try_into().unwrap())),
        num_groups,
    );
    let mut in_update_list: Vec<_> = (0..gates.len()).map(|_| false).collect();
    for id in update_list
        .iter()
        .chain(cluster_update_list.iter())
        .map(|id| id as usize)
    {
        in_update_list[id] = true;
    }
    (update_list, cluster_update_list, in_update_list)
}
