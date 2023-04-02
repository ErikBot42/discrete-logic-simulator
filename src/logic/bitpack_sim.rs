//use super::*;

// TODO: optimizations based on ignoring parts of network.

use bytemuck::cast_slice_mut;
use itertools::Itertools;

use crate::logic::network::GateNode;

use super::bitmanip::{
    bit_acc_pack, bit_get, bit_set, extract_acc_info_simd, pack_bits, wrapping_bit_get, BitAcc,
    BitAccPack, BitInt, BITS,
};
use super::{
    Csr, Gate, GateType, IndexType, InitializedNetwork, LogicSim, RunTimeGateType, UpdateList,
    UpdateStrategy,
};

/// size = 8 (u64), align = 4 (u32) -> 8 (u64)
#[derive(Debug, Copy, Clone)]
#[repr(align(8))]
struct Soap {
    base_offset: u32, // 4
    num_outputs: u16, // 2
                      //run_type: RunTimeGateType, // 1
}
#[derive(Clone)]
pub struct BitPackSimInner {
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
    const fn calc_group_id(id: usize) -> usize {
        id / BITS
    }
    #[inline(always)]
    const fn calc_inner_id(id: usize) -> usize {
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

            let update_list = if CLUSTER {
                &mut self.update_list
            } else {
                &mut self.cluster_update_list
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
                let output_group_id = Self::calc_group_id(output);

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
    fn init_state(&mut self, gates: &[Option<(usize, GateNode)>]) {
        for (id, (cluster, state)) in gates
            .iter()
            .map(|g| {
                g.as_ref().map_or((false, false), |g| {
                    (g.1.kind.is_cluster(), g.1.initial_state)
                })
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

impl crate::logic::RenderSim for BitPackSimInner {
    // WAY faster state copy
    fn get_state_in(&mut self, v: &mut Vec<u64>) {
        v.clear();
        v.extend(self.state.iter().cloned());
        v.shrink_to_fit();
    }
}
//    // Simulate 1 tick.
//    //fn rupdate(&mut self);
//    // Clear and write state bitvec
//    //fn get_state_in(&mut self, v: &mut Vec<u64>);

#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Copy, Clone)]
enum GateOrderingKey {
    Interface = 0,
    OrNand,
    AndNor,
    XorXnor,
    Latch,
    Cluster,
}


// TODO: distinct cardinality makes aligned interface impossible.
fn bit_pack_nodes(nodes: &Vec<GateNode>, csr: &Csr<u32>) -> (Vec<Option<usize>>, Vec<usize>) {
    // bit packed -> prev id
    let mut table: Vec<Option<usize>> = Vec::new();
    let mut group_kinds: Vec<(usize, GateOrderingKey)> = Vec::new();
    for (i, key) in nodes
        .iter()
        .map(|n| {
            (match n.kind {
                GateType::And | GateType::Nor => GateOrderingKey::AndNor,
                GateType::Or | GateType::Nand => GateOrderingKey::OrNand,
                GateType::Xor | GateType::Xnor => GateOrderingKey::XorXnor,
                GateType::Latch => GateOrderingKey::Latch,
                GateType::Interface(s) => GateOrderingKey::Interface,
                GateType::Cluster => GateOrderingKey::Cluster,
            })
        })
        .enumerate()
        .sorted_by_key(|(i, key)| (*key, csr[*i].len()))
    {
        if table.len() % BITS == 0 {
            group_kinds.push((csr[i].len(), key));
        }
        if let Some(&k) = group_kinds.last() && (k.1 != key || k.0 != csr[i].len()) {
            while table.len() % BITS != 0 {
                table.push(None);
            }
        }
        table.push(Some(i));
    }
    while table.len() % BITS != 0 {
        table.push(None);
    }
    // prev id -> bit packed
    let mut inv_table = (0..nodes.len()).map(|_| None).collect_vec();
    for (i, &entry) in table.iter().enumerate() {
        if let Some(entry) = entry {
            inv_table[entry] = Some(i);
        }
    }
    let inv_table = inv_table.iter().map(|&i| i.unwrap()).collect_vec();
    (table, inv_table)
}

impl LogicSim for BitPackSimInner {
    fn create(
        outputs_iter: impl IntoIterator<Item = impl IntoIterator<Item = usize>>,
        nodes: Vec<GateNode>,
        mut translation_table: Vec<u32>,
    ) -> (Vec<IndexType>, Self) {
        let csr = Csr::new(
            outputs_iter
                .into_iter()
                .map(|i| i.into_iter().map(|i| i as u32)),
        );
        let (bit_pack_table, bit_pack_inv_table) = bit_pack_nodes(&nodes, &csr);
        translation_table.iter_mut().for_each(|t| {
            *t = u32::try_from(bit_pack_inv_table[usize::try_from(*t).unwrap()]).unwrap()
        });

        let csc_pre_table = csr.as_csc();

        let csr: Csr<u32> = Csr::from_adjacency(
            csr.adjacency_iter()
                .map(|(a, b)| {
                    (
                        bit_pack_inv_table[a as usize] as u32,
                        bit_pack_inv_table[b as usize] as u32,
                    )
                })
                .collect(),
            bit_pack_table.len(),
        );
        let node_data = bit_pack_table
            .iter()
            .map(|i| i.map(|i| (csc_pre_table[i].len(), nodes[i].clone())))
            .collect_vec();

        let num_gates = node_data.len();
        let num_groups = num_gates / BITS;
        assert_eq!(num_gates % BITS, 0);
        assert_eq_len!(node_data, csr);

        let csr_indexes = csr.indexes;
        let csr_outputs = csr.outputs;
        let state: Vec<_> = (0..num_gates)
            .map(|_| false)
            .array_chunks()
            .map(pack_bits)
            .collect();
        let parity = (0..num_groups).map(|_| 0).collect();

        let kind: Vec<_> = node_data
            .iter()
            .map(|n| n.as_ref().map_or(GateType::Cluster, |n| n.1.kind))
            .collect();

        let group_run_type: Vec<RunTimeGateType> = kind
            .iter()
            .step_by(BITS)
            .map(|k| RunTimeGateType::new(*k))
            .collect();
        let acc: Vec<_> = node_data
            .iter()
            .enumerate()
            .map(|(i, n)| {
                n.as_ref().map_or(
                    group_run_type[Self::calc_group_id(i)].acc_to_never_activate(),
                    |n| Gate::calc_acc_i(n.0, n.1.kind) as BitAcc,
                )
            })
            .array_chunks()
            .map(bit_acc_pack)
            .collect();

        let (update_list, cluster_update_list, in_update_list) =
            make_update_lists(&kind, num_groups, num_gates);

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

        this.init_state(&node_data);

        (translation_table, this)
    }
    fn get_state_internal(&self, gate_id: usize) -> bool {
        let index = Self::calc_group_id(gate_id);
        wrapping_bit_get(self.state[index], gate_id)
    }
    #[inline(always)] // function used at single call site
    fn update(&mut self) {
        self.update_inner::<false>();
        self.update_inner::<true>();
    }
    const STRATEGY: UpdateStrategy = UpdateStrategy::BitPack;

    fn num_gates_internal(&self) -> usize {
        self.state.len() * BitInt::BITS as usize
    }
}

fn make_update_lists(
    kind: &[GateType],
    num_groups: usize,
    num_gates: usize,
) -> (UpdateList, UpdateList, Vec<bool>) {
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
    let mut in_update_list: Vec<_> = (0..num_gates).map(|_| false).collect();
    for id in update_list
        .iter()
        .chain(cluster_update_list.iter())
        .map(|id| id as usize)
    {
        in_update_list[id] = true;
    }
    (update_list, cluster_update_list, in_update_list)
}
