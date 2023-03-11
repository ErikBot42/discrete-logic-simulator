//use super::*;


use bytemuck::cast_slice_mut;

use super::{
    pack_sparse_matrix, repack_single_sparse_matrix, AccType, Csr, GateType, IndexType,
    InitializedNetwork, LogicSim, RunTimeGateType, UpdateList, UpdateStrategy,
};

use super::bitmanip::*;


/// size = 8 (u64), align = 4 (u32) -> 8 (u64)
#[derive(Debug, Copy, Clone)]
#[repr(align(8))]
struct Soap {
    base_offset: u32, // 4
    num_outputs: u16, // 2
                      //run_type: RunTimeGateType, // 1
}



//#[derive(Debug)]
pub struct BitPackSimInner /*<const LATCH: bool>*/ {
    translation_table: Vec<IndexType>,
    acc: Vec<BitAccPack>,
    state: Vec<BitInt>,
    parity: Vec<BitInt>,
    csr_single: Vec<IndexType>,
    group_run_type: Vec<RunTimeGateType>,
    update_list: UpdateList,
    cluster_update_list: UpdateList,
    in_update_list: Vec<bool>,
    soap: Vec<Soap>,
}

impl BitPackSimInner {
    //const BITS: usize = BitInt::BITS as usize;
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
        //use std::sync::mpsc::channel;
        //let (send, reciv) = channel();

        //send.send(1).unwrap();

        //reciv.recv().unwrap();
        // send mpsc
        //
        //self.barrier.wait();
        // reciv mpsc
        // std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        // self.barrier2.wait();
        //self.barrier.wait(); // <- same barrier can be used again :)
        // std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        // send mpsc

        //let mut buffer = self.buffers.0.lock();

        //buffer.clear();

        self.update_inner_direct::<CLUSTER>();
        //swap(&mut self.local_buffer, &mut buffer);
        //drop(buffer);
        //swap(&mut self.buffers.0, &mut self.buffers.1);
    }

    #[inline(always)]
    fn update_inner_direct<const CLUSTER: bool>(&mut self) {
        let (update_list, next_update_list) = if CLUSTER {
            (&mut self.cluster_update_list, &mut self.update_list)
        } else {
            (&mut self.update_list, &mut self.cluster_update_list)
        };
        for group_id in update_list.iter_rev().map(|g| g as usize) {
            //let offset = group_id * Self::BITS;

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
                    //offset,
                    new_state,
                    &self.csr_single,
                    cast_slice_mut(&mut self.acc),
                    &mut self.in_update_list,
                    next_update_list,
                    soap.base_offset,
                    soap.num_outputs,
                    //*unsafe { self.group_csr_base_outputs.get_unchecked(group_id) },
                    //*unsafe { self.group_num_outputs.get_unchecked(group_id) },
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

        for id in self.csr_single
            [(self.csr_single[id] as usize)..(self.csr_single[id + 1] as usize)]
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
        //offset: usize,
        new_state: BitInt,
        single_packed_outputs: &[IndexType],
        acc: &mut [u8],
        in_update_list: &mut [bool],
        next_update_list: &mut UpdateList,
        base_offset: IndexType,
        num_outputs: u16,
        //is_invalid: &[bool],
    ) {
        let base_offset = base_offset as u32;
        let num_outputs = num_outputs as u32;

        //let base_offset = unsafe { *single_packed_outputs.get_unchecked(offset) as usize };
        //let num_outputs =
        //    *unsafe { single_packed_outputs.get_unchecked(offset + 1) } as usize - base_offset;
        if num_outputs == 0 {
            return;
        }
        while changed != 0 {
            let i_u32 = changed.trailing_zeros();
            let i_usize = i_u32 as usize;

            /*let gate_id = offset + i_usize;
            let (outputs_start, outputs_end) = (
                *unsafe { single_packed_outputs.get_unchecked(gate_id) } as usize,
                *unsafe { single_packed_outputs.get_unchecked(gate_id + 1) } as usize,
            );*/

            let outputs_start = base_offset + (i_u32 * num_outputs);
            let outputs_end = outputs_start + num_outputs;

            //dbg!(outputs_start, outputs_end);
            //assert_eq!(
            //    (outputs_start, outputs_end),
            //    (outputs_start2, outputs_end2),
            //    "num_outputs: {num_outputs}, i_usize: {i_usize}, base_offset: {base_offset}, foo: {:?}", &single_packed_outputs[offset..(offset+Self::BITS)]
            //);

            changed &= !(1 << i_u32); // ANY

            //let delta = (AccType::from(bit_get(new_state, i_usize)) * 2).wrapping_sub(1);
            let delta = if bit_get(new_state, i_usize) {
                1
            } else {
                (0 as AccType).wrapping_sub(1)
            };

            for output in unsafe {
                single_packed_outputs.get_unchecked(outputs_start as usize..outputs_end as usize)
            }
            .iter()
            .map(
                |&i| i as usize, /* Truncating cast needed for performance */
            ) {
                let output_group_id = BitPackSimInner::calc_group_id(output);

                let acc_mut = unsafe { acc.get_unchecked_mut(output) };
                *acc_mut = acc_mut.wrapping_add(delta);
                //unsafe {
                //    std::intrinsics::atomic_xadd_relaxed(acc_mut, delta);
                //}

                let in_update_list_mut =
                    unsafe { in_update_list.get_unchecked_mut(output_group_id) };
                if !*in_update_list_mut
                //if unsafe {
                //    std::intrinsics::atomic_xchg_relaxed(transmute(in_update_list_mut), 1_u8)
                //} == 0
                {
                    unsafe {
                        next_update_list.push_unchecked(output_group_id as IndexType /* Truncating cast is needed for performance */ )
                    };
                    *in_update_list_mut = true;
                }
            }
        }
    }
}

impl LogicSim for BitPackSimInner /*<LATCH>*/ {
    fn create(network: InitializedNetwork) -> Self {
        //let network = network.prepare_for_bitpack_packing(Self::BITS);
        let network = network.prepare_for_bitpack_packing_no_type_overlap_equal_cardinality(BITS);

        let number_of_gates_with_padding = network.gates.len();
        assert_eq!(number_of_gates_with_padding % BITS, 0);
        let number_of_buckets = number_of_gates_with_padding / BITS;
        let gates = network.gates;

        let translation_table = network.translation_table;
        let (csr_indexes, csr_outputs) = pack_sparse_matrix(
            gates
                .iter()
                .map(|g| g.as_ref().map_or_else(Vec::new, |g| g.outputs.clone())),
        );
        let csr_single = repack_single_sparse_matrix(&csr_indexes, &csr_outputs);

        let state: Vec<_> = gates
            .iter()
            //.map(|g| g.as_ref().map_or(false, |g| g.initial_state))
            .map(|_| false)
            .array_chunks()
            .map(pack_bits)
            .collect();
        assert_eq!(state.len(), number_of_buckets);
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
        let update_list = UpdateList::collect_size(
            kind.iter()
                .step_by(BITS)
                .map(|k| !k.is_cluster())
                .enumerate()
                .filter_map(|(i, b)| b.then_some(i.try_into().unwrap())),
            number_of_buckets,
        );
        let cluster_update_list = UpdateList::collect_size(
            kind.iter()
                .step_by(BITS)
                .map(|k| k.is_cluster())
                .enumerate()
                .filter_map(|(i, b)| b.then_some(i.try_into().unwrap())),
            number_of_buckets,
        );

        let mut in_update_list: Vec<_> = (0..gates.len()).map(|_| false).collect();
        for id in update_list
            .iter()
            .chain(cluster_update_list.iter())
            .map(|id| id as usize)
        {
            in_update_list[id] = true;
        }

        let parity = (0..kind.len()).map(|_| 0).collect();

        let (group_csr_base_outputs, group_num_outputs): (Vec<_>, Vec<_>) = (0..number_of_buckets)
            .map(|i| i * BITS)
            .map(|i| {
                (
                    csr_single[i],
                    u16::try_from(csr_single[i + 1] - csr_single[i]).unwrap(),
                )
            })
            .unzip();

        let soap = group_csr_base_outputs
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
            csr_single,
            group_run_type,
            update_list,
            cluster_update_list,
            in_update_list,
            soap,
        };

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
                    this.set_state::<true>(id);
                } else {
                    this.set_state::<false>(id);
                }
            }
        }

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

