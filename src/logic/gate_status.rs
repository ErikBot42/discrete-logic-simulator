//use super::*;

use super::{RunTimeGateType, AccType};
use std::simd::{LaneCount, Mask, Simd, SimdPartialEq, SupportedLaneCount};
//TODO: this only uses 4 bits, 2 adjacent gates could share their
//      in_update_list flag and be updated at the same time.
pub(crate) type Inner = u8;
pub(crate) type InnerSigned = i8;
pub(crate) type Packed = u64;
pub(crate) const PACKED_ELEMENTS: usize = std::mem::size_of::<Packed>();
// bit locations
const STATE: Inner = 0;
const IN_UPDATE_LIST: Inner = 1;
const IS_INVERTED: Inner = 2;
const IS_XOR: Inner = 3;

const FLAG_STATE: Inner = 1 << STATE;
const FLAG_IN_UPDATE_LIST: Inner = 1 << IN_UPDATE_LIST;
const FLAG_IS_INVERTED: Inner = 1 << IS_INVERTED;
const FLAG_IS_XOR: Inner = 1 << IS_XOR;
const FLAG_IS_LATCH: Inner = FLAG_IS_XOR | FLAG_IS_INVERTED;
const FLAGS_MASK: Inner = FLAG_IS_INVERTED | FLAG_IS_XOR;
//const ANY_MASK: Inner = FLAG_STATE | FLAG_IN_UPDATE_LIST | FLAG_IS_INVERTED | FLAG_IS_XOR;

pub(crate) fn new(in_update_list: bool, state: bool, kind: RunTimeGateType) -> Inner {
    //let in_update_list = in_update_list as u8;
    //let state = state as u8;
    let (is_inverted, is_xor) = RunTimeGateType::calc_flags(kind);

    (Inner::from(state) << STATE)
        | (Inner::from(in_update_list) << IN_UPDATE_LIST)
        | (Inner::from(is_inverted) << IS_INVERTED)
        | (Inner::from(is_xor) << IS_XOR)
}

/// Evaluate and update internal state.
/// # Returns
/// Delta (+-1) if state changed (0 = no change)
#[inline]
pub(crate) fn eval_mut<const CLUSTER: bool>(
    inner_mut: &mut Inner,
    acc: AccType,
    acc_prev: AccType,
) -> AccType {
    // <- high, low ->
    //(     3,           2,              1,     0)
    //(is_xor, is_inverted, in_update_list, state)
    // variables are valid for their *first* bit
    let inner = *inner_mut;
    //debug_assert!(in_update_list(inner));

    //let inner = self.inner; // 0000XX1X

    let flag_bits = inner & FLAGS_MASK;

    let state_1 = (inner >> STATE) & 1;
    let acc = Inner::from(acc); // XXXXXXXX
    let new_state_1 = if CLUSTER {
        AccType::from(acc != 0)
    } else {
        match flag_bits {
            0 => Inner::from(acc != 0),
            FLAG_IS_INVERTED => Inner::from(acc == 0),
            FLAG_IS_XOR => acc & 1,
            FLAG_IS_LATCH => Inner::from((state_1 == 1) != ((acc & 1 == 1) && (acc_prev & 1 == 0))),
            _ => unreachable!(),
            //_ => 0,
            /*_ => unreachable!(),*/ /*unsafe {
                debug_assert!(false);
                std::hint::unreachable_unchecked()
            },*/
        }
    };
    /*{
            let is_xor = inner >> IS_XOR; // 0|1
            debug_assert_eq!(is_xor & 1, is_xor);
            let acc_parity = acc; // XXXXXXXX
            let xor_term = is_xor & acc_parity; // 0|1
            debug_assert_eq!(xor_term & 1, xor_term);
            let acc_not_zero = Inner::from(acc != 0); // 0|1
            let is_inverted = inner >> IS_INVERTED; // XX
            let not_xor = !is_xor; // 0|11111111
            let acc_term = not_xor & (is_inverted ^ acc_not_zero); // XXXXXXXX
            let new_state_1_other = xor_term | (acc_term & 1);
            debug_assert_eq!(
                new_state_1, new_state_1_other,
                "
    CLUSTER: {CLUSTER}
    FLAG_IS_INVERTED: {FLAG_IS_INVERTED:b}
    FLAG_IS_XOR: {FLAG_IS_XOR:b}
    inner: {inner:b}
    is_xor: {is_xor:b}
    acc_parity: {acc_parity:b}
    xor_term: {xor_term:b}
    acc_not_zero: {acc_not_zero:b}
    is_inverted: {is_inverted:b}
    not_xor: {not_xor:b}
    acc_term: {acc_term:b}",
            );
        }*/

    let state_changed_1 = new_state_1 ^ state_1;

    // automatically sets "in_update_list" bit to zero
    *inner_mut = (new_state_1 << STATE) | flag_bits;

    debug_assert!(!in_update_list(*inner_mut));
    //debug_assert_eq!(expected_new_state, new_state_1 != 0);
    //super::debug_assert_assume(true);
    debug_assert!(state_changed_1 == 0 || state_changed_1 == 1);
    debug_assert!(state_changed_1 < 2);
    //unsafe {
    //    std::intrinsics::assume(state_changed_1 == 0 || state_changed_1 == 1);
    //}

    if state_changed_1 == 0 {
        0
    } else {
        AccType::from((new_state_1 << 1).wrapping_sub(1))
    }
}

pub(crate) const fn splat_u32(value: u8) -> Packed {
    pack_single([value; PACKED_ELEMENTS])
}
/// if byte contains any bit set, it will be
/// replaced with 0xff
pub(crate) const fn or_combine(value: Packed) -> Packed {
    //TODO: try using a tree here.
    //      this is extremely sequential
    let mut value = value;
    value |= (splat_u32(0b1111_0000) & value) >> 4;
    value |= (splat_u32(0b1100_1100) & value) >> 2;
    value |= (splat_u32(0b1010_1010) & value) >> 1;
    value |= (splat_u32(0b0000_1111) & value) << 4;
    value |= (splat_u32(0b0011_0011) & value) << 2;
    value |= (splat_u32(0b0101_0101) & value) << 1;
    value
}
/// like `or_combine`, but replaces with 0x1 instead.
/// equivalent to `BYTEwise` != 0
const fn or_combine_1(value: Packed) -> Packed {
    //let mut value = value;
    //value |= (splat_u32(0b1111_0000) & value) >> 4;
    //value |= (splat_u32(0b1100_1100) & value) >> 2;
    //value |= (splat_u32(0b1010_1010) & value) >> 1;
    //value & splat_u32(1)
    let value = value | ((value & splat_u32(0b1111_0000)) >> 4);
    let value = value | ((value & splat_u32(0b0000_1100)) >> 2);
    let value = value | ((value & splat_u32(0b0000_0010)) >> 1);
    value & splat_u32(1)
}
/// for each byte:
/// 0 -> `0b0000_0000`
/// 1 -> `0b1111_1111`
const fn mask_if_one(value: Packed) -> Packed {
    //let value = value & splat_u32(1);
    //let value = value | (value << 4);
    //let value = value | (value << 2);
    //value | (value << 1)
    (value & splat_u32(1)) * 0b1111_1111
}
// TODO: save on the 4 unused bits, maybe merge 2 iterations?
// TODO: relaxed or_combine
pub(crate) fn eval_mut_scalar<const CLUSTER: bool>(
    inner_mut: &mut Packed,
    acc: Packed,
    acc_prev: Packed,
) -> Packed {
    let inner = *inner_mut;
    let state_1 = (inner >> STATE) & splat_u32(1);

    let is_xor_1 = inner >> IS_XOR & splat_u32(1);
    let is_inverted_1 = inner >> IS_INVERTED & splat_u32(1);

    let xor_term = is_xor_1 & acc & !is_inverted_1;
    let acc_not_zero = or_combine_1(acc);
    let not_xor = !is_xor_1;
    let acc_term = not_xor & (is_inverted_1 ^ acc_not_zero);

    let latch_term = is_xor_1 & is_inverted_1 & (state_1 ^ (acc & !acc_prev));

    let new_state_1 = (xor_term | acc_term | latch_term) & splat_u32(1);
    let state_changed_1 = (new_state_1 ^ state_1) & splat_u32(1);

    // automatically sets "in_update_list" bit to zero
    let new_inner_mut = (new_state_1 << STATE) | (inner & splat_u32(FLAGS_MASK));
    *inner_mut = new_inner_mut;
    let increment_1 = new_state_1 & state_changed_1;
    let decrement_1 = !new_state_1 & state_changed_1;
    debug_assert_eq!(increment_1 & decrement_1, 0, "{increment_1}, {decrement_1}");
    mask_if_one(decrement_1) | increment_1
}

#[inline]
pub(crate) fn eval_mut_simd<const CLUSTER: bool, const LANES: usize>(
    inner_mut: &mut Simd<Inner, LANES>,
    acc: Simd<AccType, LANES>,
    acc_prev: Simd<AccType, LANES>,
) -> Simd<AccType, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    use super::SimdLogicType;
    let inner = *inner_mut;
    let state_1 = (inner >> Simd::splat(STATE)) & Simd::splat(1);

    let flag_bits = inner & Simd::splat(FLAGS_MASK);

    let acc = acc.cast(); // XXXXXXXX
    let new_state_1: Simd<SimdLogicType, LANES> = if CLUSTER {
        //(acc != 0) as u8
        acc.simd_ne(Simd::splat(0 as SimdLogicType))
            .select(Simd::splat(1), Simd::splat(0))
    } else {
        let is_inverted = inner >> Simd::splat(IS_INVERTED) & Simd::splat(1);
        let is_xor = (inner >> Simd::splat(IS_XOR)) & Simd::splat(1);
        let acc_parity = acc & Simd::splat(1);
        let acc_parity_prev = acc_prev & Simd::splat(1);
        let xor_term = is_xor & acc_parity & !(is_inverted);
        let latch_term = is_xor & is_inverted & (state_1 ^ (acc_parity & !acc_parity_prev));

        let acc_not_zero = acc
            .simd_ne(Simd::splat(0))
            .select(Simd::splat(1), Simd::splat(0)); // 0|1
        let not_xor = !is_xor; // 0|11111111
        let acc_term = not_xor & (is_inverted ^ acc_not_zero); // XXXXXXXX
        (xor_term | acc_term | latch_term) & Simd::splat(1)
    };

    let state_changed_1 = new_state_1 ^ state_1;

    // TODO: optimize
    let state_changed_1_signed: Simd<InnerSigned, LANES> = state_changed_1.cast();

    let state_changed_mask =
        unsafe { Mask::from_int_unchecked(state_changed_1_signed - Simd::splat(1)) };

    // automatically sets "in_update_list" bit to zero
    *inner_mut = (new_state_1 << Simd::splat(STATE)) | flag_bits;

    state_changed_mask.select(
        Simd::splat(0),
        (new_state_1 << Simd::splat(1)) - Simd::splat(1),
    )
}

//#[inline]
////pub(crate) fn mark_in_update_list(inner: &mut Inner) {
//    *inner |= FLAG_IN_UPDATE_LIST;
//}
#[inline]
pub(crate) fn in_update_list(inner: Inner) -> bool {
    inner & FLAG_IN_UPDATE_LIST != 0
}
#[inline]
pub(crate) fn state(inner: Inner) -> bool {
    inner & FLAG_STATE != 0
}

pub(crate) fn pack(mut iter: impl Iterator<Item = u8>) -> Vec<Packed> {
    let mut tmp = Vec::new();
    loop {
        tmp.push(pack_single([
            unwrap_or_else!(iter.next(), break),
            iter.next().unwrap_or(0),
            iter.next().unwrap_or(0),
            iter.next().unwrap_or(0),
            iter.next().unwrap_or(0),
            iter.next().unwrap_or(0),
            iter.next().unwrap_or(0),
            iter.next().unwrap_or(0),
        ]));
    }
    tmp
}
#[inline(always)]
pub(crate) const fn pack_single(unpacked: [u8; PACKED_ELEMENTS]) -> Packed {
    Packed::from_le_bytes(unpacked)
}
#[inline(always)]
pub(crate) const fn unpack_single(packed: Packed) -> [u8; PACKED_ELEMENTS] {
    Packed::to_le_bytes(packed)
}
////pub(crate) fn packed_state(packed: Packed) -> [bool; PACKED_ELEMENTS] {
//    let mut res = [false; PACKED_ELEMENTS];
//    res.iter_mut()
//        .zip(unpack_single(packed & splat_u32(FLAG_STATE)))
//        .for_each(|(res, x)| *res = x != 0);
//    res
//}
pub(crate) fn packed_state_vec(packed: Packed) -> Vec<bool> {
    unpack_single(packed & splat_u32(FLAG_STATE))
        .into_iter()
        .map(|x| x != 0)
        .collect()
}
////pub(crate) fn get_state_from_packed_slice(packed: &[Packed], index: usize) -> bool {
//    let outer_index = index / PACKED_ELEMENTS;
//    let inner_index = index % PACKED_ELEMENTS;
//
//    packed_state(packed[outer_index])[inner_index]
//}

mod tests {
    use super::*;
    #[test]
    fn pack_unpack_single() {
        test_pack_single([1, 2, 3, 4, 5, 6, 7, 8]);
        test_pack_single([255, 2, 254, 4, 23, 34, 4, 3]);
        test_pack_single([25, 122, 254, 124, 6, 2, 6, 1]);
    }
    fn test_pack_single(t: [u8; PACKED_ELEMENTS]) {
        assert_eq!(t, unpack_single(pack_single(t)));
    }
    #[test]
    fn pack_multiple() {
        test_pack(&[1, 2, 3, 4, 5, 6, 7, 8]);
        test_pack(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        test_pack(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        test_pack(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    }
    fn test_pack(buffer: &[u8]) {
        let packed = pack(buffer.iter().cloned());
        let buffer_iter = buffer.iter().cloned();
        packed
            .into_iter()
            .map(|x| unpack_single(x))
            .flatten()
            .zip(buffer_iter)
            .for_each(|(a, b)| assert_eq!(a, b));
    }
    #[test]
    fn test_or_combine() {
        for value2 in 0..32 {
            for value in 0..32 {
                let value: Packed = (1 << value) | (1 << value2);
                let mut bytes = value.to_le_bytes();
                for byte in bytes.iter_mut() {
                    if *byte != 0 {
                        *byte = 255;
                    }
                }
                let expected = Packed::from_le_bytes(bytes);
                assert_eq!(
                    expected,
                    or_combine(value),
                    "invalid or_combine() for: {}",
                    value
                );

                let expected_1 = expected & splat_u32(1);
                assert_eq!(
                    expected_1,
                    or_combine_1(value),
                    "invalid or_combine_1() for: {}",
                    value
                );
            }
        }
    }
}
