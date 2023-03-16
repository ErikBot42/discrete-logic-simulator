//! bit manipulation and simd 

#[inline(always)]
fn inline_arr_from_fn<T, const N: usize, F>(mut cb: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    let mut idx = 0;
    [(); N].map(|_| {
        let res = cb(idx);
        idx += 1;
        res
    })
}

/// Number of gates in one group; Number of bits in BitInt
pub(crate) const BITS: usize = BitInt::BITS as usize;

#[repr(C, align(64))] // in bits: 64*8 = 512 bits
#[derive(Debug, Copy, Clone)]
pub(crate) struct BitAccPack([BitAcc; BITS]);
unsafe impl bytemuck::Zeroable for BitAccPack {}
unsafe impl bytemuck::Pod for BitAccPack {}
const _: () = {
    let size = size_of::<BitAccPack>();
    let align = align_of::<BitAccPack>();
    assert!(size == align, "BitAccPack: size diffrent from alignment");
};
pub(crate) fn bit_acc_pack(arr: [BitAcc; BITS]) -> BitAccPack {
    BitAccPack(arr)
}

/// Acc type for bitpack
pub(crate) type BitAcc = u8;
/// Bit array type for bitpack
pub(crate) type BitInt = u64;

use std::mem::{align_of, size_of, transmute};
use std::ops::Range;


use core::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi8, _mm256_load_si256, _mm256_movemask_epi8, _mm256_setzero_si256,
    _mm256_slli_epi64,
};

/// # SAFETY
/// Pointer MUST be aligned
#[inline(always)] // function used at single call site
unsafe fn acc_parity_m256i(acc_ptr: *const __m256i) -> u32 {
    unsafe {
        let data = _mm256_load_si256(acc_ptr); // load value
        let data = _mm256_slli_epi64::<7>(data); // shift LSB to MSB for each byte
        let data = _mm256_movemask_epi8(data); // put MSB of each byte in an int
        transmute(data)
    }
}
#[inline(always)] // function used at single call site
fn acc_parity_simd(acc: &BitAccPack) -> BitInt {
    unsafe {
        assert!(align_of::<BitAccPack>() >= 32);
        let acc_ptr: *const __m256i = (acc as *const BitAccPack).cast();
        let array: [u32; size_of::<BitAccPack>() / u32::BITS as usize] =
            inline_arr_from_fn(|x| acc_parity_m256i(acc_ptr.add(x)));
        transmute(array) // compiler can statically check size here
    }
}

/// # SAFETY
/// Pointer MUST be aligned
#[inline(always)] // function used at single call site
unsafe fn acc_zero_m256i(acc_ptr: *const __m256i) -> u32 {
    unsafe {
        let zero = _mm256_setzero_si256();
        let data = _mm256_load_si256(acc_ptr); // load value
        let data = _mm256_cmpeq_epi8(data, zero); // compare with zero
        let data = _mm256_movemask_epi8(data); // put MSB of each byte in an int
        transmute(!data)
    }
}

#[inline(always)] // function used at single call site
fn acc_zero_simd(acc: &BitAccPack) -> BitInt {
    unsafe {
        assert!(align_of::<BitAccPack>() >= 32);
        let acc_ptr: *const __m256i = (acc as *const BitAccPack).cast();
        let array: [u32; size_of::<BitAccPack>() / u32::BITS as usize] =
            inline_arr_from_fn(|x| acc_zero_m256i(acc_ptr.add(x)));
        transmute(array) // compiler can statically check size here
    }
}
#[inline(always)] // function used at single call site
pub(crate) fn extract_acc_info_simd(acc: &BitAccPack) -> (BitInt, BitInt) {
    (acc_zero_simd(acc), acc_parity_simd(acc))
}

/// Mask out range of bits
#[must_use]
#[inline(always)]
pub(crate) fn bit_slice(int: BitInt, range: Range<usize>) -> BitInt {
    (int >> range.start) & (((1 as BitInt) << range.len()).wrapping_sub(1))
}

#[must_use]
#[inline(always)]
pub(crate) fn bit_set(int: BitInt, index: usize, set: bool) -> BitInt {
    int | ((BitInt::from(set)) << index)
}
#[must_use]
#[inline(always)]
pub(crate) fn bit_get(int: BitInt, index: u32) -> bool {
    int & (1 << index) != 0
}
#[must_use]
#[inline(always)]
pub(crate) fn wrapping_bit_get(int: BitInt, index: usize) -> bool {
    // Truncating semantics desired here.
    int & (1 as BitInt).wrapping_shl(index as u32) != 0
}
pub(crate) fn pack_bits(arr: [bool; BITS]) -> BitInt {
    let mut tmp_int: BitInt = 0;
    for (i, b) in arr.into_iter().enumerate() {
        tmp_int = bit_set(tmp_int, i, b);
    }
    tmp_int
}

#[cfg(test)]
mod test {
    /// Reference implementation 
    fn acc_parity(acc: &BitAccPack) -> BitInt {
        let acc: &[BitAcc] = &acc.0;
        let mut acc_parity: BitInt = 0;
        for (i, b) in acc.iter().map(|a| a & 1 == 1).enumerate() {
            acc_parity = bit_set(acc_parity, i, b);
        }
        acc_parity
    }
    /// Reference implementation
    fn acc_zero(acc: &BitAccPack) -> BitInt {
        let acc: &[BitAcc] = &acc.0;
        let mut acc_zero: BitInt = 0;
        for (i, b) in acc.iter().map(|a| *a != 0).enumerate() {
            acc_zero = bit_set(acc_zero, i, b);
        }
        acc_zero
    }
    /// Reference implmentation 
    fn extract_acc_info(acc: &BitAccPack) -> (BitInt, BitInt) {
        (acc_zero(acc), acc_parity(acc))
    }
    use super::*;
    #[test]
    fn test_extract_acc_info() {
        let bitpack: BitAccPack = bit_acc_pack(std::array::from_fn(|i| (i % 5) as BitAcc));
        assert_eq!(extract_acc_info(&bitpack), extract_acc_info_simd(&bitpack));
    }
}
