//! A list that is just a raw array that is manipulated directly.
//! Very unsafe but slightly faster than a normal vector
//! This is needed because it's used in the innermost loop
#![allow(dead_code)]
#![allow(clippy::inline_always)]

#[derive(Debug, Default, Clone)]
pub(crate) struct RawList<T>
where
    T: Default + Clone,
{
    list: Box<[T]>,
    len: usize,
}
impl<T> RawList<T>
where
    T: Default + Clone,
{
    /// SAFE
    pub(crate) fn collect_size(iter: impl Iterator<Item = T>, max_size: usize) -> Self {
        let mut list = Self::new(max_size);
        for el in iter {
            list.push_safe(el);
        }
        list
    }

    /// SAFE
    pub(crate) fn collect(&mut self, iter: impl Iterator<Item = T>) {
        self.clear();
        for el in iter {
            self.push_safe(el);
        }
    }

    /// SAFE
    pub(crate) fn new(max_size: usize) -> Self {
        RawList {
            list: vec![T::default(); max_size].into_boxed_slice(),
            len: 0,
        }
    }

    /// SAFE
    pub(crate) fn push_safe(&mut self, el: T) {
        self.list[self.len] = el;
        self.len += 1;
    }

    /// SAFE
    #[inline(always)]
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }
    /// # Safety
    /// Length must not become longer than capacity.
    #[inline(always)]
    pub(crate) unsafe fn push_unchecked(&mut self, el: T) {
        *unsafe { self.list.get_unchecked_mut(self.len) } = el;
        self.len += 1;
        debug_assert!(
            self.list.len() > self.len,
            "{} <= {}",
            self.list.len(),
            self.len
        );
    }
    /// SAFE because checked during construction
    #[inline(always)]
    pub(crate) fn get_slice(&self) -> &[T] {
        debug_assert!(
            self.list.len() > self.len,
            "{} <= {}",
            self.list.len(),
            self.len
        );
        unsafe { self.list.get_unchecked(..self.len) }
    }
    /// SAFE because checked during construction
    #[inline(always)]
    pub(crate) fn get_slice_mut(&mut self) -> &mut [T] {
        debug_assert!(
            self.list.len() > self.len,
            "{} <= {}",
            self.list.len(),
            self.len
        );
        unsafe { self.list.get_unchecked_mut(..self.len) }
    }
    /// SAFE
    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }
    /// SAFE
    #[inline(always)]
    pub(crate) fn capacity(&self) -> usize {
        self.list.len()
    }
    /// SAFE
    pub(crate) fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.get_slice().iter().cloned()
    }
}
