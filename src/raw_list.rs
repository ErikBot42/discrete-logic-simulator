/// A list that is just a raw array that is manipulated directly.
/// Very unsafe but slightly faster than a normal vector
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
    pub(crate) fn collect_size(iter: impl Iterator<Item = T>, max_size: usize) -> Self {
        let mut list = Self::new(max_size);
        for el in iter {
            list.push_safe(el);
        }
        list
    }

    pub(crate) fn collect(&mut self, iter: impl Iterator<Item = T>) {
        self.clear();
        for el in iter {
            self.push_safe(el);
        }
    }

    pub(crate) fn new(max_size: usize) -> Self {
        RawList {
            list: vec![T::default(); max_size].into_boxed_slice(),
            len: 0,
        }
    }
    #[inline(always)]
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }
    pub(crate) fn push_safe(&mut self, el: T) {
        self.list[self.len] = el;
        self.len += 1;
    }
    #[inline(always)]
    pub(crate) unsafe fn push(&mut self, el: T) {
        *unsafe { self.list.get_unchecked_mut(self.len) } = el;
        self.len += 1;
        debug_assert!(
            self.list.len() > self.len,
            "{} <= {}",
            self.list.len(),
            self.len
        );
    }
    #[inline(always)]
    pub(crate) unsafe fn get_slice(&self) -> &[T] {
        // &self.list[0..self.len]
        debug_assert!(
            self.list.len() > self.len,
            "{} <= {}",
            self.list.len(),
            self.len
        );
        unsafe { self.list.get_unchecked(..self.len) }
    }
    #[inline(always)]
    pub(crate) unsafe fn get_slice_mut(&mut self) -> &mut [T] {
        // &self.list[0..self.len]
        debug_assert!(
            self.list.len() > self.len,
            "{} <= {}",
            self.list.len(),
            self.len
        );
        unsafe { self.list.get_unchecked_mut(..self.len) }
    }
    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}
