use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use serde::{de::SeqAccess, de::Visitor, Deserialize, Deserializer, Serialize, Serializer};

/// 零堆分配的高性能固定容量栈数组容器 (天然实现 Copy, Clone，专为极大化游戏状态拷贝性能设计)
#[derive(Clone, Copy)]
pub struct StackVec<T: Copy, const CAP: usize> {
    data: [std::mem::MaybeUninit<T>; CAP],
    len: usize,
}

impl<T: Copy, const CAP: usize> Default for StackVec<T, CAP> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const CAP: usize> StackVec<T, CAP> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [const { std::mem::MaybeUninit::uninit() }; CAP],
            len: 0,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub const fn capacity(&self) -> usize {
        CAP
    }

    #[inline]
    pub fn push(&mut self, val: T) {
        assert!(self.len < CAP, "StackVec capacity {} exceeded", CAP);
        self.data[self.len].write(val);
        self.len += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(unsafe { self.data[self.len].assume_init() })
        }
    }

    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "Index out of bounds");
        let val = unsafe { self.data[index].assume_init() };
        // 将后续元素向前移动一位
        for i in index..self.len - 1 {
            self.data[i] = self.data[i + 1];
        }
        self.len -= 1;
        val
    }

    #[inline]
    pub fn insert(&mut self, index: usize, val: T) {
        assert!(self.len < CAP, "StackVec capacity exceeded");
        assert!(index <= self.len, "Index out of bounds");
        for i in (index..self.len).rev() {
            self.data[i + 1] = self.data[i];
        }
        self.data[index].write(val);
        self.len += 1;
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len)
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len)
        }
    }

    #[inline]
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T: Copy, const CAP: usize> Deref for StackVec<T, CAP> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Copy, const CAP: usize> DerefMut for StackVec<T, CAP> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Copy, const CAP: usize> Index<usize> for StackVec<T, CAP> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T: Copy, const CAP: usize> IndexMut<usize> for StackVec<T, CAP> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<T: Copy + PartialEq, const CAP: usize> PartialEq for StackVec<T, CAP> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Eq, const CAP: usize> Eq for StackVec<T, CAP> {}

impl<T: Copy + fmt::Debug, const CAP: usize> fmt::Debug for StackVec<T, CAP> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<T: Copy, const CAP: usize> IntoIterator for StackVec<T, CAP> {
    type Item = T;
    type IntoIter = StackVecIntoIter<T, CAP>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        StackVecIntoIter { vec: self, idx: 0 }
    }
}

impl<'a, T: Copy, const CAP: usize> IntoIterator for &'a StackVec<T, CAP> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T: Copy, const CAP: usize> IntoIterator for &'a mut StackVec<T, CAP> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

pub struct StackVecIntoIter<T: Copy, const CAP: usize> {
    vec: StackVec<T, CAP>,
    idx: usize,
}

impl<T: Copy, const CAP: usize> Iterator for StackVecIntoIter<T, CAP> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.vec.len {
            let item = self.vec[self.idx];
            self.idx += 1;
            Some(item)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.vec.len - self.idx;
        (rem, Some(rem))
    }
}

impl<T: Copy + Serialize, const CAP: usize> Serialize for StackVec<T, CAP> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_slice().serialize(serializer)
    }
}

impl<'de, T: Copy + Deserialize<'de>, const CAP: usize> Deserialize<'de> for StackVec<T, CAP> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StackVecVisitor<T, const CAP: usize> {
            _marker: std::marker::PhantomData<T>,
        }

        impl<'de, T: Copy + Deserialize<'de>, const CAP: usize> Visitor<'de> for StackVecVisitor<T, CAP> {
            type Value = StackVec<T, CAP>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "a sequence with at most {} elements", CAP)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut vec = StackVec::new();
                while let Some(element) = seq.next_element()? {
                    if vec.len >= CAP {
                        return Err(serde::de::Error::custom(format!(
                            "StackVec capacity {} exceeded",
                            CAP
                        )));
                    }
                    vec.push(element);
                }
                Ok(vec)
            }
        }

        deserializer.deserialize_seq(StackVecVisitor {
            _marker: std::marker::PhantomData,
        })
    }
}
