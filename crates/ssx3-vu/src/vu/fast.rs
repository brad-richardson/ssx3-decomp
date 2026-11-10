//! Helpers for statically translated microcode (`generated/vu/`).
//!
//! The reference interpreter and macro mode drive the per-pair model in
//! `mod.rs` (`begin_pair` / staged upper / `end_pair`, every pipeline
//! advanced every pair). Translated code does the same work, but the
//! translator (`recomp.rs`) decides at translate time what each pair
//! actually needs, and calls these small `#[inline(always)]` pieces:
//!
//! - **Stalls:** only the VF fields that can still be in flight are checked
//!   (`stall`). A field written more than 3 pairs back is always ready.
//! - **Results:** the upper op is computed into locals (`fmac`, `plain`),
//!   the lower op runs against the old state, then the upper result is
//!   written (`put_vf`). No staging struct.
//! - **Ready times** (`vf_ready`, `ready_max`) are written only when a read
//!   within 3 pairs, a program end or a dynamic jump can observe them.
//! - **Q, P and the flag pipelines** are retired lazily, right before an op
//!   that observes them (`q_sync`, `p_sync`, `flags_sync`). A flag result
//!   that no flag op can observe within its latency is made visible at once
//!   (`flags_eager`) instead of being queued.
//! - **VI backup** (`vi_bk`, `vi_brf`): the backed-up register carries the
//!   index of the pair that wrote it, instead of being cleared every pair.
//!
//! The result is the same architectural state, cycle count, flags and memory
//! as the per-pair model (checked by the oracle test and `vu_diff digest`).

use super::float::{self, Exc, ABS, SIGN};
use super::{FlagSnap, Vu, CLIP_LATENCY, FMAC_LATENCY, STATUS_DIV, STATUS_FMAC};

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn lane_mac(l: u32, v: u32, e: Exc) -> u32 {
    let b = 3 - l;
    (((v & ABS == 0) as u32) << b)
        | (((v & SIGN != 0) as u32) << (b + 4))
        | ((e.u as u32) << (b + 8))
        | ((e.o as u32) << (b + 12))
}

/// MAC flags of the lanes in `D` (bit 3 = x) from per-lane results.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn mac_of<const D: u8>(v: &[u32; 4], e: &[Exc; 4]) -> u32 {
    let mut mac = 0;
    for l in 0..4 {
        if D & (8 >> l) != 0 {
            mac |= lane_mac(l as u32, v[l], e[l]);
        }
    }
    mac
}

// FMAC ops on all four lanes (straight-line, so LLVM can vectorise them);
// only the lanes in `D` are flagged and later written. Broadcast, Q and I
// operands are passed as `[x; 4]`.
#[allow(unused_macros)]
macro_rules! fmac2 {
    ($name:ident, $f:path) => {
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub fn $name<const D: u8, const NS: bool, const NT: bool>(
            s: &[u32; 4],
            t: &[u32; 4],
        ) -> ([u32; 4], u32) {
            let mut v = [0u32; 4];
            let mut e = [Exc::default(); 4];
            for l in 0..4 {
                let (x, y) = $f(s[l], t[l]);
                v[l] = x;
                e[l] = y;
            }
            (v, mac_of::<D>(&v, &e))
        }
    };
}
#[allow(unused_macros)]
macro_rules! fmac3 {
    ($name:ident, $f:path) => {
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub fn $name<const D: u8, const NA: bool, const NS: bool, const NT: bool>(
            a: &[u32; 4],
            s: &[u32; 4],
            t: &[u32; 4],
        ) -> ([u32; 4], u32) {
            let mut v = [0u32; 4];
            let mut e = [Exc::default(); 4];
            for l in 0..4 {
                let (x, y) = $f(a[l], s[l], t[l]);
                v[l] = x;
                e[l] = y;
            }
            (v, mac_of::<D>(&v, &e))
        }
    };
}
#[cfg(not(target_arch = "aarch64"))]
fmac2!(add4, float::add);
#[cfg(not(target_arch = "aarch64"))]
fmac2!(sub4, float::sub);
#[cfg(not(target_arch = "aarch64"))]
fmac2!(mul4, float::mul);
#[cfg(not(target_arch = "aarch64"))]
fmac3!(madd4, float::madd);
#[cfg(not(target_arch = "aarch64"))]
fmac3!(msub4, float::msub);

// NEON: four lanes per instruction (`simd.rs`), same bits as the scalar code.
#[cfg(target_arch = "aarch64")]
mod neon_ops {
    use super::super::simd::{self, load, store};
    // `NS`, `NT`, `NA`: whether an operand must be normalised (false when
    // the caller knows it is in normal form; see "Known-normal operands").
    macro_rules! neon2 {
        ($name:ident, |$s:ident, $t:ident| $e:expr) => {
            #[cfg_attr(not(debug_assertions), inline(always))]
            pub fn $name<const D: u8, const NS: bool, const NT: bool>(
                s: &[u32; 4],
                t: &[u32; 4],
            ) -> ([u32; 4], u32) {
                let ($s, $t) = (load(s), load(t));
                let r = $e;
                (store(r.v), simd::mac::<D>(&r))
            }
        };
    }
    macro_rules! neon3 {
        ($name:ident, $f:ident) => {
            #[cfg_attr(not(debug_assertions), inline(always))]
            pub fn $name<const D: u8, const NA: bool, const NS: bool, const NT: bool>(
                a: &[u32; 4],
                s: &[u32; 4],
                t: &[u32; 4],
            ) -> ([u32; 4], u32) {
                let r = simd::$f::<NA, NS, NT>(load(a), load(s), load(t));
                (store(r.v), simd::mac::<D>(&r))
            }
        };
    }
    neon2!(add4, |s, t| simd::add_k::<NS, NT>(s, t));
    // (Negating keeps a normal operand normal.)
    neon2!(sub4, |s, t| simd::add_k::<NS, NT>(s, simd::neg(t)));
    neon2!(mul4, |s, t| simd::mul_k::<NS, NT>(s, t));
    neon3!(madd4, madd_k);
    neon3!(msub4, msub_k);

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn itof4<const F: i32>(s: &[u32; 4]) -> [u32; 4] {
        store(simd::itof::<F>(load(s)))
    }
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn ftoi4<const F: i32>(s: &[u32; 4]) -> [u32; 4] {
        store(simd::ftoi::<F>(load(s)))
    }
}
#[cfg(target_arch = "aarch64")]
pub use neon_ops::{add4, ftoi4, itof4, madd4, msub4, mul4, sub4};

#[cfg_attr(not(debug_assertions), inline(always))]
pub fn max4(s: &[u32; 4], t: &[u32; 4]) -> [u32; 4] {
    std::array::from_fn(|l| float::max(s[l], t[l]))
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn min4(s: &[u32; 4], t: &[u32; 4]) -> [u32; 4] {
    std::array::from_fn(|l| float::min(s[l], t[l]))
}
#[cfg(not(target_arch = "aarch64"))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn itof4<const F: i32>(s: &[u32; 4]) -> [u32; 4] {
    std::array::from_fn(|l| float::from_fixed(s[l] as i32, F))
}
#[cfg(not(target_arch = "aarch64"))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn ftoi4<const F: i32>(s: &[u32; 4]) -> [u32; 4] {
    std::array::from_fn(|l| float::to_fixed(s[l], F))
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn abs4(s: &[u32; 4]) -> [u32; 4] {
    std::array::from_fn(|l| s[l] & ABS)
}

// Known-normal operands. Every FMAC op first puts its operands in normal
// form (`float::norm_in`: denormal -> ±0, exponent 255 -> ±max), and the
// `NA`, `NS`, `NT` flags skip that step for one operand. Skipping is exact
// when the operand is already in normal form, because normalising it then
// changes nothing and reports nothing. (The flags only act on AArch64: the
// scalar ops on other targets ignore them and always normalise.)
//
// - An FMAC result is in normal form, given that its operands were
//   normalised or known normal: an overflow packs to ±max (exponent 254),
//   an underflow to ±0, every other result has an exponent in 1..=254, and
//   the zero and cancellation cases give ±0 or a normal operand.
// - The accumulator, where a caller skips it: each ACC lane read into a kept
//   result was written by an FMAC op earlier in the same routine. Lanes
//   outside the destination mask may hold raw bits (host code and test
//   harnesses can set ACC directly), but their results and flags are
//   discarded. The translator uses the same per-lane rule.
// - So is anything that went through [`norm4`] (`norm_in` is idempotent).

/// Every lane of `v` in normal form (see above).
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn is_normal4(v: &[u32; 4]) -> bool {
    v.iter().all(|&x| float::norm_in(x) == x)
}

/// `v` as an FMAC op would see it: each lane through `float::norm_in`.
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn norm4(v: &[u32; 4]) -> [u32; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        use super::simd;
        simd::store(simd::norm(simd::load(v)))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        v.map(float::norm_in)
    }
}

/// `s * t` (all lanes, no flags) for operands already in normal form.
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn mul_normal(s: &[u32; 4], t: &[u32; 4]) -> [u32; 4] {
    debug_assert!(is_normal4(s) && is_normal4(t));
    mul4::<0b1111, false, false>(s, t).0
}

/// `acc + s * t` (all lanes, no flags) for operands already in normal form.
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn madd_normal(acc: &[u32; 4], s: &[u32; 4], t: &[u32; 4]) -> [u32; 4] {
    debug_assert!(is_normal4(acc) && is_normal4(s) && is_normal4(t));
    madd4::<0b1111, false, false, false>(acc, s, t).0
}

/// The columns of a transform, in normal form, so that a loop over many
/// points normalises them once instead of once per point. Bit-identical to
/// the same MULAx / MADDAy / ... sequence with every operand normalised at
/// every step (see above). Only for arithmetic: what goes back to the guest
/// or to the GPU is the original matrix.
#[derive(Clone, Copy)]
pub struct Columns<const N: usize>([[u32; 4]; N]);

impl<const N: usize> Columns<N> {
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn new(columns: &[[u32; 4]; N]) -> Self {
        Self(columns.map(|c| norm4(&c)))
    }

    /// Columns that are FMAC results already.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn from_fmac(columns: [[u32; 4]; N]) -> Self {
        debug_assert!(columns.iter().all(is_normal4));
        Self(columns)
    }

    /// `c0·p.x + c1·p.y + ...` over the columns, in order.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn xform(&self, p: &[u32; 4]) -> [u32; 4] {
        let p = norm4(p);
        #[cfg(target_arch = "aarch64")]
        {
            use super::simd::{k, load, madd_k, mul_k, store};
            let mut acc = mul_k::<false, false>(load(&self.0[0]), k(p[0])).v;
            for (column, lane) in self.0.iter().zip(p).skip(1) {
                acc = madd_k::<false, false, false>(acc, load(column), k(lane)).v;
            }
            store(acc)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut acc = mul_normal(&self.0[0], &[p[0]; 4]);
            for (column, lane) in self.0.iter().zip(p).skip(1) {
                acc = madd_normal(&acc, column, &[lane; 4]);
            }
            acc
        }
    }
}

impl<const N: usize> std::ops::Index<usize> for Columns<N> {
    type Output = [u32; 4];
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn index(&self, column: usize) -> &[u32; 4] {
        &self.0[column]
    }
}

/// Out-of-line copies for the rarely run single-pair arms (keeps the
/// translated functions small).
pub mod cold {
    macro_rules! cold2 {
        ($name:ident) => {
            #[inline(never)]
            pub fn $name<const D: u8>(s: &[u32; 4], t: &[u32; 4]) -> ([u32; 4], u32) {
                super::$name::<D, true, true>(s, t)
            }
        };
    }
    macro_rules! cold3 {
        ($name:ident) => {
            #[inline(never)]
            pub fn $name<const D: u8>(a: &[u32; 4], s: &[u32; 4], t: &[u32; 4]) -> ([u32; 4], u32) {
                super::$name::<D, true, true, true>(a, s, t)
            }
        };
    }
    cold2!(add4);
    cold2!(sub4);
    cold2!(mul4);
    cold3!(madd4);
    cold3!(msub4);
    #[inline(never)]
    pub fn max4(s: &[u32; 4], t: &[u32; 4]) -> [u32; 4] {
        super::max4(s, t)
    }
    #[inline(never)]
    pub fn min4(s: &[u32; 4], t: &[u32; 4]) -> [u32; 4] {
        super::min4(s, t)
    }
    #[inline(never)]
    pub fn itof4<const F: i32>(s: &[u32; 4]) -> [u32; 4] {
        super::itof4::<F>(s)
    }
    #[inline(never)]
    pub fn ftoi4<const F: i32>(s: &[u32; 4]) -> [u32; 4] {
        super::ftoi4::<F>(s)
    }
    #[inline(never)]
    pub fn abs4(s: &[u32; 4]) -> [u32; 4] {
        super::abs4(s)
    }
}

/// `d[l] = v[l]` for the lanes in `D`. On AArch64 a partial write is one
/// 128-bit read-select-write, so a later full-register load forwards from
/// a single store (partial lane stores would stall it).
#[cfg_attr(not(debug_assertions), inline(always))]
fn merge<const D: u8>(d: &mut [u32; 4], v: &[u32; 4]) {
    #[cfg(target_arch = "aarch64")]
    {
        if D == 0b1111 {
            *d = *v;
        } else if D != 0 {
            *d = super::simd::select::<D>(v, d);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for l in 0..4 {
            if D & (8 >> l) != 0 {
                d[l] = v[l];
            }
        }
    }
}

/// CLIP judgement bits of `s.xyz` against `|w|` (see `Vu::clipw`).
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn clip(s: [u32; 4], w: u32) -> u32 {
    let w = float::norm_in(w) & ABS;
    let wk = w as i32;
    let mut flags = 0u32;
    for l in 0..3 {
        let x = float::norm_in(s[l]);
        let k = if x & SIGN != 0 {
            -((x & ABS) as i32)
        } else {
            (x & ABS) as i32
        };
        if k > wk {
            flags |= 1 << (l * 2);
        }
        if k < -wk {
            flags |= 2 << (l * 2);
        }
    }
    flags
}

impl Vu {
    /// Stall until cycle `need` (the max of the ready times of the fields
    /// this pair reads). A stalling pair sees no VI backup (microVU
    /// `analyzeBranchVI`), as in `begin_pair`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn stall(&mut self, need: u64) {
        if need > self.cycle {
            self.cycle = need;
            self.fbk_reg = 0;
        }
    }

    /// Write lanes `D` of VF `r` (r != 0).
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn put_vf<const D: u8>(&mut self, r: usize, v: &[u32; 4]) {
        merge::<D>(&mut self.vf[r & 31], v);
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn put_acc<const D: u8>(&mut self, v: &[u32; 4]) {
        merge::<D>(&mut self.acc, v);
    }

    /// The fields `F` of VF `r` are ready FMAC_LATENCY cycles from now.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn ready<const F: u8>(&mut self, r: usize) {
        let t = self.cycle + FMAC_LATENCY;
        let d = &mut self.vf_ready[r & 31];
        if F & 8 != 0 {
            d[0] = t;
        }
        if F & 4 != 0 {
            d[1] = t;
        }
        if F & 2 != 0 {
            d[2] = t;
        }
        if F & 1 != 0 {
            d[3] = t;
        }
    }

    /// A VF write near a program end or dynamic jump: `flush` must see it.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn ready_max_now(&mut self) {
        let t = self.cycle + FMAC_LATENCY;
        if t > self.ready_max {
            self.ready_max = t;
        }
    }

    /// Commit an FMAC op's MAC flags and the status bits derived from them.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn set_mac(&mut self, mac: u32) {
        self.mac = mac;
        let s = ((mac & 0x000f != 0) as u32)
            | (((mac & 0x00f0 != 0) as u32) << 1)
            | (((mac & 0x0f00 != 0) as u32) << 2)
            | (((mac & 0xf000 != 0) as u32) << 3);
        self.status = (self.status & !0xf) | s | (s << 6);
    }

    /// Commit a run of FMAC ops at once: `last` is the final MAC, `acc` the
    /// OR of all their MACs (the sticky status bits are an OR, so this equals
    /// committing them one by one).
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn mac_commit(&mut self, last: u32, acc: u32) {
        let nib = |m: u32| {
            ((m & 0x000f != 0) as u32)
                | (((m & 0x00f0 != 0) as u32) << 1)
                | (((m & 0x0f00 != 0) as u32) << 2)
                | (((m & 0xf000 != 0) as u32) << 3)
        };
        self.mac = last;
        self.status = (self.status & !0xf) | nib(last) | (nib(acc) << 6);
    }

    /// Flags no flag op can observe before they would be visible anyway:
    /// make them visible now. Older queued snapshots are superseded.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn flags_eager(&mut self) {
        self.vis_mac = self.mac;
        self.vis_status = (self.vis_status & STATUS_DIV) | (self.status & STATUS_FMAC);
        if !self.flag_pipe.is_empty() {
            self.flag_pipe.clear();
        }
    }

    /// Queue the current flags, visible FMAC_LATENCY cycles from now.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn flags_push(&mut self) {
        self.retire_flags();
        self.flag_pipe.push_back(FlagSnap {
            ready: self.cycle + FMAC_LATENCY,
            mac: self.mac,
            status: self.status,
            clip: self.clip,
        });
    }

    /// CLIP: shift in the judgement bits; queue them or make them visible.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn clip_commit(&mut self, c: u32, push: bool) {
        self.clip = ((self.clip << 6) | c) & 0xff_ffff;
        if push {
            self.retire_clip();
            self.clip_pipe
                .push_back((self.cycle + CLIP_LATENCY, self.clip));
            self.flags_push();
        } else {
            self.vis_clip = self.clip;
            if !self.clip_pipe.is_empty() {
                self.clip_pipe.clear();
            }
            self.flags_eager();
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn retire_flags(&mut self) {
        while let Some(f) = self.flag_pipe.front() {
            if f.ready > self.cycle {
                break;
            }
            let f = *f;
            self.flag_pipe.pop_front();
            self.vis_mac = f.mac;
            self.vis_status = (self.vis_status & STATUS_DIV) | (f.status & STATUS_FMAC);
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn retire_clip(&mut self) {
        while let Some(&(ready, c)) = self.clip_pipe.front() {
            if ready > self.cycle {
                break;
            }
            self.clip_pipe.pop_front();
            self.vis_clip = c;
        }
    }

    /// Before a flag op: retire the flag snapshots that are due.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn flags_sync(&mut self) {
        self.retire_flags();
        self.retire_clip();
    }

    /// Before an op that reads Q or the status flags: land a due DIV result.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn q_sync(&mut self) {
        if let Some(p) = self.q_pend {
            if p.ready <= self.cycle {
                self.q_pend = None;
                self.apply_q(p);
            }
        }
    }

    /// Before MFP: land a due EFU result.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn p_sync(&mut self) {
        if let Some(p) = self.p_pend {
            if p.ready <= self.cycle {
                self.p_pend = None;
                self.p = p.value;
            }
        }
    }

    /// VI write that the next pair's conditional branch reads as the old
    /// value (`set_vi_bk`). `tag` is this pair's index (`pairs` numbering).
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn vi_bk(&mut self, r: u8, v: u16, tag: u64) {
        let r = r & 15;
        let old = if self.fbk_reg == r && self.fbk_tag.wrapping_add(1) == tag {
            self.fbk_old
        } else {
            self.vi[r as usize]
        };
        self.fbk_reg = r;
        self.fbk_old = old;
        self.fbk_tag = tag;
        self.vi[r as usize] = v;
    }

    /// VI `r` (!= 0) as read by the conditional branch of pair `tag`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn vi_brf(&self, r: u8, tag: u64) -> u16 {
        let r = r & 15;
        if self.fbk_reg == r && self.fbk_tag.wrapping_add(1) == tag && super::vi_backup_enabled() {
            self.fbk_old
        } else {
            self.vi[r as usize]
        }
    }

    /// Plain VI write (ILW, flag ops, links, XTOP): no backup.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn vi_set(&mut self, r: u8, v: u16) {
        self.set_vi(r, v);
    }

    /// Byte offset of data qword `a` (wrapping at the end of data memory).
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn qoff(&self, a: u32) -> usize {
        ((a & (self.data_qwords() - 1)) as usize) * 16
    }

    /// Load the qword at qword address `a`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn ldq(&self, a: u32) -> [u32; 4] {
        let o = self.qoff(a);
        let b: &[u8; 16] = self.data[o..o + 16].try_into().unwrap();
        [
            u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
            u32::from_le_bytes([b[12], b[13], b[14], b[15]]),
        ]
    }

    /// Store lanes `D` of `v` at qword address `a`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn stq<const D: u8>(&mut self, a: u32, v: &[u32; 4]) {
        let o = self.qoff(a);
        let b: &mut [u8; 16] = (&mut self.data[o..o + 16]).try_into().unwrap();
        if D & 8 != 0 {
            b[0..4].copy_from_slice(&v[0].to_le_bytes());
        }
        if D & 4 != 0 {
            b[4..8].copy_from_slice(&v[1].to_le_bytes());
        }
        if D & 2 != 0 {
            b[8..12].copy_from_slice(&v[2].to_le_bytes());
        }
        if D & 1 != 0 {
            b[12..16].copy_from_slice(&v[3].to_le_bytes());
        }
    }

    /// RNEXT: step the LFSR and return R.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn r_next(&mut self) -> u32 {
        self.lfsr_step();
        self.r
    }
}

#[cfg(test)]
mod known_normal_tests {
    use super::float::{self, Exc};
    use super::*;

    const SIGN_BIT: u32 = 0x8000_0000;
    const MAX_NORMAL: u32 = 0x7f7f_ffff;
    /// Zeros, denormals, exponent 255, the largest normals and the smallest
    /// (with a pair that underflows when subtracted and one that overflows
    /// when added), and ordinary values (1, 2, -1.5, 1000).
    const BOUNDARY: [u32; 19] = [
        0,
        SIGN_BIT,
        1,
        SIGN_BIT | 1,
        0x007f_ffff,
        SIGN_BIT | 0x007f_ffff,
        0x7f80_0000,
        SIGN_BIT | 0x7f80_0000,
        0x7fc0_0001,
        0xffff_ffff,
        MAX_NORMAL,
        SIGN_BIT | MAX_NORMAL,
        0x7f7f_fffe,
        0x0080_0000,
        0x0080_0001,
        0x3f80_0000,
        0x4000_0000,
        0xbfc0_0000,
        0x447a_0000,
    ];

    /// Random cases per test (`SSX3_FLOAT_ITERS`, as in `simd.rs`).
    fn iters() -> u64 {
        std::env::var("SSX3_FLOAT_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50_000)
    }

    struct Words(u64);
    impl Words {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            if self.0 & 7 < 3 {
                BOUNDARY[(self.0 >> 8) as usize % BOUNDARY.len()]
            } else {
                (self.0 >> 16) as u32
            }
        }
        fn vector(&mut self) -> [u32; 4] {
            [self.next(), self.next(), self.next(), self.next()]
        }
    }

    fn lanes(f: impl Fn(usize) -> (u32, Exc)) -> [u32; 4] {
        std::array::from_fn(|l| f(l).0)
    }

    #[test]
    fn norm4_is_norm_in_and_idempotent() {
        for x in BOUNDARY {
            let n = norm4(&[x; 4]);
            assert_eq!(n, [float::norm_in(x); 4], "{x:#010x}");
            assert_eq!(norm4(&n), n, "{x:#010x}");
            assert!(is_normal4(&n), "{x:#010x}");
        }
        assert_eq!(float::norm_in(1), 0);
        assert_eq!(float::norm_in(SIGN_BIT | 1), SIGN_BIT);
        assert_eq!(float::norm_in(0x7f80_0000), MAX_NORMAL);
        assert_eq!(float::norm_in(0xffff_ffff), SIGN_BIT | MAX_NORMAL);
        assert_eq!(float::norm_in(MAX_NORMAL), MAX_NORMAL);
    }

    #[test]
    fn fmac_results_are_in_normal_form() {
        for a in BOUNDARY {
            for b in BOUNDARY {
                for c in BOUNDARY {
                    let (a4, b4, c4) = ([a; 4], [b; 4], [c; 4]);
                    assert!(
                        is_normal4(&add4::<0b1111, true, true>(&a4, &b4).0),
                        "{a:#010x} {b:#010x} {c:#010x}"
                    );
                    assert!(
                        is_normal4(&sub4::<0b1111, true, true>(&a4, &b4).0),
                        "{a:#010x} {b:#010x} {c:#010x}"
                    );
                    assert!(
                        is_normal4(&mul4::<0b1111, true, true>(&a4, &b4).0),
                        "{a:#010x} {b:#010x} {c:#010x}"
                    );
                    let m = madd4::<0b1111, true, true, true>(&c4, &a4, &b4).0;
                    assert!(is_normal4(&m), "{a:#010x} {b:#010x} {c:#010x}");
                    let m = msub4::<0b1111, true, true, true>(&c4, &a4, &b4).0;
                    assert!(is_normal4(&m), "{a:#010x} {b:#010x} {c:#010x}");
                }
            }
        }
    }

    #[test]
    fn skipping_normalisation_is_exact() {
        // The scalar reference normalises every operand at every step.
        for a in BOUNDARY {
            for b in BOUNDARY {
                for c in BOUNDARY {
                    let (s, t) = (norm4(&[a; 4]), norm4(&[b; 4]));
                    let acc = norm4(&[c; 4]);
                    assert_eq!(
                        mul_normal(&s, &t),
                        [float::mul_ref(a, b).0; 4],
                        "{a:#010x} {b:#010x} {c:#010x}"
                    );
                    assert_eq!(
                        madd_normal(&acc, &s, &t),
                        [float::madd(c, a, b).0; 4],
                        "{a:#010x} {b:#010x} {c:#010x}"
                    );
                    // The accumulator flag alone, flags included: raw `s`
                    // and `t`, normal `acc`.
                    let (v, mac) = madd4::<0b1111, false, true, true>(&acc, &[a; 4], &[b; 4]);
                    let (want_v, want_mac) =
                        madd4::<0b1111, true, true, true>(&[c; 4], &[a; 4], &[b; 4]);
                    assert_eq!(
                        (v, mac),
                        (want_v, want_mac),
                        "{a:#010x} {b:#010x} {c:#010x}"
                    );
                    let (v, mac) = msub4::<0b1111, false, true, true>(&acc, &[a; 4], &[b; 4]);
                    assert_eq!(
                        v,
                        [float::msub(c, a, b).0; 4],
                        "{a:#010x} {b:#010x} {c:#010x}"
                    );
                    let (want_v, want_mac) =
                        msub4::<0b1111, true, true, true>(&[c; 4], &[a; 4], &[b; 4]);
                    assert_eq!(
                        (v, mac),
                        (want_v, want_mac),
                        "{a:#010x} {b:#010x} {c:#010x}"
                    );
                }
            }
        }
    }

    fn scalar_xform<const N: usize>(m: &[[u32; 4]; N], p: &[u32; 4]) -> [u32; 4] {
        lanes(|l| {
            let mut acc = float::mul_ref(m[0][l], p[0]);
            for c in 1..N {
                acc = float::madd(acc.0, m[c][l], p[c]);
            }
            acc
        })
    }

    #[test]
    fn columns_xform_matches_every_step_normalised() {
        let mut w = Words(0x2545_f491_4f6c_dd1d);
        for _ in 0..iters() {
            let m4 = [w.vector(), w.vector(), w.vector(), w.vector()];
            let p = w.vector();
            let c4 = Columns::new(&m4);
            assert_eq!(c4.xform(&p), scalar_xform(&m4, &p), "{m4:x?} {p:x?}");
            let m3 = [m4[0], m4[1], m4[2]];
            assert_eq!(
                Columns::new(&m3).xform(&p),
                scalar_xform(&m3, &p),
                "{m3:x?} {p:x?}"
            );
        }
    }

    #[test]
    fn columns_from_fmac_results_match() {
        let mut w = Words(0x9e37_79b9_7f4a_7c15);
        for _ in 0..iters() / 4 {
            let raw = [w.vector(), w.vector(), w.vector(), w.vector()];
            let (a, b) = (w.vector(), w.vector());
            let fmac: [[u32; 4]; 4] = std::array::from_fn(|c| {
                add4::<0b1111, true, true>(&mul4::<0b1111, true, true>(&raw[c], &a).0, &b).0
            });
            let p = w.vector();
            assert_eq!(Columns::from_fmac(fmac).xform(&p), scalar_xform(&fmac, &p));
        }
    }

    #[test]
    fn columns_read_exponent_255_as_max() {
        // As the FMAC does: an exponent-255 column acts as ±max, so halving
        // it gives max / 2 (un-normalised it would give 0x7f00_0000).
        let m = [[0x7f80_0000; 4], [0; 4], [0; 4], [0; 4]];
        let p = [0x3f00_0000, 0, 0, 0];
        assert_eq!(Columns::new(&m).xform(&p), [0x7eff_ffff; 4]);
    }
}
