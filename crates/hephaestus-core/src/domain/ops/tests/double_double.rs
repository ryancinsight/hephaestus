//! Double-double (about 106-bit) arithmetic for accuracy references.
//!
//! A value is the unevaluated sum `hi + lo` of two `f64`s with
//! `|lo| ≤ ulp(hi) / 2`. Sums and products use the error-free transformations
//! `two_sum` (Knuth) and `two_prod` (a fused multiply-add recovers the
//! product's rounding error exactly), so each operation carries a relative
//! error near `2^-104`. Only `f64` `+ - * /` and `mul_add` are used: no
//! transcendental from `std` or eunomia enters a reference built here.

/// `k` as an `i32`; `k` is an integer-valued `f64` of magnitude below 1100.
#[expect(
    clippy::cast_possible_truncation,
    reason = "k is integer-valued and |k| < 1100, so the conversion is exact"
)]
fn integer_exponent(k: f64) -> i32 {
    k as i32
}

/// An unevaluated sum `hi + lo`.
#[derive(Clone, Copy, Debug)]
pub(super) struct DoubleDouble {
    pub(super) hi: f64,
    pub(super) lo: f64,
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

fn quick_two_sum(a: f64, b: f64) -> DoubleDouble {
    let s = a + b;
    DoubleDouble {
        hi: s,
        lo: b - (s - a),
    }
}

impl DoubleDouble {
    pub(super) const fn from_f64(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    pub(super) fn add(self, other: Self) -> Self {
        let (s, e) = two_sum(self.hi, other.hi);
        let (t, f) = two_sum(self.lo, other.lo);
        let r = quick_two_sum(s, e + t);
        quick_two_sum(r.hi, r.lo + f)
    }

    pub(super) fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    pub(super) fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }

    pub(super) fn mul(self, other: Self) -> Self {
        let p = self.hi * other.hi;
        let e = self.hi.mul_add(other.hi, -p);
        quick_two_sum(p, e + (self.hi * other.lo + self.lo * other.hi))
    }

    fn mul_f64(self, other: f64) -> Self {
        self.mul(Self::from_f64(other))
    }

    pub(super) fn div(self, other: Self) -> Self {
        let q1 = self.hi / other.hi;
        let r = self.sub(other.mul_f64(q1));
        let q2 = r.hi / other.hi;
        let r = r.sub(other.mul_f64(q2));
        let q3 = r.hi / other.hi;
        quick_two_sum(q1, q2).add(Self::from_f64(q3))
    }

    /// `exp(self)` for `|self| < 700`: reduce by `k·ln 2` so that
    /// `|r| ≤ ln 2 / 2`, sum the Taylor series of `exp(r)` to 27 terms (the
    /// remainder `(ln 2 / 2)^28 / 28!` is below `2^-130`), and scale by `2^k`.
    pub(super) fn exp(self) -> Self {
        const LN_2: DoubleDouble = DoubleDouble {
            hi: core::f64::consts::LN_2,
            lo: 2.319_046_813_846_299_6e-17,
        };
        debug_assert!(self.hi.abs() < 700.0, "reference exp argument out of range");
        let k = (self.hi / core::f64::consts::LN_2).round();
        let r = self.sub(LN_2.mul_f64(k));
        let mut sum = Self::from_f64(1.0);
        for n in (1..=27_u32).rev() {
            sum = Self::from_f64(1.0).add(r.mul(sum).div(Self::from_f64(f64::from(n))));
        }
        let scale = 2f64.powi(integer_exponent(k));
        Self {
            hi: sum.hi * scale,
            lo: sum.lo * scale,
        }
    }
}
