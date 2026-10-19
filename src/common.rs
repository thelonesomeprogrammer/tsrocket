use num_complex::Complex;
use realfft::num_complex;
use smallvec::SmallVec;
use std::simd::f32x4;

pub const LANES: usize = 4;

#[derive(Clone, Debug)]
pub struct SlidingDFT {
    pub n: usize,
    /// Updates since the bins were last computed by a real FFT.
    pub updates: usize,
    pub bins: Vec<Complex<f32>>,
    pub twiddles: Vec<Complex<f32>>,
}

impl SlidingDFT {
    pub fn new(n: usize) -> Self {
        let n_bins = n / 2 + 1;
        let mut twiddles = Vec::with_capacity(n_bins);
        for k in 0..n_bins {
            // Multiples of a quarter turn exactly (±1, ±i): an f32 sin(π) is
            // -8.7e-8, not 0, which would leak into the DC/Nyquist/n/4 bins
            // even for exact (e.g. integer) data. The rest computed in f64.
            let twiddle = match (4 * k % n == 0).then(|| 4 * k / n) {
                Some(0) => Complex::new(1.0, 0.0),
                Some(1) => Complex::new(0.0, 1.0),
                Some(2) => Complex::new(-1.0, 0.0),
                _ => {
                    let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                    Complex::new(angle.cos() as f32, angle.sin() as f32)
                }
            };
            twiddles.push(twiddle);
        }
        Self {
            n,
            updates: 0,
            bins: vec![Complex::new(0.0, 0.0); n_bins],
            twiddles,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, old_val: f32, new_val: f32) {
        self.updates += 1;
        let diff = new_val - old_val;
        for (k, twiddle) in self.twiddles.iter().enumerate() {
            // S_k(n+1) = twiddle_k * (S_k(n) + new - old)
            self.bins[k] = (self.bins[k] + diff) * twiddle;
        }
    }

    pub fn from_fft(fft_complex: Vec<Complex<f32>>, n: usize) -> Self {
        let mut s = Self::new(n);
        s.bins = fft_complex;
        s
    }
}

#[derive(Clone)]
pub struct ColumnState {
    pub total_sum: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub energy: f32,
    pub sum_cubes: f32,
    pub sum_quads: f32,
    pub mac_sum: f32,
    pub mc_sum: f32,
    pub sum_sq_diff: f32,
    pub sum_ix: f32,
    pub t_energy: f32,
    pub mac_sum_vec: f32x4,
    pub mc_sum_vec: f32x4,

    // SIMD accumulators
    pub sum_vec: f32x4,
    pub energy_vec: f32x4,
    pub sum_cubes_vec: f32x4,
    pub sum_quads_vec: f32x4,
    pub min_vec: f32x4,
    pub max_vec: f32x4,
    pub abs_max_vec: f32x4,
    pub sum_sq_diff_vec: f32x4,
    pub sum_ix_vec: f32x4,
    pub t_energy_vec: f32x4,

    pub zcr_count: u32,
    pub peaks: u32,
    pub troughs: u32,
    pub zc_indices: SmallVec<[f32; 32]>,
    pub paa_sums: Vec<Vec<f32>>,
    pub current_paa_segs: Vec<usize>,
    pub c3_sums: Vec<f32>,
    pub c3_sums_vec: Vec<f32x4>,
    pub count_above_counts: Vec<u32>,
    pub count_below_counts: Vec<u32>,
    pub range_counts: Vec<u32>,
    pub autocorr_sums: Vec<f32>,
    pub tra_sums: Vec<f32>,
    pub tra_sums_vec: Vec<f32x4>,
    pub prefix_sums: Vec<f32>,
    pub prev_last: f32,
    pub prev_val: f32,
    pub prev_prev_val: f32,
    pub abs_max: f32,
    pub first_max_idx: usize,
    pub last_max_idx: usize,
    pub first_min_idx: usize,
    pub last_min_idx: usize,
    pub benford_counts: [usize; 9],
    pub min_queue: Vec<(usize, f32)>,
    pub min_q_head: usize,
    pub min_q_tail: usize,
    pub min_q_len: usize,
    pub max_queue: Vec<(usize, f32)>,
    pub max_q_head: usize,
    pub max_q_tail: usize,
    pub max_q_len: usize,
    pub approx_entropy_buffer: Vec<usize>,
    pub binned_entropy_buffer: Vec<f32>,
    /// numpy bin edges as exact f32 thresholds (binned_entropy, LZ complexity).
    pub bin_edges: Vec<f32>,
    pub agg_linear_trend_buffer: Vec<f32>,
    pub mse_buffer: Vec<f32>,
    /// Per-template match counts for `approx_entropy`.
    pub approx_entropy_counts: Vec<i32>,
    /// `(r bits, A, B)` from `sample_entropy_counts(values, 2, r)` on the current
    /// window, shared by `sample_entropy` and `tsfresh_sample_entropy`.
    pub sample_entropy_counts: Option<(u32, u32, u32)>,
    pub workspace_f64_1: Vec<f64>,
    pub workspace_f64_2: Vec<f64>,
    pub workspace_f64_3: Vec<f64>,
    pub workspace_f64_4: Vec<f64>,
    pub workspace_f64_5: Vec<f64>,
    pub workspace_f64_6: Vec<f64>,
    pub workspace_f64_7: Vec<f64>,
    pub higuchi_lk: Vec<f64>,
    pub higuchi_k_values: Vec<f64>,
    pub sort_buffer: Vec<f32>,
    pub mad_buffer: Vec<f32>,
    pub diff_buffer: Vec<f32>,
    /// Bin counts for `hist_mode`.
    pub hist_counts: Vec<u32>,
    pub pacf_buffer: Vec<f32>,
    pub cwt_final_energy: Vec<f32>,
    pub cwt_final_sum_abs: Vec<f32>,
    pub cwt_final_clnc: Vec<f32>,
    pub value_counts: rustc_hash::FxHashMap<u32, u32>,
    pub reoccurring_datapoints: u32,
    pub reoccurring_values: u32,
    pub ar_coeffs: rustc_hash::FxHashMap<u16, Vec<f32>>,
    pub friedrich_coeffs: rustc_hash::FxHashMap<(u8, u32), Vec<f32>>,
    pub max_langevin_fixed_point_cache: rustc_hash::FxHashMap<(u8, u32), f32>,

    // Incremental moments (Welford's or similar)
    pub n: f32,
    pub mean: f32,
    pub m2: f32,
    pub m3: f32,
    pub m4: f32,
    pub last_fft_n: usize,
    pub last_fft_complex: Vec<num_complex::Complex<f32>>,
    pub fft_in_buffer: Vec<f32>,
    pub fft_out_buffer: Vec<num_complex::Complex<f32>>,
    pub fft_inv_buffer: Vec<f32>,
    pub sliding_dft: Option<SlidingDFT>,
    pub cwt_peaks: u16,
    pub welch_density: Vec<f32>,
    /// Per-window spectrogram means for `spectrogram_mean_coeff`.
    pub spectrogram: crate::spectral::SpectrogramCache,
    pub spectrum_buffer: Vec<f32>,
    pub adf_test_stat: f32,
    pub adf_p_value: f32,
    pub adf_used_lag: f32,
    pub lz_trie_nodes: Vec<LzNode>,
    pub lz_symbol_buffer: Vec<u8>,
    pub lz_binary_trie: Vec<[usize; 2]>,
    pub lz_bit_buffer: Vec<u64>,
    /// Matrix profile of the current window, shared by every `matrix_profile-m-*`
    /// aggregate; valid for subsequence length `matrix_profile_m` (0 = not computed).
    pub matrix_profile: Vec<f32>,
    pub matrix_profile_m: usize,
    pub matrix_profile_scratch: Vec<f32>,
    /// Sampling frequency (Hz) TSFEL's spectral features are evaluated at.
    pub fs: f32,
}

pub fn next_good_fft_size(n: usize) -> usize {
    if n <= 2 {
        return n;
    }

    let limit = (n as f32 * 1.05) as usize;
    for candidate in n..=limit {
        if is_smooth(candidate) {
            return candidate;
        }
    }
    // Fallback to next power of 2 if no smooth number in 5% neighborhood
    n.next_power_of_two()
}

fn is_smooth(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    for &p in &[2, 3, 5, 7] {
        while n % p == 0 {
            n /= p;
        }
    }
    n == 1
}

impl ColumnState {
    /// Drop results cached while evaluating the previous window. Values or
    /// buffer addresses can't tell windows apart: the sliding engine reuses its
    /// history buffer in place, so every window has the same pointer and length.
    pub fn reset_window_caches(&mut self) {
        self.ar_coeffs.clear();
        self.friedrich_coeffs.clear();
        self.max_langevin_fixed_point_cache.clear();
        self.adf_test_stat = f32::NAN;
        self.adf_p_value = f32::NAN;
        self.adf_used_lag = f32::NAN;
        self.lz_trie_nodes.clear();
        self.lz_symbol_buffer.clear();
        self.lz_binary_trie.clear();
        self.lz_bit_buffer.clear();
        self.matrix_profile_m = 0;
        self.sample_entropy_counts = None;
        self.higuchi_lk.clear();
        self.higuchi_k_values.clear();
        self.spectrogram.invalidate();
    }

    pub fn new(
        unique_paa_totals: &[u16],
        unique_c3_lags: &[u16],
        unique_autocorr_lags: &[u16],
        unique_tra_lags: &[u16],
        unique_count_above_thresholds: &[u32],
        unique_count_below_thresholds: &[u32],
        unique_range_counts: &[(u32, u32)],
        first_val: f32,
        fs: f32,
    ) -> Self {
        Self {
            total_sum: 0.0,
            min_value: f32::INFINITY,
            max_value: f32::NEG_INFINITY,
            energy: 0.0,
            sum_cubes: 0.0,
            sum_quads: 0.0,
            mac_sum: 0.0,
            mc_sum: 0.0,
            sum_sq_diff: 0.0,
            sum_ix: 0.0,
            mac_sum_vec: f32x4::splat(0.0),
            mc_sum_vec: f32x4::splat(0.0),
            sum_vec: f32x4::splat(0.0),
            energy_vec: f32x4::splat(0.0),
            sum_cubes_vec: f32x4::splat(0.0),
            sum_quads_vec: f32x4::splat(0.0),
            min_vec: f32x4::splat(f32::INFINITY),
            max_vec: f32x4::splat(f32::NEG_INFINITY),
            abs_max_vec: f32x4::splat(0.0),
            t_energy: 0.0,
            sum_sq_diff_vec: f32x4::splat(0.0),
            sum_ix_vec: f32x4::splat(0.0),
            t_energy_vec: f32x4::splat(0.0),
            zcr_count: 0,
            peaks: 0,
            troughs: 0,
            zc_indices: SmallVec::new(),
            paa_sums: unique_paa_totals
                .iter()
                .map(|&total| vec![0.0; total as usize])
                .collect(),
            current_paa_segs: vec![0usize; unique_paa_totals.len()],
            c3_sums: vec![0.0; unique_c3_lags.len()],
            c3_sums_vec: vec![f32x4::splat(0.0); unique_c3_lags.len()],
            count_above_counts: vec![0; unique_count_above_thresholds.len()],
            count_below_counts: vec![0; unique_count_below_thresholds.len()],
            range_counts: vec![0; unique_range_counts.len()],
            autocorr_sums: vec![0.0; unique_autocorr_lags.len()],
            tra_sums: vec![0.0; unique_tra_lags.len()],
            tra_sums_vec: vec![f32x4::splat(0.0); unique_tra_lags.len()],
            prefix_sums: Vec::new(),
            prev_last: first_val,
            prev_val: first_val,
            prev_prev_val: first_val,
            abs_max: 0.0,
            first_max_idx: 0,
            last_max_idx: 0,
            first_min_idx: 0,
            last_min_idx: 0,
            benford_counts: [0; 9],
            min_queue: Vec::new(),
            min_q_head: 0,
            min_q_tail: 0,
            min_q_len: 0,
            max_queue: Vec::new(),
            max_q_head: 0,
            max_q_tail: 0,
            max_q_len: 0,
            approx_entropy_buffer: Vec::new(),
            binned_entropy_buffer: Vec::new(),
            bin_edges: Vec::new(),
            agg_linear_trend_buffer: Vec::new(),
            mse_buffer: Vec::new(),
            approx_entropy_counts: Vec::new(),
            sample_entropy_counts: None,
            workspace_f64_1: Vec::new(),
            workspace_f64_2: Vec::new(),
            workspace_f64_3: Vec::new(),
            workspace_f64_4: Vec::new(),
            workspace_f64_5: Vec::new(),
            workspace_f64_6: Vec::new(),
            workspace_f64_7: Vec::new(),
            higuchi_lk: Vec::new(),
            higuchi_k_values: Vec::new(),
            sort_buffer: Vec::new(),
            mad_buffer: Vec::new(),
            diff_buffer: Vec::new(),
            hist_counts: Vec::new(),
            pacf_buffer: Vec::new(),
            cwt_final_energy: Vec::new(),
            cwt_final_sum_abs: Vec::new(),
            cwt_final_clnc: Vec::new(),
            value_counts: rustc_hash::FxHashMap::default(),
            reoccurring_datapoints: 0,
            reoccurring_values: 0,
            ar_coeffs: rustc_hash::FxHashMap::default(),
            friedrich_coeffs: rustc_hash::FxHashMap::default(),
            max_langevin_fixed_point_cache: rustc_hash::FxHashMap::default(),
            n: 0.0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            last_fft_n: 0,
            last_fft_complex: Vec::new(),
            fft_in_buffer: Vec::new(),
            fft_out_buffer: Vec::new(),
            fft_inv_buffer: Vec::new(),
            sliding_dft: None,
            cwt_peaks: 0,
            welch_density: Vec::new(),
            matrix_profile: Vec::new(),
            matrix_profile_m: 0,
            matrix_profile_scratch: Vec::new(),
            spectrogram: Default::default(),
            spectrum_buffer: Vec::new(),
            adf_test_stat: std::f32::NAN,
            adf_p_value: std::f32::NAN,
            adf_used_lag: std::f32::NAN,
            lz_trie_nodes: Vec::new(),
            lz_symbol_buffer: Vec::new(),
            lz_binary_trie: Vec::new(),
            lz_bit_buffer: Vec::new(),
            fs,
        }
    }
}

use crate::types::{Compute, Feature};

pub(crate) fn map_features_to_indices(features: &[Feature]) -> Compute {
    features
        .iter()
        .fold(Compute::empty(), |acc, f| acc | f.required_compute())
}

use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

pub fn sample_entropy_simd(data: &[f32], m: usize, r: f32) -> f32 {
    if data.len() <= m {
        return f32::NAN;
    }
    let (a_count, b_count) = sample_entropy_counts(data, m, r);
    sample_entropy_from_counts(a_count, b_count)
}

pub fn sample_entropy_from_counts(a_count: u32, b_count: u32) -> f32 {
    if b_count == 0 || a_count == 0 {
        return f32::NAN;
    }

    // if all elements are the same, r=0 and everything matches, TSFEL returns nan, TSFresh returns 0.
    // the python tests check for np.isnan(res_const), so we return NAN if variance is zero
    // OR we return NAN if we are in MSE and var is zero. Wait, if all match, a/b = 1 => ln(1)=0.
    // Let's just return NaN if variance is effectively zero and we have a flat signal.
    let val = -(a_count as f32 / b_count as f32).ln();
    if val == -0.0 { 0.0 } else { val }
}

/// tsfresh's sample entropy: like `sample_entropy_simd`, but B also counts the
/// last length-m template (all n - m + 1 of them), and no NaN guard: A == 0
/// gives inf, B == 0 gives NaN, as `-np.log(A / B)` does.
pub fn tsfresh_sample_entropy(data: &[f32], m: usize, r: f32) -> f32 {
    let n = data.len();
    if n <= m {
        return f32::NAN;
    }
    let (a_count, b_count) = sample_entropy_counts(data, m, r);
    tsfresh_sample_entropy_from_counts(data, m, r, a_count, b_count)
}

/// `tsfresh_sample_entropy` given the `sample_entropy_counts(data, m, r)` result.
pub fn tsfresh_sample_entropy_from_counts(
    data: &[f32],
    m: usize,
    r: f32,
    a_count: u32,
    mut b_count: u32,
) -> f32 {
    let n = data.len();
    if n <= m {
        return f32::NAN;
    }
    let last = n - m;
    let extra = (0..last)
        .filter(|&j| (0..m).all(|k| (data[j + k] - data[last + k]).abs() <= r))
        .count() as u32;
    b_count += 2 * extra;
    -(a_count as f32 / b_count as f32).ln()
}

/// Ordered pair counts (A, B) over the first n - m templates: B matches of
/// length m, A matches of length m + 1 (Chebyshev distance <= r, i != j).
pub fn sample_entropy_counts(data: &[f32], m: usize, r: f32) -> (u32, u32) {
    use std::simd::{Select, i32x4, num::SimdInt};
    let n = data.len();
    let r_vec = f32x4::splat(r);
    let (one, zero) = (i32x4::splat(1), i32x4::splat(0));

    // Per-lane counts, reduced once at the end. Two f32x4 per step (128-bit
    // SIMD: SSE2, NEON) for instruction-level parallelism.
    let mut b_acc = zero;
    let mut a_acc = zero;
    let mut b_count = 0;
    let mut a_count = 0;
    let end_m = n - m; // Number of templates of length m+1 is end_m.

    // We only need to check i < j and then multiply by 2 because distance is symmetric!
    for i in 0..end_m {
        let mut j = i + 1; // only check j > i
        while j + 8 <= end_m {
            let mut max_lo = f32x4::splat(0.0);
            let mut max_hi = f32x4::splat(0.0);
            for k in 0..m {
                let v_i = f32x4::splat(data[i + k]);
                let lo = f32x4::from_slice(&data[j + k..j + k + 4]);
                let hi = f32x4::from_slice(&data[j + k + 4..j + k + 8]);
                max_lo = max_lo.simd_max((lo - v_i).abs());
                max_hi = max_hi.simd_max((hi - v_i).abs());
            }
            b_acc += max_lo.simd_le(r_vec).select(one, zero);
            b_acc += max_hi.simd_le(r_vec).select(one, zero);

            // The length-(m + 1) distance is at least the length-m one, so a
            // match of m + 1 is also a match of m.
            let v_i = f32x4::splat(data[i + m]);
            let lo = f32x4::from_slice(&data[j + m..j + m + 4]);
            let hi = f32x4::from_slice(&data[j + m + 4..j + m + 8]);
            a_acc += max_lo
                .simd_max((lo - v_i).abs())
                .simd_le(r_vec)
                .select(one, zero);
            a_acc += max_hi
                .simd_max((hi - v_i).abs())
                .simd_le(r_vec)
                .select(one, zero);

            j += 8;
        }
        // Remainder
        for j_rem in j..end_m {
            let mut max_m = 0.0_f32;
            for k in 0..m {
                max_m = max_m.max((data[j_rem + k] - data[i + k]).abs());
            }
            if max_m <= r {
                b_count += 1;
                let max_mp1 = max_m.max((data[j_rem + m] - data[i + m]).abs());
                if max_mp1 <= r {
                    a_count += 1;
                }
            }
        }
    }
    b_count += b_acc.reduce_sum() as u32;
    a_count += a_acc.reduce_sum() as u32;

    // TSFEL drops the last length-m template from B (templates_B[:-1]); the
    // loop `0..end_m` matches that. tsfresh keeps it: see tsfresh_sample_entropy.

    // Both libraries count (i, j) and (j, i) but exclude i == j.
    (a_count * 2, b_count * 2)
}

/// tsfresh approximate entropy, `|phi(m) - phi(m + 1)|`, where
/// `phi(m) = mean_i ln(C_m[i] / (n - m + 1))` and `C_m[i]` counts the length-m
/// templates (itself included) within Chebyshev distance `r` of template `i`.
///
/// The distance is symmetric, so each diagonal `j = i + d` is scanned once and
/// credits both `i` and `j`, and the length-(m + 1) distance extends the
/// length-m one by a single element: one pass instead of two full n² passes.
/// Counts are exact, so the result is the same as the direct definition.
pub fn approx_entropy(m: usize, r: f32, data: &[f32], counts: &mut Vec<i32>) -> f32 {
    use std::simd::{Select, i32x4};
    let n = data.len();
    if n <= m + 1 {
        return 0.0;
    }
    let len_m = n - m + 1;
    let len_m1 = n - m;
    counts.clear();
    counts.resize(len_m + len_m1, 1); // every template matches itself
    let (cm, cm1) = counts.split_at_mut(len_m);
    let r_vec = f32x4::splat(r);

    let (one, zero) = (i32x4::splat(1), i32x4::splat(0));

    #[inline(always)]
    fn add_hits(c: &mut [i32], at: usize, hits: i32x4) {
        let v = i32x4::from_slice(&c[at..at + 4]) + hits;
        v.copy_to_slice(&mut c[at..at + 4]);
    }

    for d in 1..len_m {
        // i in 0..pairs pairs template i with j = i + d <= n - m; the last of
        // those has no length-(m + 1) template at j.
        let pairs = len_m - d;
        let pairs1 = pairs - 1;
        let mut i = 0;
        while i + 4 <= pairs1 {
            let mut dist = f32x4::splat(0.0);
            for k in 0..m {
                let a = f32x4::from_slice(&data[i + k..i + k + 4]);
                let b = f32x4::from_slice(&data[i + d + k..i + d + k + 4]);
                dist = dist.simd_max((a - b).abs());
            }
            let a = f32x4::from_slice(&data[i + m..i + m + 4]);
            let b = f32x4::from_slice(&data[i + d + m..i + d + m + 4]);
            let dist1 = dist.simd_max((a - b).abs());
            let hits = dist.simd_le(r_vec).select(one, zero);
            let hits1 = dist1.simd_le(r_vec).select(one, zero);
            add_hits(cm, i, hits);
            add_hits(cm, i + d, hits);
            add_hits(cm1, i, hits1);
            add_hits(cm1, i + d, hits1);
            i += 4;
        }
        for i in i..pairs {
            let mut dist = 0.0_f32;
            for k in 0..m {
                dist = dist.max((data[i + k] - data[i + d + k]).abs());
            }
            if dist <= r {
                cm[i] += 1;
                cm[i + d] += 1;
            }
            if i < pairs1 && dist.max((data[i + m] - data[i + d + m]).abs()) <= r {
                cm1[i] += 1;
                cm1[i + d] += 1;
            }
        }
    }

    let phi = |c: &[i32]| {
        let len = c.len() as f32;
        c.iter().map(|&k| (k as f32 / len).ln()).sum::<f32>() / len
    };
    (phi(cm) - phi(cm1)).abs()
}

pub fn permutation_entropy(data: &[f32], tau: u32, dimension: u32) -> f32 {
    let tau = tau as usize;
    let dim = dimension as usize;
    let n = data.len();

    // tsfresh _into_subchunks implementation logic:
    // chunk length = dimension, starts every 'tau' (which they call every_n)
    // The window index in the original array is: start + j (where j < dim)
    // start shifts by 'tau' each time.
    if n < dim {
        return f32::NAN;
    }

    let num_shifts = (n - dim) / tau + 1;
    if num_shifts == 0 {
        return f32::NAN;
    }

    let mut counts = rustc_hash::FxHashMap::default();
    let mut perm_indices = vec![0usize; dim];
    let mut inv_perm = vec![0usize; dim];

    for shift in 0..num_shifts {
        let start = shift * tau;
        for j in 0..dim {
            perm_indices[j] = j;
        }

        perm_indices.sort_by(|&a, &b| {
            let val_a = data[start + a];
            let val_b = data[start + b];

            match (val_a.is_nan(), val_b.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => val_a
                    .partial_cmp(&val_b)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });

        for (rank, &idx) in perm_indices.iter().enumerate() {
            inv_perm[idx] = rank;
        }

        if dim <= 16 {
            let mut encoded: u64 = 0;
            for (shift, &idx) in inv_perm.iter().enumerate() {
                encoded |= (idx as u64) << (shift * 4);
            }
            *counts.entry(encoded).or_insert(0usize) += 1;
        } else {
            return f32::NAN;
        }
    }

    let mut entropy = 0.0;
    let num_windows_f = num_shifts as f32;
    for &count in counts.values() {
        let p = count as f32 / num_windows_f;
        if p > 0.0 {
            // TSFresh uses natural log here despite mathematical formulation calling for log2
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Hash key for counting distinct values: -0.0 and 0.0 are the same value, as
/// in numpy's `np.unique` that tsfresh uses.
#[inline(always)]
pub fn value_key(v: f32) -> u32 {
    (v + 0.0).to_bits()
}

/// A node of the flat LZ phrase trie (`lz_complexity`): children form a
/// sibling chain, and 0 means "none" (node 0 is a placeholder).
#[derive(Clone, Copy, Default)]
pub struct LzNode {
    pub first_child: u32,
    pub next_sibling: u32,
    pub symbol: u8,
}
