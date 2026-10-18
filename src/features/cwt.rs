//! Ports of the wavelet routines the reference libraries call, so tsrocket
//! matches them: `pywt.cwt(x, scales, "mexh")` (TSFEL wavelet features,
//! tsfresh cwt_coefficients) and `scipy.signal.find_peaks_cwt` with tsfresh's
//! ricker wavelet (tsfresh number_cwt_peaks). Kernels are built in f64; the mexh convolution runs in f32, the ricker one
//! (feeding a discrete peak count) in f64.

use std::sync::OnceLock;

/// pywt.integrate_wavelet("mexh", precision=12) (pywt.cwt's default): cumulative
/// integral of the Mexican hat sampled at linspace(-8, 8, 4096), and the step.
fn mexh_integral() -> &'static (Vec<f64>, f64) {
    static INT_PSI: OnceLock<(Vec<f64>, f64)> = OnceLock::new();
    INT_PSI.get_or_init(|| {
        let len = 4096;
        let step = 16.0 / (len - 1) as f64;
        let c = 2.0 / (3.0f64.sqrt() * std::f64::consts::PI.powf(0.25));
        let mut acc = 0.0;
        let int_psi = (0..len)
            .map(|i| {
                let x = -8.0 + i as f64 * step;
                acc += c * (1.0 - x * x) * (-x * x / 2.0).exp();
                acc * step
            })
            .collect();
        (int_psi, step)
    })
}

/// pywt.cwt's mexh row for `scale` as one convolution: pywt takes
/// `-sqrt(scale) * diff(conv(x, kernel))`, which equals a convolution with the
/// differenced kernel, so each coefficient is a single dot product (no
/// `next - prev` cancellation in f32). Built in f64 from pywt's table, stored
/// reversed and pre-scaled; `start` is pywt's centre-trim offset.
struct MexhKernel {
    rev_diff: Vec<f32>,
    start: usize,
}

fn mexh_kernel(scale: f64) -> Option<MexhKernel> {
    let (int_psi, step) = mexh_integral();
    // j = floor(arange(scale * 16 + 1) / (scale * step)), clipped to int_psi.
    let kernel: Vec<f64> = (0..=(scale * 16.0) as usize)
        .map(|i| (i as f64 / (scale * step)) as usize)
        .filter(|&j| j < int_psi.len())
        .map(|j| int_psi[j])
        .rev()
        .collect();
    if kernel.is_empty() {
        return None;
    }
    let k = kernel.len();
    let at = |j: usize| if j < k { kernel[j] } else { 0.0 };
    let sqrt_s = scale.sqrt();
    // rev_diff[q] = -sqrt(s) * (kernel[k - q] - kernel[k - q - 1]), q = 0..=k
    let rev_diff = (0..=k)
        .map(|q| {
            let j = k - q;
            let prev = if j == 0 { 0.0 } else { kernel[j - 1] };
            (-sqrt_s * (at(j) - prev)) as f32
        })
        .collect();
    // Full convolution has n + k - 1 samples; pywt keeps n of them from floor((k - 2) / 2).
    let start = ((k as f64 - 2.0) / 2.0).floor() as usize;
    Some(MexhKernel { rev_diff, start })
}

/// Kernels for the integer scales the features use, built once.
fn cached_mexh_kernel(scale: u16) -> Option<&'static MexhKernel> {
    const CACHED: usize = 64;
    static KERNELS: OnceLock<Vec<Option<MexhKernel>>> = OnceLock::new();
    KERNELS
        .get_or_init(|| (0..CACHED).map(|s| mexh_kernel(s as f64)).collect())
        .get(scale as usize)?
        .as_ref()
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    use std::simd::{f32x4, num::SimdFloat};
    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);
    let mut acc0 = f32x4::splat(0.0);
    let mut acc1 = f32x4::splat(0.0);
    let mut i = 0;
    while i + 8 <= len {
        acc0 += f32x4::from_slice(&a[i..i + 4]) * f32x4::from_slice(&b[i..i + 4]);
        acc1 += f32x4::from_slice(&a[i + 4..i + 8]) * f32x4::from_slice(&b[i + 4..i + 8]);
        i += 8;
    }
    let mut sum = (acc0 + acc1).reduce_sum();
    for j in i..len {
        sum += a[j] * b[j];
    }
    sum
}

fn mexh_coeff(values: &[f32], kern: &MexhKernel, i: usize) -> f32 {
    let k = kern.rev_diff.len() - 1;
    let p = kern.start + i + 1;
    let lo = p.saturating_sub(k);
    let hi = p.min(values.len() - 1);
    if lo > hi {
        return 0.0;
    }
    dot_f32(&values[lo..=hi], &kern.rev_diff[k + lo - p..])
}

fn with_mexh_kernel<R>(scale: u16, f: impl FnOnce(&MexhKernel) -> R) -> Option<R> {
    match cached_mexh_kernel(scale) {
        Some(kern) => Some(f(kern)),
        None => mexh_kernel(scale as f64).map(|kern| f(&kern)),
    }
}

/// One row of pywt.cwt(values, [scale], "mexh") into `out` (`values.len()` coefficients).
pub fn mexh_cwt_into(values: &[f32], scale: u16, out: &mut Vec<f32>) {
    out.clear();
    let n = values.len();
    let filled = n > 0
        && with_mexh_kernel(scale, |kern| {
            out.extend((0..n).map(|i| mexh_coeff(values, kern, i)))
        })
        .is_some();
    if !filled {
        out.clear();
        out.resize(n, 0.0);
    }
}

/// pywt.cwt(values, [scale], "mexh")[0, i], without computing the rest of the row.
pub fn mexh_cwt_coeff(values: &[f32], scale: u16, i: usize) -> f32 {
    if i >= values.len() {
        return 0.0;
    }
    with_mexh_kernel(scale, |kern| mexh_coeff(values, kern, i)).unwrap_or(0.0)
}

fn ricker(points: usize, a: f64) -> Vec<f64> {
    let amp = 2.0 / ((3.0 * a).sqrt() * std::f64::consts::PI.powf(0.25));
    let centre = (points as f64 - 1.0) / 2.0;
    (0..points)
        .map(|i| {
            let x2 = (i as f64 - centre).powi(2);
            amp * (1.0 - x2 / (a * a)) * (-x2 / (2.0 * a * a)).exp()
        })
        .collect()
}

/// scipy's old `signal.cwt` row: convolve(data, ricker(min(10w, n), w)[::-1], "same").
///
/// Scalar f64, summed in order: the peak count compares neighbouring
/// coefficients, and integer-valued series produce exact ties that only the
/// reference's precision and summation order reproduce (f32 or a SIMD
/// reduction disagrees with tsfresh on ~1% of such series).
fn ricker_cwt(values: &[f32], width: f64) -> Vec<f64> {
    let n = values.len();
    let points = ((10.0 * width) as usize).min(n);
    let mut wav = ricker(points, width);
    wav.reverse();
    let offset = (points - 1) / 2;
    (0..n)
        .map(|i| {
            let k = i + offset; // index into the full convolution
            let lo = k.saturating_sub(points - 1);
            let hi = k.min(n - 1);
            (lo..=hi).map(|m| values[m] as f64 * wav[k - m]).sum()
        })
        .collect()
}

/// `len(scipy.signal.find_peaks_cwt(values, widths=1..=max_width, wavelet=_ricker))`
/// with scipy's defaults: max_distances = widths / 4, gap_thresh = widths[0],
/// min_length = ceil(rows / 4), window_size = ceil(n / 20), min_snr = 1,
/// noise_perc = 10.
pub fn number_cwt_peaks(values: &[f32], max_width: usize) -> usize {
    let n = values.len();
    if n == 0 || max_width == 0 {
        return 0;
    }
    let widths: Vec<f64> = (1..=max_width).map(|w| w as f64).collect();
    let cwt: Vec<Vec<f64>> = widths.iter().map(|&w| ricker_cwt(values, w)).collect();
    let rows = cwt.len();

    // Relative maxima along each row; edges compare with themselves ("clip"), so never maxima.
    let is_max = |row: &[f64], i: usize| {
        let left = row[i.saturating_sub(1)];
        let right = row[(i + 1).min(n - 1)];
        row[i] > left && row[i] > right
    };
    let max_cols: Vec<Vec<usize>> = cwt
        .iter()
        .map(|row| (0..n).filter(|&i| is_max(row, i)).collect())
        .collect();

    // _identify_ridge_lines: follow maxima from the widest row down.
    struct Line {
        rows: Vec<usize>,
        cols: Vec<usize>,
        gap: usize,
    }
    let Some(start_row) = (0..rows).rev().find(|&r| !max_cols[r].is_empty()) else {
        return 0;
    };
    let gap_thresh = widths[0].ceil() as usize;
    let mut lines: Vec<Line> = max_cols[start_row]
        .iter()
        .map(|&c| Line {
            rows: vec![start_row],
            cols: vec![c],
            gap: 0,
        })
        .collect();
    let mut finished: Vec<Line> = Vec::new();
    for row in (0..start_row).rev() {
        for line in &mut lines {
            line.gap += 1;
        }
        let prev_cols: Vec<usize> = lines.iter().map(|l| *l.cols.last().unwrap_or(&0)).collect();
        let max_distance = widths[row] / 4.0;
        for &col in &max_cols[row] {
            // First closest previous ridge, as np.argmin.
            let closest = prev_cols
                .iter()
                .enumerate()
                .min_by_key(|&(_, &pc)| pc.abs_diff(col))
                .filter(|&(_, &pc)| pc.abs_diff(col) as f64 <= max_distance)
                .map(|(i, _)| i);
            match closest {
                Some(i) => {
                    lines[i].rows.push(row);
                    lines[i].cols.push(col);
                    lines[i].gap = 0;
                }
                None => lines.push(Line {
                    rows: vec![row],
                    cols: vec![col],
                    gap: 0,
                }),
            }
        }
        for i in (0..lines.len()).rev() {
            if lines[i].gap > gap_thresh {
                finished.push(lines.remove(i));
            }
        }
    }
    finished.extend(lines);

    // _filter_ridge_lines: minimum length and SNR against the 10th percentile
    // of the narrowest row in a window around the ridge's start.
    let min_length = (rows as f64 / 4.0).ceil() as usize;
    let window = (n as f64 / 20.0).ceil() as usize;
    let (half, odd) = (window / 2, window % 2);
    let row0 = &cwt[0];
    let noise_at = |col: usize| {
        let lo = col.saturating_sub(half);
        let hi = (col + half + odd).min(n);
        let mut w: Vec<f64> = row0[lo..hi].to_vec();
        w.sort_by(f64::total_cmp);
        let pos = 0.10 * (w.len() - 1) as f64;
        let (i, frac) = (pos.floor() as usize, pos.fract());
        w[i] + (w[(i + 1).min(w.len() - 1)] - w[i]) * frac
    };
    finished
        .iter()
        .filter(|line| {
            if line.rows.len() < min_length {
                return false;
            }
            // The ridge point on the lowest row (scipy sorts each line by row).
            let k = (0..line.rows.len())
                .min_by_key(|&k| line.rows[k])
                .unwrap_or(0);
            let (r, c) = (line.rows[k], line.cols[k]);
            // scipy drops a line only when snr < min_snr, so a NaN snr keeps it.
            !((cwt[r][c] / noise_at(c)).abs() < 1.0)
        })
        .count()
}
