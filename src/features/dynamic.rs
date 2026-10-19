use crate::types::Feature;
use ndarray::{Array1, Array2};
use realfft::num_complex::Complex;
use std::f32;
use std::f64;
use std::simd::{f64x8, num::SimdFloat};

pub fn eval_dynamic(feat: &Feature, context: &mut crate::context::FeatureContext) -> Option<f32> {
    let values = context.values;
    let n = values.len();
    if n == 0 {
        return Some(f32::NAN);
    }

    let state = &mut *context.state;

    let mut res = None;

    match feat {
        Feature::ArCoefficient(k, p) => {
            let k_val = *k as usize;
            let p_val = *p as usize;

            if p_val > k_val {
                res = Some(f32::NAN);
            } else {
                let coeffs = state.ar_coeffs.entry(*k).or_insert_with(|| {
                    if n <= k_val {
                        return Vec::new();
                    }
                    let (xtx, xty) = compute_ar_matrices_full(values, k_val, &mut state.ar_values);
                    solve_ols_f64(&xtx, &xty).unwrap_or_default()
                });
                // When the fit fails tsfresh falls back to k NaNs, so asking
                // for the last coefficient (p == k) hits its IndexError -> 0.
                res = Some(match coeffs.get(p_val) {
                    Some(&c) => c,
                    None if p_val == k_val => 0.0,
                    None => f32::NAN,
                });
            }
        }
        Feature::FriedrichCoefficients(m, r_bits, coeff) => {
            let m_val = *m as usize;
            let r_val = f32::from_bits(*r_bits);
            let coeff_val = *coeff as usize;

            let coeffs = state
                .friedrich_coeffs
                .entry((*m, *r_bits))
                .or_insert_with(|| vec![]);

            if coeffs.is_empty() {
                *coeffs = compute_friedrich_coeffs(
                    values,
                    m_val,
                    r_val,
                    &mut state.friedrich_signal,
                    &mut state.friedrich_delta,
                    &mut state.friedrich_sorted,
                );
            }

            if coeff_val < coeffs.len() {
                res = Some(coeffs[coeff_val]);
            } else {
                res = Some(f32::NAN);
            }
        }
        Feature::MaxLangevinFixedPoint(m, r_bits) => {
            let m_val = *m as usize;
            let r_val = f32::from_bits(*r_bits);

            if let Some(&max_root) = state.max_langevin_fixed_point_cache.get(&(*m, *r_bits)) {
                res = Some(max_root);
            } else {
                let coeffs = state
                    .friedrich_coeffs
                    .entry((*m, *r_bits))
                    .or_insert_with(|| vec![]);

                if coeffs.is_empty() {
                    *coeffs = compute_friedrich_coeffs(
                        values,
                        m_val,
                        r_val,
                        &mut state.friedrich_signal,
                        &mut state.friedrich_delta,
                        &mut state.friedrich_sorted,
                    );
                }

                let max_root = compute_max_langevin(coeffs);
                state
                    .max_langevin_fixed_point_cache
                    .insert((*m, *r_bits), max_root);
                res = Some(max_root);
            }
        }
        _ => return None,
    }

    res
}

fn compute_ar_matrices_full(
    values: &[f32],
    k: usize,
    v64: &mut Vec<f64>,
) -> (Array2<f64>, Array1<f64>) {
    let m = k + 1;
    let mut xtx = Array2::<f64>::zeros((m, m));
    let mut xty = Array1::<f64>::zeros(m);

    v64.clear();
    v64.extend(values.iter().map(|&x| x as f64));
    let v64 = &v64[..];
    let n = values.len();

    // Use SIMD for full matrices
    for i in 0..m {
        if i == 0 {
            let mut sum = 0.0;
            let mut t = k;
            let mut sum_vec = f64x8::splat(0.0);
            while t + 8 <= n {
                let y_vec = f64x8::from_slice(&v64[t..t + 8]);
                sum_vec += y_vec;
                t += 8;
            }
            sum += sum_vec.reduce_sum();
            while t < n {
                sum += v64[t];
                t += 1;
            }
            xty[i] = sum;
        } else {
            let mut sum = 0.0;
            let mut t = k;
            let mut sum_vec = f64x8::splat(0.0);
            while t + 8 <= n {
                let x_vec = f64x8::from_slice(&v64[t - i..t - i + 8]);
                let y_vec = f64x8::from_slice(&v64[t..t + 8]);
                sum_vec += x_vec * y_vec;
                t += 8;
            }
            sum += sum_vec.reduce_sum();
            while t < n {
                sum += v64[t - i] * v64[t];
                t += 1;
            }
            xty[i] = sum;
        }

        for j in i..m {
            let mut sum = 0.0;
            if i == 0 && j == 0 {
                sum = (n - k) as f64;
            } else if i == 0 {
                let mut t = k;
                let mut sum_vec = f64x8::splat(0.0);
                while t + 8 <= n {
                    let x_vec = f64x8::from_slice(&v64[t - j..t - j + 8]);
                    sum_vec += x_vec;
                    t += 8;
                }
                sum += sum_vec.reduce_sum();
                while t < n {
                    sum += v64[t - j];
                    t += 1;
                }
            } else {
                let mut t = k;
                let mut sum_vec = f64x8::splat(0.0);
                while t + 8 <= n {
                    let xi_vec = f64x8::from_slice(&v64[t - i..t - i + 8]);
                    let xj_vec = f64x8::from_slice(&v64[t - j..t - j + 8]);
                    sum_vec += xi_vec * xj_vec;
                    t += 8;
                }
                sum += sum_vec.reduce_sum();
                while t < n {
                    sum += v64[t - i] * v64[t - j];
                    t += 1;
                }
            }
            xtx[[i, j]] = sum;
            xtx[[j, i]] = sum;
        }
    }

    (xtx, xty)
}

pub(crate) fn solve_ols_f64(xtx: &Array2<f64>, xty: &Array1<f64>) -> Option<Vec<f32>> {
    let n = xtx.nrows();
    let mut a = xtx.clone();
    let mut b = xty.clone();

    // Gaussian elimination with partial pivoting
    for i in 0..n {
        let mut max_row = i;
        for k in i + 1..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        if a[[max_row, i]].abs() < 1e-12 {
            return None;
        }

        if max_row != i {
            for j in i..n {
                let temp = a[[i, j]];
                a[[i, j]] = a[[max_row, j]];
                a[[max_row, j]] = temp;
            }
            let temp = b[i];
            b[i] = b[max_row];
            b[max_row] = temp;
        }

        for k in i + 1..n {
            let factor = a[[k, i]] / a[[i, i]];
            for j in i..n {
                a[[k, j]] -= factor * a[[i, j]];
            }
            b[k] -= factor * b[i];
        }
    }

    let mut x_res = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i + 1..n {
            sum += a[[i, j]] * x_res[j];
        }
        x_res[i] = (b[i] - sum) / a[[i, i]];
    }

    Some(x_res.iter().map(|&v| v as f32).collect())
}

fn solve_ols_polyfit(x: &Array2<f64>, y: &Array1<f64>) -> Option<Vec<f32>> {
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);
    solve_ols_f64(&xtx, &xty)
}

fn compute_ar_coeffs(values: &[f32], k: usize) -> Vec<f32> {
    let n = values.len();
    if n <= k {
        return vec![f32::NAN; k + 1];
    }

    let m = k + 1;
    let mut xtx = Array2::<f64>::zeros((m, m));
    let mut xty = Array1::<f64>::zeros(m);

    let v64: Vec<f64> = values.iter().map(|&x| x as f64).collect();

    for i in 0..m {
        if i == 0 {
            let mut sum = 0.0;
            let mut t = k;
            let mut sum_vec = f64x8::splat(0.0);
            while t + 8 <= n {
                let y_vec = f64x8::from_slice(&v64[t..t + 8]);
                sum_vec += y_vec;
                t += 8;
            }
            sum += sum_vec.reduce_sum();
            while t < n {
                sum += v64[t];
                t += 1;
            }
            xty[i] = sum;
        } else {
            let mut sum = 0.0;
            let mut t = k;
            let mut sum_vec = f64x8::splat(0.0);
            while t + 8 <= n {
                let x_vec = f64x8::from_slice(&v64[t - i..t - i + 8]);
                let y_vec = f64x8::from_slice(&v64[t..t + 8]);
                sum_vec += x_vec * y_vec;
                t += 8;
            }
            sum += sum_vec.reduce_sum();
            while t < n {
                sum += v64[t - i] * v64[t];
                t += 1;
            }
            xty[i] = sum;
        }

        for j in i..m {
            let mut sum = 0.0;
            if i == 0 && j == 0 {
                sum = (n - k) as f64;
            } else if i == 0 {
                let mut t = k;
                let mut sum_vec = f64x8::splat(0.0);
                while t + 8 <= n {
                    let x_vec = f64x8::from_slice(&v64[t - j..t - j + 8]);
                    sum_vec += x_vec;
                    t += 8;
                }
                sum += sum_vec.reduce_sum();
                while t < n {
                    sum += v64[t - j];
                    t += 1;
                }
            } else {
                let mut t = k;
                let mut sum_vec = f64x8::splat(0.0);
                while t + 8 <= n {
                    let xi_vec = f64x8::from_slice(&v64[t - i..t - i + 8]);
                    let xj_vec = f64x8::from_slice(&v64[t - j..t - j + 8]);
                    sum_vec += xi_vec * xj_vec;
                    t += 8;
                }
                sum += sum_vec.reduce_sum();
                while t < n {
                    sum += v64[t - i] * v64[t - j];
                    t += 1;
                }
            }
            xtx[[i, j]] = sum;
            xtx[[j, i]] = sum;
        }
    }

    if let Some(coeffs) = solve_ols_f64(&xtx, &xty) {
        coeffs
    } else {
        vec![f32::NAN; k + 1]
    }
}

fn compute_friedrich_coeffs(
    values: &[f32],
    m: usize,
    r: f32,
    signal: &mut Vec<f64>,
    delta: &mut Vec<f64>,
    sorted: &mut Vec<f64>,
) -> Vec<f32> {
    let n = values.len();
    if n < 2 {
        return vec![f32::NAN; m + 1];
    }

    signal.clear();
    delta.clear();

    for i in 0..n - 1 {
        signal.push(values[i] as f64);
        delta.push((values[i + 1] - values[i]) as f64);
    }

    // tsfresh bins with pd.qcut(signal, r): edges are linear-interpolated
    // quantiles, bins are right-closed (the first also includes its left edge),
    // and duplicate edges raise, which tsfresh turns into NaN coefficients.
    let r_int = r as usize;
    if r_int < 1 {
        return vec![f32::NAN; m + 1];
    }
    let sorted = &mut *sorted;
    sorted.clear();
    sorted.extend_from_slice(signal);
    sorted.sort_by(f64::total_cmp);
    let last = (sorted.len() - 1) as f64;
    let edges: Vec<f64> = (0..=r_int)
        .map(|q| {
            let pos = q as f64 / r_int as f64 * last;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(sorted.len() - 1);
            sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
        })
        .collect();
    if edges.windows(2).any(|w| w[0] == w[1]) {
        return vec![f32::NAN; m + 1];
    }

    let mut sum_x = vec![0.0f64; r_int];
    let mut sum_y = vec![0.0f64; r_int];
    let mut count = vec![0usize; r_int];
    for (&sx, &dy) in signal.iter().zip(delta.iter()) {
        // First bin whose right edge is >= sx.
        let bin = edges[1..].partition_point(|&e| e < sx).min(r_int - 1);
        sum_x[bin] += sx;
        sum_y[bin] += dy;
        count[bin] += 1;
    }
    let mut x_means = Vec::with_capacity(r_int);
    let mut y_means = Vec::with_capacity(r_int);
    for q in 0..r_int {
        if count[q] > 0 {
            x_means.push(sum_x[q] / count[q] as f64);
            y_means.push(sum_y[q] / count[q] as f64);
        }
    }

    if x_means.is_empty() {
        return vec![f32::NAN; m + 1];
    }

    let n_pts = x_means.len();
    let mut x_mat_vec = Vec::with_capacity(n_pts * (m + 1));

    for i in 0..n_pts {
        let x_val = x_means[i];
        for j in 0..=m {
            x_mat_vec.push(x_val.powi((m - j) as i32));
        }
    }

    let x = if let Ok(mat) = Array2::from_shape_vec((n_pts, m + 1), x_mat_vec) {
        mat
    } else {
        return vec![f32::NAN; m + 1];
    };
    let y = Array1::from(y_means);

    if let Some(coeffs) = solve_ols_polyfit(&x, &y) {
        coeffs
    } else {
        vec![f32::NAN; m + 1]
    }
}

/// tsfresh: `np.max(np.real(np.roots(coeffs)))`, the largest real part over
/// *all* roots of the Friedrich polynomial (complex ones included). Roots by
/// Durand-Kerner in f64, started on a circle of the Cauchy bound radius at
/// the non-symmetric powers of 0.4 + 0.9i (a symmetric start can stall).
fn compute_max_langevin(coeffs: &[f32]) -> f32 {
    if coeffs.iter().any(|c| !c.is_finite()) {
        return f32::NAN;
    }
    // np.roots strips leading zeros; no roots left means tsfresh's NaN.
    let lead = coeffs
        .iter()
        .position(|&c| c != 0.0)
        .unwrap_or(coeffs.len());
    let c: Vec<f64> = coeffs[lead..].iter().map(|&c| c as f64).collect();
    if c.len() < 2 {
        return f32::NAN;
    }
    let m = c.len() - 1;
    let monic: Vec<f64> = c.iter().map(|&v| v / c[0]).collect();
    let bound = 1.0 + monic[1..].iter().fold(0.0f64, |a, &v| a.max(v.abs()));

    let seed = Complex::new(0.4f64, 0.9);
    let mut roots: Vec<Complex<f64>> = Vec::with_capacity(m);
    let mut z = Complex::new(1.0f64, 0.0);
    for _ in 0..m {
        z = z * seed;
        roots.push(z * (bound / z.norm()));
    }

    for _ in 0..500 {
        let mut max_change = 0.0f64;
        for i in 0..m {
            let mut p = Complex::new(monic[0], 0.0);
            for &a in &monic[1..] {
                p = p * roots[i] + a;
            }
            let mut prod = Complex::new(1.0f64, 0.0);
            for j in 0..m {
                if i != j {
                    prod = prod * (roots[i] - roots[j]);
                }
            }
            if prod.norm() == 0.0 {
                continue;
            }
            let change = p / prod;
            roots[i] -= change;
            max_change = max_change.max(change.norm() / roots[i].norm().max(1.0));
        }
        if max_change < 1e-14 {
            break;
        }
    }

    roots.iter().map(|r| r.re).fold(f64::NEG_INFINITY, f64::max) as f32
}
