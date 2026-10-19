use crate::types::{AdfAttr, Feature};
use statrs::distribution::{ContinuousCDF, Normal};
use std::f32;

// MacKinnon (1994) critical values (regression "c")
const MAC_MAX: f64 = 2.74;
const MAC_MIN: f64 = -18.83;
const MAC_STAR: f64 = -1.61;
const TAU_SMALLPS: [f64; 3] = [2.1659, 1.4412, 0.038269];
const TAU_LARGEPS: [f64; 4] = [1.7339, 0.93202, -0.12745, -0.010368];

fn mackinnonp(teststat: f64) -> f64 {
    if teststat > MAC_MAX {
        return 1.0;
    } else if teststat < MAC_MIN {
        return 0.0;
    }
    let mut val = 0.0;
    if teststat <= MAC_STAR {
        for (_, &c) in TAU_SMALLPS.iter().rev().enumerate() {
            val = val * teststat + c;
        }
    } else {
        for (_, &c) in TAU_LARGEPS.iter().rev().enumerate() {
            val = val * teststat + c;
        }
    }
    let norm = match Normal::new(0.0, 1.0) {
        Ok(n) => n,
        Err(_) => return f64::NAN,
    };
    norm.cdf(val)
}

pub fn eval_stationarity(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let values = context.values;
    let n = values.len();

    let state = &mut *context.state;

    match feat {
        Feature::AugmentedDickeyFuller(attr) => {
            if state.adf_test_stat.is_nan() {
                if n < 3 {
                    return Some(f32::NAN);
                }

                // statsmodels: maxlag = min(n // 2 - ntrend - 1, ceil(12 * (n / 100)^(1/4))),
                // ntrend = 1 for regression="c"; it raises (tsfresh: NaN) below 0.
                let Some(cap) = (n / 2).checked_sub(2) else {
                    return Some(f32::NAN);
                };
                let maxlag = ((12.0 * ((n as f64) / 100.0).powf(0.25)).ceil() as usize).min(cap);

                let mut xdiff = std::mem::take(&mut state.adf_xdiff);
                xdiff.clear();
                xdiff.extend((1..n).map(|i| (values[i] - values[i - 1]) as f64));
                let mut cols = std::mem::take(&mut state.adf_design);
                let scratch = state.adf_scratch.get_or_insert_default();
                let result = adf(values, &xdiff, maxlag, &mut cols, scratch);
                state.adf_xdiff = xdiff;
                state.adf_design = cols;
                let Some((best_t_stat, best_lag)) = result else {
                    return Some(f32::NAN);
                };

                state.adf_test_stat = best_t_stat;
                state.adf_used_lag = best_lag as f32;
                if !best_t_stat.is_nan() {
                    state.adf_p_value = mackinnonp(best_t_stat as f64) as f32;
                }
            }

            match attr {
                AdfAttr::TestStat => Some(state.adf_test_stat),
                AdfAttr::PValue => Some(state.adf_p_value),
                AdfAttr::UsedLag => Some(state.adf_used_lag),
            }
        }
        _ => None,
    }
}

/// ADF regression `Δx_t ~ x_{t-1} + Δx_{t-1} + ... + Δx_{t-lags} + 1`, rows
/// `t = first..xdiff.len()`, laid out column-major in `cols` (constant last).
fn adf_design(values: &[f32], xdiff: &[f64], lags: usize, first: usize, cols: &mut Vec<f64>) {
    let rows = xdiff.len() - first;
    cols.clear();
    cols.extend((first..xdiff.len()).map(|t| values[t] as f64));
    for i in 0..lags {
        cols.extend((first..xdiff.len()).map(|t| xdiff[t - 1 - i]));
    }
    cols.extend(std::iter::repeat_n(1.0, rows));
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solves `a x = b` (`a` is k x k row-major) in place by Gaussian elimination
/// with partial pivoting; the solution is left in `b`. False if singular.
fn solve_in_place(a: &mut [f64], b: &mut [f64], k: usize) -> bool {
    for i in 0..k {
        let mut max_row = i;
        for row in i + 1..k {
            if a[row * k + i].abs() > a[max_row * k + i].abs() {
                max_row = row;
            }
        }
        if a[max_row * k + i].abs() < 1e-12 {
            return false;
        }
        if max_row != i {
            for j in 0..k {
                a.swap(i * k + j, max_row * k + j);
            }
            b.swap(i, max_row);
        }
        for row in i + 1..k {
            let factor = a[row * k + i] / a[i * k + i];
            for j in i..k {
                a[row * k + j] -= factor * a[i * k + j];
            }
            b[row] -= factor * b[i];
        }
    }
    for i in (0..k).rev() {
        let sum: f64 = (i + 1..k).map(|j| a[i * k + j] * b[j]).sum();
        b[i] = (b[i] - sum) / a[i * k + i];
    }
    true
}

/// Reusable buffers for `adf`, kept on `ColumnState` between windows.
#[derive(Clone, Default)]
pub struct AdfScratch {
    gram: Vec<f64>,
    xty: Vec<f64>,
    idx: Vec<usize>,
    ols: OlsScratch,
    e0: Vec<f64>,
}

#[derive(Clone, Default)]
struct OlsScratch {
    /// The reduced Gram matrix, solved in place.
    a: Vec<f64>,
    /// The coefficients of the last successful fit.
    coeffs: Vec<f64>,
    resid: Vec<f64>,
}

/// OLS of `y` on the columns `idx` of the column-major design `cols`, given
/// the full Gram matrix `gram` (`kk` x `kk`) and `xty`. Returns the residual
/// sum of squares, computed from the residuals, and leaves the coefficients
/// in `s.coeffs`.
#[allow(clippy::too_many_arguments)]
fn ols_subset(
    cols: &[f64],
    rows: usize,
    y: &[f64],
    gram: &[f64],
    xty: &[f64],
    kk: usize,
    idx: &[usize],
    s: &mut OlsScratch,
) -> Option<f64> {
    let OlsScratch { a, coeffs, resid } = s;
    let k = idx.len();
    a.clear();
    a.extend(
        idx.iter()
            .flat_map(|&r| idx.iter().map(move |&c| gram[r * kk + c])),
    );
    coeffs.clear();
    coeffs.extend(idx.iter().map(|&r| xty[r]));
    if !solve_in_place(a, coeffs, k) {
        return None;
    }
    resid.clear();
    resid.extend_from_slice(y);
    for (&c, &b) in idx.iter().zip(coeffs.iter()) {
        for (r, &x) in resid.iter_mut().zip(&cols[c * rows..(c + 1) * rows]) {
            *r -= x * b;
        }
    }
    Some(dot(resid, resid))
}

fn gram_matrix(
    cols: &[f64],
    rows: usize,
    y: &[f64],
    kk: usize,
    gram: &mut Vec<f64>,
    xty: &mut Vec<f64>,
) {
    let col = |c: usize| &cols[c * rows..(c + 1) * rows];
    gram.clear();
    gram.resize(kk * kk, 0.0);
    for i in 0..kk {
        for j in i..kk {
            let v = dot(col(i), col(j));
            gram[i * kk + j] = v;
            gram[j * kk + i] = v;
        }
    }
    xty.clear();
    xty.extend((0..kk).map(|i| dot(col(i), y)));
}

/// statsmodels `adfuller(x, regression="c", autolag="AIC")`: picks the lag
/// by AIC over a common sample (rows from `maxlag` on), then refits that lag
/// on all available rows. Every lag's regressors are a subset of `maxlag`'s,
/// so one Gram matrix serves the whole search. Returns (t-stat, lag).
fn adf(
    values: &[f32],
    xdiff: &[f64],
    maxlag: usize,
    cols: &mut Vec<f64>,
    s: &mut AdfScratch,
) -> Option<(f32, usize)> {
    let rows = xdiff.len().checked_sub(maxlag)?;
    if rows <= 2 {
        return None;
    }

    adf_design(values, xdiff, maxlag, maxlag, cols);
    let kk = maxlag + 2;
    let y = &xdiff[maxlag..];
    gram_matrix(cols, rows, y, kk, &mut s.gram, &mut s.xty);

    let mut best_aic = f64::INFINITY;
    let mut best_lag = 0;
    let n_obs = rows as f64;
    // statsmodels picks min((aic, lag)): ties go to the smallest lag, so scan
    // upwards and only replace on a strictly smaller AIC.
    for lag in 0..=maxlag {
        s.idx.clear();
        s.idx.extend(0..=lag);
        s.idx.push(kk - 1);
        let fit = ols_subset(cols, rows, y, &s.gram, &s.xty, kk, &s.idx, &mut s.ols);
        if let Some(rss) = fit {
            let llf = -n_obs / 2.0 * ((2.0 * std::f64::consts::PI).ln() + (rss / n_obs).ln() + 1.0);
            let aic = -2.0 * llf + 2.0 * (s.idx.len() as f64);
            if aic < best_aic || best_aic.is_infinite() {
                best_aic = aic;
                best_lag = lag;
            }
        }
    }

    // Now re-run with best_lag using full available data
    let rows = xdiff.len() - best_lag;
    let mut t_stat = f32::NAN;
    if rows > best_lag + 2 {
        adf_design(values, xdiff, best_lag, best_lag, cols);
        let kk = best_lag + 2;
        let y = &xdiff[best_lag..];
        gram_matrix(cols, rows, y, kk, &mut s.gram, &mut s.xty);
        s.idx.clear();
        s.idx.extend(0..kk);
        let fit = ols_subset(cols, rows, y, &s.gram, &s.xty, kk, &s.idx, &mut s.ols);
        if let Some(rss) = fit {
            // (X'X)^-1 [0, 0]
            let a = &mut s.ols.a;
            a.clear();
            a.extend_from_slice(&s.gram);
            s.e0.clear();
            s.e0.resize(kk, 0.0);
            s.e0[0] = 1.0;
            if solve_in_place(a, &mut s.e0, kk) {
                let sigma2 = rss / (rows as f64 - kk as f64);
                let var_coeff0 = sigma2 * s.e0[0];
                if var_coeff0 > 0.0 {
                    t_stat = (s.ols.coeffs[0] / var_coeff0.sqrt()) as f32;
                }
            }
        }
    }
    Some((t_stat, best_lag))
}
