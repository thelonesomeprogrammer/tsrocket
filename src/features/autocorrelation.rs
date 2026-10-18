use crate::types::Feature;
use std::simd::num::SimdFloat;

#[inline(always)]
pub fn eval_autocorrelation(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let values = context.values;
    let state = &mut *context.state;
    let n = context.n;
    let mean = context.mean;
    let _m2 = context.m2;
    let _m3 = context.m3;
    let _m4 = context.m4;
    let var = context.var;
    let _std_dev = context.std_dev;
    let _mac_sum = context.mac_sum;
    let _mc_sum = context.mc_sum;
    let _first_max_idx = context.first_max_idx;
    let _last_max_idx = context.last_max_idx;
    let _first_min_idx = context.first_min_idx;
    let _last_min_idx = context.last_min_idx;
    let _median = context.median;
    let _iqr = context.iqr;
    let _mad_sum = context.mad_sum;
    let _count_a = context.count_a;
    let _count_b = context.count_b;
    let _max_strike_a = context.max_strike_a;
    let _max_strike_b = context.max_strike_b;
    let _zc_mean = context.zc_mean;
    let _zc_std = context.zc_std;
    let _freq_centroid = context.freq_centroid;
    let _spectral_decrease = context.spectral_decrease;
    let _spectral_slope = context.spectral_slope;
    let fft_autocorr = context.fft_autocorr;
    let _fft_complex = context.fft_complex;
    let _spectrum = context.spectrum;
    let _unique_c3_lags = context.unique_c3_lags;
    let unique_tra_lags = context.unique_tra_lags;
    let _unique_paa_totals = context.unique_paa_totals;
    let _paa_boundaries = context.paa_boundaries;

    let res = match feat {
        // tsfresh autocorrelation(x, 1), computed directly from the window.
        Feature::AutocorrLag1 if var > 1e-9 && n > 1.0 => {
            let cov: f32 = values
                .windows(2)
                .map(|w| (w[0] - mean) * (w[1] - mean))
                .sum();
            cov / ((n - 1.0) * var)
        }
        Feature::AutocorrFirst1e => {
            if !fft_autocorr.is_empty() {
                let threshold = 0.36787944;
                let mut found = false;
                let mut first_lag = 0.0;
                for (l, &val) in fft_autocorr.iter().enumerate().skip(1) {
                    if val < threshold {
                        first_lag = l as f32;
                        found = true;
                        break;
                    }
                }
                if found { first_lag } else { 0.0 }
            } else {
                0.0
            }
        }
        Feature::AggAutocorrelation(func, maxlag) => {
            if fft_autocorr.is_empty() || *maxlag == 0 || var <= 1e-9 {
                0.0
            } else {
                let max_l = (*maxlag as usize).min(fft_autocorr.len().saturating_sub(1));
                if max_l == 0 {
                    0.0
                } else {
                    let mut adjusted: Vec<f32> =
                        (1..=max_l).map(|l| adjusted_acf(fft_autocorr, l)).collect();
                    let slice = &adjusted[..];
                    match func {
                        crate::types::AggFunc::Mean => {
                            let sum: f32 = slice.iter().sum();
                            sum / max_l as f32
                        }
                        crate::types::AggFunc::Var => {
                            let mean = slice.iter().sum::<f32>() / max_l as f32;
                            let var_sum: f32 = slice.iter().map(|&v| (v - mean) * (v - mean)).sum();
                            var_sum / max_l as f32 // ddof=0 for population variance
                        }
                        crate::types::AggFunc::Max => {
                            slice.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                        }
                        crate::types::AggFunc::Min => {
                            slice.iter().copied().fold(f32::INFINITY, f32::min)
                        }
                        crate::types::AggFunc::Median => super::median_in_place(&mut adjusted),
                    }
                }
            }
        }
        Feature::Autocorr(lag) if var > 1e-9 && values.len() > *lag as usize => {
            let l = *lag as usize;
            if !fft_autocorr.is_empty() && l < fft_autocorr.len() {
                adjusted_acf(fft_autocorr, l)
            } else {
                0.0
            }
        }
        Feature::TimeReversalAsymmetry(lag) if values.len() > 2 * *lag as usize => {
            let l = *lag as usize;
            let l_idx = unique_tra_lags.iter().position(|&l| l == *lag);
            if let Some(idx) = l_idx {
                let n_iters = values.len() - 2 * l;
                state.tra_sums[idx] / n_iters as f32
            } else {
                let mut sum = 0.0;
                let n_iters = values.len() - 2 * l;

                // SIMD optimization
                let mut i = 0;
                let mut sum_simd = std::simd::f32x4::splat(0.0);
                while i + 3 < n_iters {
                    let v_i = std::simd::f32x4::from_slice(&values[i..i + 4]);
                    let v_il = std::simd::f32x4::from_slice(&values[i + l..i + l + 4]);
                    let v_i2l = std::simd::f32x4::from_slice(&values[i + 2 * l..i + 2 * l + 4]);
                    sum_simd += v_i2l * v_i2l * v_il - v_il * v_i * v_i;
                    i += 4;
                }
                sum += sum_simd.reduce_sum();

                // Remainder
                for j in i..n_iters {
                    sum += values[j + 2 * l].powi(2) * values[j + l]
                        - values[j + l] * values[j].powi(2);
                }

                sum / n_iters as f32
            }
        }
        // tsfresh computes pacf up to max_lag = min(D, n / 2 - 1), where D is
        // the largest lag requested together (9 in its default settings), and
        // NaN beyond that or when max_lag < 1.
        Feature::PartialAutocorr(lag) if !pacf_lag_is_computed(values.len(), *lag) => f32::NAN,
        // pacf at lag 0 is 1 by definition, constant series included.
        Feature::PartialAutocorr(0) => 1.0,
        Feature::PartialAutocorr(lag) if var > 1e-9 && values.len() > *lag as usize => {
            let l = *lag as usize;
            if fft_autocorr.is_empty() || fft_autocorr.len() <= l {
                0.0
            } else {
                // statsmodels pacf(method="ld") runs Levinson-Durbin on the adjusted acf.
                let mut r = std::mem::take(&mut state.diff_buffer);
                r.clear();
                r.extend((0..=l).map(|k| adjusted_acf(fft_autocorr, k)));
                let stride = l + 1;
                let req_len = stride * stride;
                let mut phi = std::mem::take(&mut state.pacf_buffer);
                if phi.len() < req_len {
                    phi.resize(req_len, 0.0);
                }
                let mut error = r[0] as f64;
                let val = if error.abs() < 1e-9 {
                    0.0
                } else {
                    phi[stride + 1] = r[1] / r[0];
                    error *= 1.0 - (phi[stride + 1] * phi[stride + 1]) as f64;
                    for k in 1..l {
                        let mut sum = 0.0;
                        for i in 1..=k {
                            sum += phi[k * stride + i] * r[k + 1 - i];
                        }
                        phi[(k + 1) * stride + (k + 1)] = (r[k + 1] - sum) / error as f32;
                        for i in 1..=k {
                            phi[(k + 1) * stride + i] = phi[k * stride + i]
                                - phi[(k + 1) * stride + (k + 1)] * phi[k * stride + (k + 1 - i)];
                        }
                        error *= 1.0
                            - (phi[(k + 1) * stride + (k + 1)] * phi[(k + 1) * stride + (k + 1)])
                                as f64;
                        if error.abs() < 1e-9 {
                            break;
                        }
                    }
                    phi[l * stride + l]
                };
                state.pacf_buffer = phi;
                state.diff_buffer = r;
                val
            }
        }
        _ => return None,
    };
    Some(res)
}

/// The FFT autocorrelation divides every lag by n; tsfresh and statsmodels
/// ("adjusted") divide lag l by n - l.
fn adjusted_acf(fft_autocorr: &[f32], lag: usize) -> f32 {
    let n = fft_autocorr.len() as f32;
    fft_autocorr[lag] * n / (n - lag as f32)
}

/// Whether tsfresh computes partial_autocorrelation at `lag` over `n`
/// samples, requested alongside its default lags 0..=9: it runs pacf up to
/// max_lag = D if n > 2 D else n / 2 - 1 (D the largest lag requested), and
/// returns NaN for every lag when max_lag < 1, lag 0 included.
fn pacf_lag_is_computed(n: usize, lag: u16) -> bool {
    let demanded = (lag as usize).max(9);
    let max_lag = if n > 2 * demanded {
        demanded
    } else {
        (n / 2).saturating_sub(1)
    };
    max_lag >= 1 && lag as usize <= max_lag
}
