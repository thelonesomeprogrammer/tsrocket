use crate::types::Feature;

#[inline(always)]
pub fn eval_changes(feat: &Feature, context: &mut crate::context::FeatureContext) -> Option<f32> {
    let values = context.values;
    let state = &mut *context.state;
    let n = context.n;
    let mean = context.mean;
    let _m2 = context.m2;
    let _m3 = context.m3;
    let _m4 = context.m4;
    let _var = context.var;
    let _std_dev = context.std_dev;
    let mac_sum = context.mac_sum;
    let mc_sum = context.mc_sum;
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
    let _fft_autocorr = context.fft_autocorr;
    let _fft_complex = context.fft_complex;
    let _spectrum = context.spectrum;
    let _unique_c3_lags = context.unique_c3_lags;
    let _unique_paa_totals = context.unique_paa_totals;
    let _paa_boundaries = context.paa_boundaries;

    let res = match feat {
        Feature::MeanAbsChange => {
            if n > 1.0 {
                (mac_sum / (n - 1.0)) as f32
            } else {
                0.0
            }
        }
        Feature::MeanChange => {
            if n > 1.0 {
                (mc_sum / (n - 1.0)) as f32
            } else {
                0.0
            }
        }
        // tsfresh: (x[-1] - x[-2] - x[1] + x[0]) / (2 * (n - 2))
        Feature::MeanSecondDerivativeCentral => {
            let len = values.len();
            if len > 2 {
                (values[len - 1] - values[len - 2] - values[1] + values[0])
                    / (2.0 * (len - 2) as f32)
            } else {
                f32::NAN
            }
        }
        Feature::MedianDiff | Feature::MedianAbsDiff => {
            let mut diffs: Vec<f32> = std::mem::take(&mut state.diff_buffer);
            diffs.clear();

            if values.len() > 1 {
                let is_abs = matches!(feat, Feature::MedianAbsDiff);
                diffs.reserve(values.len() - 1);
                for i in 0..values.len() - 1 {
                    let mut d = values[i + 1] - values[i];
                    if is_abs {
                        d = d.abs();
                    }
                    diffs.push(d);
                }

                let n_len = diffs.len();
                let res = if n_len == 0 {
                    f32::NAN
                } else if n_len % 2 == 1 {
                    *diffs
                        .select_nth_unstable_by(n_len / 2, |a: &f32, b: &f32| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .1
                } else {
                    let mid = n_len / 2;
                    let m1 = *diffs
                        .select_nth_unstable_by(mid, |a: &f32, b: &f32| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .1;
                    let m2 = *diffs[..mid]
                        .iter()
                        .max_by(|a: &&f32, b: &&f32| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or(&0.0);
                    (m1 + m2) / 2.0
                };
                state.diff_buffer = diffs;
                res
            } else {
                state.diff_buffer = diffs;
                0.0
            }
        }

        Feature::CidCe => state.sum_sq_diff.sqrt() as f32,
        // tsfresh cid_ce(normalize=True): cid_ce of the z-normalised series,
        // 0 for a constant one. min == max is exact in every engine; the
        // sliding engine's running variance is not.
        Feature::CidCeNormalized if state.min_value < state.max_value && context.var > 0.0 => {
            state.sum_sq_diff.max(0.0).sqrt() / context.var.sqrt()
        }
        Feature::CidCeNormalized => 0.0,
        Feature::Slope => {
            let mean_i = (n - 1.0) * 0.5;
            let s_xx = (n * (n * n - 1.0)) / 12.0;
            let s_xy = state.sum_ix - n * mean_i * mean;
            if s_xx.abs() > 1e-9 {
                (s_xy / s_xx) as f32
            } else {
                0.0
            }
        }
        Feature::Intercept => {
            let mean_i = (n - 1.0) * 0.5;
            let s_xx = (n * (n * n - 1.0)) / 12.0;
            let s_xy = state.sum_ix - n * mean_i * mean;
            let slope = if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 };
            (mean - slope * mean_i) as f32
        }
        Feature::LinearTrend(attr) | Feature::LinearTrendTimewise(attr, _) => {
            // Timewise regresses on t = i * period hours instead of i, which
            // only rescales slope and stderr; the rest is unit-free.
            let per_x = match feat {
                Feature::LinearTrendTimewise(_, period_bits) => {
                    3600.0 / f32::from_bits(*period_bits)
                }
                _ => 1.0,
            };
            let mean_i = (n - 1.0) * 0.5;
            let s_xx = (n * (n * n - 1.0)) / 12.0;
            let s_xy = state.sum_ix - n * mean_i * mean;
            let slope = if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 };
            let intercept = mean - slope * mean_i;

            match attr {
                crate::types::AggAttr::Slope => (slope * per_x) as f32,
                crate::types::AggAttr::Intercept => intercept as f32,
                crate::types::AggAttr::Stderr
                | crate::types::AggAttr::RValue
                | crate::types::AggAttr::PValue => {
                    if n > 2.0 && s_xx.abs() > 1e-9 {
                        // ss_tot is variance * n
                        // For accurate PValue we need to use m2 (sum of squared diffs from mean)
                        // var is already calculated as m2 / (n-1) or similar.
                        let ss_tot = context.m2;
                        // ss_res can be calculated as ss_tot - slope^2 * s_xx
                        let ss_res = ss_tot - slope * slope * s_xx;
                        // Need to clamp ss_res to 0 to avoid negative float issues
                        let ss_res = if ss_res < 0.0 { 0.0 } else { ss_res };

                        if matches!(attr, crate::types::AggAttr::Stderr) {
                            ((ss_res / (n - 2.0) / s_xx).sqrt() * per_x) as f32
                        } else if matches!(attr, crate::types::AggAttr::RValue) {
                            if ss_tot > 1e-9 {
                                (1.0 - ss_res / ss_tot).sqrt() as f32 * slope.signum() as f32
                            } else {
                                0.0
                            }
                        } else {
                            let t = slope / (ss_res / (n - 2.0) / s_xx).sqrt();
                            two_sided_t_pvalue(t as f64, n as f64 - 2.0) as f32
                        }
                    } else {
                        0.0
                    }
                }
            }
        }
        Feature::AbsSumChange => mac_sum as f32,
        // TSFEL: trapezoid area of |x| with t = i / fs.
        Feature::Auc => {
            let area: f32 = values.windows(2).map(|w| (w[0] + w[1]).abs()).sum();
            0.5 * area / state.fs
        }
        Feature::AggLinearTrend(attr, chunk_len, func) => {
            let cl = *chunk_len as usize;
            let mut agg_series = std::mem::take(&mut state.agg_linear_trend_buffer);
            agg_series.clear();
            // Only used by AggFunc::Median.
            let mut chunk_buf = std::mem::take(&mut state.sort_buffer);
            // tsfresh aggregates numpy chunks over ceil(n / chunk_len) chunks,
            // so the last chunk may be partial and var is ddof=0.
            for chunk in values.chunks(cl) {
                let len = chunk.len() as f32;
                let mean = chunk.iter().sum::<f32>() / len;
                let val = match func {
                    crate::types::AggFunc::Max => {
                        chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                    }
                    crate::types::AggFunc::Min => {
                        chunk.iter().copied().fold(f32::INFINITY, f32::min)
                    }
                    crate::types::AggFunc::Mean => mean,
                    crate::types::AggFunc::Var => {
                        chunk.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / len
                    }
                    crate::types::AggFunc::Median => {
                        chunk_buf.clear();
                        chunk_buf.extend_from_slice(chunk);
                        super::median_in_place(&mut chunk_buf)
                    }
                };
                agg_series.push(val);
            }
            state.sort_buffer = chunk_buf;
            let m_n = agg_series.len() as f32;
            if m_n < 2.0 {
                state.agg_linear_trend_buffer = agg_series;
                f32::NAN // tsfresh: linregress on fewer than 2 chunks
            } else {
                let m_sum_x: f32 = (0..agg_series.len()).map(|i| i as f32).sum();
                let m_sum_y: f32 = agg_series.iter().sum();
                let m_sum_xx: f32 = (0..agg_series.len()).map(|i| (i as f32).powi(2)).sum();
                let m_sum_xy: f32 = agg_series
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| i as f32 * v)
                    .sum();
                let s_xx = m_sum_xx - (m_sum_x * m_sum_x) / m_n;
                let s_xy = m_sum_xy - (m_sum_x * m_sum_y) / m_n;
                let slope = if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 };
                let intercept = (m_sum_y - slope * m_sum_x) / m_n;
                let res = match attr {
                    crate::types::AggAttr::Slope => slope,
                    crate::types::AggAttr::Intercept => intercept,
                    crate::types::AggAttr::Stderr | crate::types::AggAttr::RValue => {
                        let mut ss_res = 0.0;
                        let mut ss_tot = 0.0;
                        let m_y = m_sum_y / m_n;
                        for (i, &y) in agg_series.iter().enumerate() {
                            let y_hat = intercept + slope * i as f32;
                            ss_res += (y - y_hat).powi(2);
                            ss_tot += (y - m_y).powi(2);
                        }
                        if matches!(attr, crate::types::AggAttr::Stderr) {
                            if m_n > 2.0 && s_xx.abs() > 1e-9 {
                                (ss_res / (m_n - 2.0) / s_xx).sqrt()
                            } else {
                                0.0
                            }
                        } else {
                            if ss_tot > 1e-9 {
                                (1.0 - ss_res / ss_tot).sqrt() * slope.signum()
                            } else {
                                0.0
                            }
                        }
                    }
                    crate::types::AggAttr::PValue => {
                        let m_y = m_sum_y / m_n;
                        let ss_res: f32 = agg_series
                            .iter()
                            .enumerate()
                            .map(|(i, &y)| (y - intercept - slope * i as f32).powi(2))
                            .sum();
                        let ss_tot: f32 = agg_series.iter().map(|&y| (y - m_y).powi(2)).sum();
                        if m_n > 2.0 && s_xx.abs() > 1e-9 && ss_tot > 1e-12 {
                            let t = slope / (ss_res / (m_n - 2.0) / s_xx).sqrt();
                            two_sided_t_pvalue(t as f64, m_n as f64 - 2.0) as f32
                        } else {
                            1.0
                        }
                    }
                };
                state.agg_linear_trend_buffer = agg_series;
                res
            }
        }
        Feature::SignalDistance => {
            let mut dist = 0.0;
            for i in 1..values.len() {
                dist += ((values[i] - values[i - 1]).powi(2) + 1.0).sqrt();
            }
            dist
        }
        _ => return None,
    };
    Some(res)
}

/// scipy.stats.linregress p-value: two-sided Student-t test of slope == 0.
fn two_sided_t_pvalue(t: f64, df: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, StudentsT};
    if !t.is_finite() {
        return if t.is_nan() { f64::NAN } else { 0.0 };
    }
    match StudentsT::new(0.0, 1.0, df) {
        Ok(dist) => 2.0 * dist.sf(t.abs()),
        Err(_) => f64::NAN,
    }
}
