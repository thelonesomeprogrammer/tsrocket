use crate::types::Feature;

#[inline(always)]
pub fn eval_complexity(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let values = context.values;
    let _state = &mut *context.state;
    let _n = context.n;
    let _mean = context.mean;
    let _m2 = context.m2;
    let _m3 = context.m3;
    let _m4 = context.m4;
    let _var = context.var;
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
    let _fft_autocorr = context.fft_autocorr;
    let _fft_complex = context.fft_complex;
    let _spectrum = context.spectrum;
    let _unique_c3_lags = context.unique_c3_lags;
    let _unique_paa_totals = context.unique_paa_totals;
    let _paa_boundaries = context.paa_boundaries;

    let res = match feat {
        Feature::LempelZiv => eval_lempel_ziv(context),
        Feature::LempelZivComplexity(bins) => eval_lempel_ziv_complexity(context, *bins),
        Feature::ApproxEntropy(m, r_bits) if values.len() > *m as usize + 1 => {
            let m_val = *m as usize;
            // tsfresh: r is relative to the population std.
            let r = f32::from_bits(*r_bits) * population_std(values);
            crate::common::approx_entropy(
                m_val,
                r,
                values,
                &mut context.state.approx_entropy_counts,
            )
        }
        Feature::SampleEntropy => {
            let r = 0.2 * population_std(values);
            if values.len() <= 2 {
                return Some(f32::NAN);
            }
            let (a, b) = cached_sample_entropy_counts(values, r, context.state);
            crate::common::sample_entropy_from_counts(a, b)
        }
        Feature::TsfreshSampleEntropy => {
            let r = 0.2 * population_std(values);
            if values.len() <= 2 {
                return Some(f32::NAN);
            }
            let (a, b) = cached_sample_entropy_counts(values, r, context.state);
            crate::common::tsfresh_sample_entropy_from_counts(values, 2, r, a, b)
        }
        Feature::PermutationEntropy(tau, dimension) => {
            crate::common::permutation_entropy(values, *tau, *dimension)
        }
        Feature::Mse(m, maxscale) => {
            calc_mse(values, *m, *maxscale, context.std_dev, context.state)
        }
        Feature::Dfa => calc_dfa(values, context.state),
        Feature::HurstExponent => calc_hurst(values, context.state),
        Feature::HiguchiFd => calc_higuchi_fd(values, context.state),
        Feature::MaximumFractalLength => calc_maximum_fractal_length(values, context.state),
        Feature::PetrosianFractalDimension => calc_petrosian_fractal_dimension(values),
        _ => return None,
    };
    Some(res)
}

/// `sample_entropy_counts(values, 2, r)` for the current window, computed once
/// for `sample_entropy` and `tsfresh_sample_entropy`.
fn cached_sample_entropy_counts(
    values: &[f32],
    r: f32,
    state: &mut crate::common::ColumnState,
) -> (u32, u32) {
    match state.sample_entropy_counts {
        Some((bits, a, b)) if bits == r.to_bits() => (a, b),
        _ => {
            let (a, b) = crate::common::sample_entropy_counts(values, 2, r);
            state.sample_entropy_counts = Some((r.to_bits(), a, b));
            (a, b)
        }
    }
}

fn calc_mse(
    values: &[f32],
    m: u8,
    maxscale: u16,
    std_dev: f32,
    state: &mut crate::common::ColumnState,
) -> f32 {
    let n = values.len();
    if n < 160 || std_dev == 0.0 {
        return f32::NAN;
    }

    let tolerance = 0.2 * std_dev;
    let maxscale_val = if maxscale == 0 {
        (n / 13).max(1)
    } else {
        maxscale as usize
    };

    let mut mse_area = 0.0;
    let mut finite_count = 0;
    let mut first_val = f32::NAN;
    let mut last_val = f32::NAN;

    for scale in 1..=maxscale_val {
        let windows = n / scale;
        if windows <= m as usize {
            continue;
        }

        let se = if scale == 1 {
            crate::common::sample_entropy_simd(values, m as usize, tolerance)
        } else {
            state.mse_buffer.clear();
            for i in 0..windows {
                let start = i * scale;
                let end = start + scale;
                let mut sum = 0.0;
                for &v in &values[start..end] {
                    sum += v;
                }
                state.mse_buffer.push(sum / scale as f32);
            }
            crate::common::sample_entropy_simd(&state.mse_buffer, m as usize, tolerance)
        };

        if se.is_finite() {
            if finite_count == 0 {
                first_val = se;
            }
            last_val = se;
            mse_area += se;
            finite_count += 1;
        }
    }

    if finite_count == 0 {
        return f32::NAN;
    }

    let trapezoid = mse_area - 0.5 * (first_val + last_val);
    trapezoid / finite_count as f32
}

#[inline(always)]
fn calc_dfa(values: &[f32], state: &mut crate::common::ColumnState) -> f32 {
    let n = values.len();
    if n < 160 {
        return f32::NAN;
    }

    let first = values[0];
    if values.iter().all(|&v| v == first) {
        return f32::NAN;
    }

    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / (n as f64);

    let mut y_pref = std::mem::take(&mut state.workspace_f64_2);
    y_pref.resize(n + 1, 0.0);

    let mut y2_pref = std::mem::take(&mut state.workspace_f64_3);
    y2_pref.resize(n + 1, 0.0);

    let mut iy_pref = std::mem::take(&mut state.workspace_f64_4);
    iy_pref.resize(n + 1, 0.0);

    let mut current_acc = 0.0;
    for i in 0..n {
        current_acc += (values[i] as f64) - mean;

        y_pref[i + 1] = y_pref[i] + current_acc;
        y2_pref[i + 1] = y2_pref[i] + current_acc * current_acc;
        iy_pref[i + 1] = iy_pref[i] + (i as f64) * current_acc;
    }

    let min_scale = 4;
    let max_scale = n / 10;
    let num_scales = n / 2;

    let mut scales = std::mem::take(&mut state.approx_entropy_buffer);
    scales.clear();

    if num_scales > 1 {
        for i in 0..num_scales {
            let scale = (min_scale as f64)
                + ((max_scale - min_scale) as f64) * (i as f64) / ((num_scales - 1) as f64);
            scales.push(scale.floor() as usize);
        }
    } else {
        scales.push(min_scale);
    }

    scales.sort_unstable();
    scales.dedup();

    let mut log_scales = std::mem::take(&mut state.workspace_f64_5);
    log_scales.clear();
    let mut log_flucts_f64 = std::mem::take(&mut state.workspace_f64_1);
    log_flucts_f64.clear();

    for &s in &scales {
        let num_windows = n / s;
        if num_windows == 0 {
            continue;
        }

        let mut rms_sum = 0.0;
        let x_bar = (s - 1) as f64 / 2.0;
        let ss_xx = (s as f64) * (((s * s) as f64) - 1.0) / 12.0;

        for idx in 0..num_windows {
            let start = idx * s;
            let end = start + s;

            let sum_y = y_pref[end] - y_pref[start];
            let sum_y2 = y2_pref[end] - y2_pref[start];
            let sum_iy_global = iy_pref[end] - iy_pref[start];

            let sum_jy = sum_iy_global - (start as f64) * sum_y;

            let y_bar = sum_y / (s as f64);
            let ss_xy = sum_jy - (s as f64) * x_bar * y_bar;
            let mut beta1 = 0.0;
            if ss_xx != 0.0 {
                beta1 = ss_xy / ss_xx;
            }
            let beta0 = y_bar - beta1 * x_bar;

            let mut rss = sum_y2 - beta0 * sum_y - beta1 * sum_jy;
            if rss < 0.0 {
                rss = 0.0;
            }

            rms_sum += rss / (s as f64);
        }

        let fluct = (rms_sum / (num_windows as f64)).sqrt();
        log_scales.push((s as f64).ln());
        log_flucts_f64.push(fluct.ln());
    }

    let mut i_plateau = log_flucts_f64.len();
    if log_flucts_f64.len() > 5 {
        let mean_y = log_flucts_f64.iter().sum::<f64>() / (log_flucts_f64.len() as f64);
        let consecutive_points = 5;
        let dy_len = log_flucts_f64.len() - 1;

        for i in 0..=(dy_len - consecutive_points) {
            let mut all_below = true;
            for j in 0..consecutive_points {
                let dy = log_flucts_f64[i + j + 1] - log_flucts_f64[i + j];
                if dy.abs() >= 0.1 {
                    all_below = false;
                    break;
                }
            }
            if all_below {
                let plateau_val = log_flucts_f64[i..i + consecutive_points]
                    .iter()
                    .sum::<f64>()
                    / (consecutive_points as f64);
                if plateau_val > mean_y {
                    i_plateau = i;
                    break;
                }
            }
        }
    }

    state.workspace_f64_2 = y_pref;
    state.workspace_f64_3 = y2_pref;
    state.workspace_f64_4 = iy_pref;
    state.approx_entropy_buffer = scales;

    let res = if i_plateau > 1 {
        let x = &log_scales[0..i_plateau];
        let y = &log_flucts_f64[0..i_plateau];

        let mean_x = x.iter().sum::<f64>() / (x.len() as f64);
        let mean_y = y.iter().sum::<f64>() / (y.len() as f64);

        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            den += dx * dx;
        }
        if den == 0.0 {
            f32::NAN
        } else {
            (num / den) as f32
        }
    } else {
        f32::NAN
    };

    state.workspace_f64_1 = log_flucts_f64;
    state.workspace_f64_5 = log_scales;

    res
}

/// Calculates the Hurst exponent of the signal through Rescaled Range (R/S) analysis.
///
/// This implementation matches the TSFEL logic exactly:
/// - Windows scales: `linspace(4, n // 10, n // 2, dtype=int)` deduplicated.
/// - The windows do not overlap; each chunk is evaluated independently.
/// - The profile is built as the cumulative sum of the demeaned chunk (`cumsum(chunk - mean)`).
/// - The chunk standard deviation is calculated over the raw chunk values.
/// - Minimum size behavior requires 160 elements; otherwise, NaN is returned.
#[inline(always)]
fn calc_hurst(values: &[f32], state: &mut crate::common::ColumnState) -> f32 {
    let n = values.len();
    if n < 160 {
        return f32::NAN;
    }

    let first = values[0];
    if values.iter().all(|&v| v == first) {
        return f32::NAN;
    }

    let min_scale = 4;
    let max_scale = n / 10;
    let num_scales = n / 2;

    let mut scales = std::mem::take(&mut state.approx_entropy_buffer);
    scales.clear();

    if num_scales > 1 {
        for i in 0..num_scales {
            let scale = (min_scale as f64)
                + ((max_scale - min_scale) as f64) * (i as f64) / ((num_scales - 1) as f64);
            scales.push(scale.floor() as usize);
        }
    } else {
        scales.push(min_scale);
    }

    scales.sort_unstable();
    scales.dedup();

    let mut log_scales = std::mem::take(&mut state.workspace_f64_5);
    log_scales.clear();
    let mut log_rs = std::mem::take(&mut state.workspace_f64_4);
    log_rs.clear();

    let mut y_pref = std::mem::take(&mut state.workspace_f64_1);
    y_pref.resize(n + 1, 0.0);

    let mut y2_pref = std::mem::take(&mut state.workspace_f64_2);
    y2_pref.resize(n + 1, 0.0);

    for i in 0..n {
        let v = values[i] as f64;
        y_pref[i + 1] = y_pref[i] + v;
        y2_pref[i + 1] = y2_pref[i] + v * v;
    }

    for &s in &scales {
        let num_windows = n / s;
        if num_windows == 0 {
            continue;
        }

        let mut rs_sum = 0.0;
        let mut valid_windows = 0;

        for idx in 0..num_windows {
            let start = idx * s;
            let end = start + s;

            let sum_y = y_pref[end] - y_pref[start];
            let sum_y2 = y2_pref[end] - y2_pref[start];
            let mean = sum_y / (s as f64);

            let mut var = (sum_y2 - 2.0 * mean * sum_y + (s as f64) * mean * mean) / (s as f64);
            if var < 0.0 {
                var = 0.0;
            }
            let std = var.sqrt();

            let mut acc = 0.0;
            let mut max_acc = f64::NEG_INFINITY;
            let mut min_acc = f64::INFINITY;

            for i in start..end {
                acc += (values[i] as f64) - mean;
                if acc > max_acc {
                    max_acc = acc;
                }
                if acc < min_acc {
                    min_acc = acc;
                }
            }

            let r = max_acc - min_acc;
            if std > 0.0 {
                rs_sum += r / std;
                valid_windows += 1;
            }
        }

        if valid_windows > 0 {
            let mean_rs = rs_sum / (valid_windows as f64);
            log_scales.push((s as f64).log10());
            log_rs.push(mean_rs.log10());
        }
    }

    state.workspace_f64_1 = y_pref;
    state.workspace_f64_2 = y2_pref;
    state.approx_entropy_buffer = scales;

    let res = if log_scales.len() > 1 {
        let x = &log_scales;
        let y = &log_rs;

        let mean_x = x.iter().sum::<f64>() / (x.len() as f64);
        let mean_y = y.iter().sum::<f64>() / (y.len() as f64);

        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            den += dx * dx;
        }
        if den == 0.0 {
            f32::NAN
        } else {
            (num / den) as f32
        }
    } else {
        f32::NAN
    };

    state.workspace_f64_5 = log_scales;
    state.workspace_f64_4 = log_rs;

    res
}

#[inline(always)]
fn ensure_higuchi_lengths(values: &[f32], state: &mut crate::common::ColumnState) {
    if !state.higuchi_lk.is_empty() {
        return; // Already computed
    }
    let n = values.len();
    if n < 10 {
        return;
    }
    let k_max = n / 10 - 1;
    if k_max < 2 {
        return;
    }

    let mut higuchi_lk = std::mem::take(&mut state.higuchi_lk);
    let mut higuchi_k_values = std::mem::take(&mut state.higuchi_k_values);
    higuchi_lk.clear();
    higuchi_k_values.clear();

    // Opt: Compute linear regression standard covariance/variance dynamically in one pass
    // to avoid allocating two vectors for `log_k_inv` and `log_lk`

    // TSFEL normalization logic requires evaluating sum for each m individually.
    // Instead of using a vector of size k, we iterate directly using O(1) space to avoid allocations.
    for k in 1..=k_max {
        let mut lmk_sum = 0.0f64;

        for m in 1..=k {
            let iters = (n - m) / k;
            let mut sum_length = 0.0f64;

            // Vectorize by iterating through contiguous slices where possible or just manually striding
            // The compiler typically unrolls this strided iteration
            let mut i = 1;
            while i <= iters {
                let idx1 = m + i * k - 1;
                let idx2 = m + (i - 1) * k - 1;
                sum_length += (values[idx1] - values[idx2]).abs() as f64;
                i += 1;
            }

            let norm_factor = (n - 1) as f64 / (iters * k) as f64;
            let lmk = (sum_length * norm_factor) / k as f64;
            lmk_sum += lmk;
        }

        let lk = lmk_sum / k as f64;
        higuchi_lk.push(lk);
        higuchi_k_values.push(k as f64);
    }

    state.higuchi_lk = higuchi_lk;
    state.higuchi_k_values = higuchi_k_values;
}

#[inline(always)]
fn calc_higuchi_fd(values: &[f32], state: &mut crate::common::ColumnState) -> f32 {
    let n = values.len();
    if n < 10 {
        return f32::NAN;
    }
    let k_max = n / 10 - 1;
    if k_max < 2 {
        return f32::NAN;
    }

    ensure_higuchi_lengths(values, state);

    let lk = &state.higuchi_lk;
    let k_values = &state.higuchi_k_values;

    if lk.len() < 2 {
        return f32::NAN;
    }

    let count = lk.len() as f64;
    let mut log_k_inv_sum = 0.0f64;
    let mut log_lk_sum = 0.0f64;

    for i in 0..lk.len() {
        log_k_inv_sum += (1.0 / k_values[i]).ln();
        log_lk_sum += lk[i].ln();
    }

    let x_mean = log_k_inv_sum / count;
    let y_mean = log_lk_sum / count;

    let mut num = 0.0f64;
    let mut den = 0.0f64;

    for i in 0..lk.len() {
        let dx = (1.0 / k_values[i]).ln() - x_mean;
        let dy = lk[i].ln() - y_mean;
        num += dx * dy;
        den += dx * dx;
    }
    if den == 0.0 {
        return f32::NAN;
    }

    (num / den) as f32
}

#[inline(always)]
fn calc_maximum_fractal_length(values: &[f32], state: &mut crate::common::ColumnState) -> f32 {
    let n = values.len();
    if n < 10 {
        return f32::NAN;
    }
    let k_max = n / 10 - 1;
    if k_max < 2 {
        return f32::NAN;
    }

    ensure_higuchi_lengths(values, state);

    let lk = &state.higuchi_lk;
    let k_values = &state.higuchi_k_values;

    if lk.len() < 2 {
        return f32::NAN;
    }

    let count = lk.len() as f64;
    let mut log_k_inv_sum = 0.0f64;
    let mut log_lk_sum = 0.0f64;

    for i in 0..lk.len() {
        log_k_inv_sum += (1.0 / k_values[i]).log10();
        log_lk_sum += lk[i].log10();
    }

    let x_mean = log_k_inv_sum / count;
    let y_mean = log_lk_sum / count;

    let mut num = 0.0f64;
    let mut den = 0.0f64;

    for i in 0..lk.len() {
        let dx = (1.0 / k_values[i]).log10() - x_mean;
        let dy = lk[i].log10() - y_mean;
        num += dx * dy;
        den += dx * dx;
    }

    if den == 0.0 {
        return f32::NAN;
    }

    let slope = num / den;
    let intercept = y_mean - slope * x_mean;

    intercept as f32
}

/// TSFEL: counts sign changes in `diff(sign(diff(signal)))`, where a
/// transition to/from a flat run (sign 0) also counts as a change.
#[inline(always)]
fn calc_petrosian_fractal_dimension(values: &[f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return f32::NAN;
    }

    let sign = |x: f32| -> i8 {
        if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        }
    };

    let mut num_sign_changes: u32 = 0;
    if n >= 3 {
        let mut prev_sign = sign(values[1] - values[0]);
        for i in 1..n - 1 {
            let s = sign(values[i + 1] - values[i]);
            if s != prev_sign {
                num_sign_changes += 1;
            }
            prev_sign = s;
        }
    }

    let n_f = n as f32;
    let log_n = n_f.log10();
    log_n / (log_n + (n_f / (n_f + 0.4 * num_sign_changes as f32)).log10())
}

/// np.std(x): computed here so the O(n^2) entropies don't depend on which
/// accumulation flags other requested features happen to enable.
fn population_std(values: &[f32]) -> f32 {
    let n = values.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = values
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    var.sqrt() as f32
}

use std::simd::f32x8;
use std::simd::prelude::*;

#[inline(always)]
fn lz_complexity_bits(state: &mut crate::common::ColumnState, n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }

    state.lz_binary_trie.clear();
    state.lz_binary_trie.push([0, 0]); // root node
    let mut num_substrings = 0;

    let mut ind = 0;

    // ⚡ Bolt optimization: Maintain tree state (`curr_node`) while walking down the string
    // instead of restarting from the root node on every single character.
    // This reduces the complexity from O(N^2) worst-case to O(N).
    while ind < n {
        let mut curr_node = 0;
        let mut inc = 1;

        while ind + inc <= n {
            let symbol_idx = ind + inc - 1;
            let bit = ((state.lz_bit_buffer[symbol_idx / 64] >> (symbol_idx % 64)) & 1) as usize;

            let next_node = state.lz_binary_trie[curr_node][bit];
            if next_node != 0 {
                curr_node = next_node;
                inc += 1;
            } else {
                let new_node = state.lz_binary_trie.len();
                state.lz_binary_trie.push([0, 0]);
                state.lz_binary_trie[curr_node][bit] = new_node;
                num_substrings += 1;
                ind += inc;
                inc = 1; // the next phrase starts fresh, as in tsfresh/TSFEL
                break;
            }
        }

        if ind + inc > n {
            break;
        }
    }

    num_substrings as f32 / n as f32
}

/// tsfresh's LZ phrase count over `lz_symbol_buffer`, divided by its length:
/// grow the current phrase until it is new, then start the next one after it.
/// Phrases live in a flat trie (no allocation per node); the root's children,
/// visited once per phrase, are a direct table.
#[inline(always)]
fn lz_complexity(state: &mut crate::common::ColumnState) -> f32 {
    use crate::common::LzNode;
    let sequence = &state.lz_symbol_buffer;
    let nodes = &mut state.lz_trie_nodes;
    let n = sequence.len();
    if n == 0 {
        return 0.0;
    }
    nodes.clear();
    nodes.push(LzNode::default());
    let mut root = [0u32; 256];
    let mut num_substrings = 0;
    let mut ind = 0;

    'phrases: while ind < n {
        let first = sequence[ind];
        let mut node = root[first as usize];
        if node == 0 {
            root[first as usize] = nodes.len() as u32;
            nodes.push(LzNode {
                symbol: first,
                ..LzNode::default()
            });
            num_substrings += 1;
            ind += 1;
            continue;
        }
        let mut inc = 2;
        loop {
            if ind + inc > n {
                break 'phrases; // the rest is a known phrase
            }
            let symbol = sequence[ind + inc - 1];
            let mut child = nodes[node as usize].first_child;
            while child != 0 && nodes[child as usize].symbol != symbol {
                child = nodes[child as usize].next_sibling;
            }
            if child != 0 {
                node = child;
                inc += 1;
            } else {
                let new = nodes.len() as u32;
                let next_sibling = nodes[node as usize].first_child;
                nodes.push(LzNode {
                    first_child: 0,
                    next_sibling,
                    symbol,
                });
                nodes[node as usize].first_child = new;
                num_substrings += 1;
                ind += inc;
                break;
            }
        }
    }

    num_substrings as f32 / n as f32
}

/// Computes the Lempel-Ziv's (LZ) complexity index, normalized by the signal's length.
/// TSFEL definition: binarises around the mean threshold, and counts LZ76 dictionary entries.
/// Optimizaton: binarise 8 symbols per instruction into a u64 buffer.
fn eval_lempel_ziv(context: &mut crate::context::FeatureContext) -> f32 {
    let values = context.values;
    let n = values.len();
    let state = &mut *context.state;
    if n == 0 {
        return 0.0;
    }

    let threshold = context.mean;
    let threshold_vec = f32x8::splat(threshold);

    state.lz_bit_buffer.clear();
    state.lz_bit_buffer.resize((n + 63) / 64, 0);

    let mut chunks = values.chunks_exact(8);
    let mut idx = 0;
    for chunk in &mut chunks {
        let vals = f32x8::from_slice(chunk);
        let mask = vals.simd_gt(threshold_vec);
        let bitmask = mask.to_bitmask() as u64;
        state.lz_bit_buffer[idx / 64] |= bitmask << (idx % 64);
        idx += 8;
    }
    for &v in chunks.remainder() {
        if v > threshold {
            state.lz_bit_buffer[idx / 64] |= 1 << (idx % 64);
        }
        idx += 1;
    }

    lz_complexity_bits(state, n)
}

/// Calculate a complexity estimate based on the Lempel-Ziv compression algorithm.
/// tsfresh definition: discretises into `bins` equal-width bins and then runs an LZ76-style dictionary parse.
fn eval_lempel_ziv_complexity(context: &mut crate::context::FeatureContext, bins: u16) -> f32 {
    let values = context.values;
    let state = &mut *context.state;
    if values.is_empty() || bins < 2 {
        return 0.0;
    }

    let min_val = state.min_value;
    let max_val = state.max_value;

    state.lz_symbol_buffer.clear();

    if max_val <= min_val {
        for _ in values {
            state.lz_symbol_buffer.push(0);
        }
        return lz_complexity(state);
    }

    // tsfresh: searchsorted(np.linspace(min, max, bins + 1)[1:], v, "left"),
    // in f64 like numpy, so values on an edge land in the same bin.
    use super::EdgeSide::SearchsortedLeft;
    let mut edges = std::mem::take(&mut state.bin_edges);
    let norm = super::numpy_bin_edges(
        min_val,
        max_val,
        bins as usize,
        SearchsortedLeft,
        &mut edges,
    );
    for &v in values {
        let k = super::numpy_bin(v, min_val, norm, &edges, SearchsortedLeft);
        state.lz_symbol_buffer.push(k as u8);
    }
    state.bin_edges = edges;

    lz_complexity(state)
}
