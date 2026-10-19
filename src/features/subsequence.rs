use crate::context::FeatureContext;
use crate::types::{AggFunc, Feature};

pub fn eval_subsequence(feat: &Feature, context: &mut FeatureContext) -> Option<f32> {
    match feat {
        Feature::QuerySimilarityCount(l, threshold_bits) => {
            let l = *l as usize;
            if l == 0 {
                // No query: tsfresh's default (query=None) is always NaN.
                return Some(f32::NAN);
            }
            if context.values.len() < l {
                return None;
            }
            let threshold = f32::from_bits(*threshold_bits);
            let q = &context.values[..l];
            let distances = mass_distance_profile(context.values, q, context);

            let mut count = 0;
            for &d in distances {
                if d <= threshold {
                    count += 1;
                }
            }
            Some(count as f32)
        }
        Feature::MatrixProfile(l, agg) => {
            let l = *l as usize;
            if l == 0 || context.values.len() < l {
                return None;
            }
            // One profile per window serves every aggregate over it.
            let state = &mut *context.state;
            if state.matrix_profile_m != l {
                let mut scratch = std::mem::take(&mut state.matrix_profile_scratch);
                stomp_matrix_profile(context.values, l, &mut state.matrix_profile, &mut scratch);
                state.matrix_profile_scratch = scratch;
                state.matrix_profile_m = l;
            }
            let profile = &state.matrix_profile;
            if profile.is_empty() {
                return None;
            }

            match agg {
                AggFunc::Min => Some(profile.iter().copied().fold(f32::INFINITY, f32::min)),
                AggFunc::Max => Some(profile.iter().copied().fold(f32::NEG_INFINITY, f32::max)),
                AggFunc::Mean => {
                    let sum: f32 = profile.iter().sum();
                    Some(sum / profile.len() as f32)
                }
                AggFunc::Var => {
                    let mean = profile.iter().sum::<f32>() / profile.len() as f32;
                    let var = profile.iter().map(|&v| (v - mean).powi(2)).sum::<f32>()
                        / (profile.len() - 1).max(1) as f32;
                    Some(var)
                }
                AggFunc::Median => {
                    let mut buf = std::mem::take(&mut state.sort_buffer);
                    buf.clear();
                    buf.extend_from_slice(profile);
                    let median = super::median_in_place(&mut buf);
                    state.sort_buffer = buf;
                    Some(median)
                }
            }
        }
        _ => None,
    }
}

fn mass_distance_profile<'a>(x: &[f32], q: &[f32], context: &'a mut FeatureContext) -> &'a [f32] {
    let m = q.len();
    let n = x.len();
    if n < m {
        return &[];
    }

    let mut q_mean = 0.0;
    for &v in q {
        q_mean += v;
    }
    q_mean /= m as f32;
    let mut q_std = 0.0;
    for &v in q {
        q_std += (v - q_mean) * (v - q_mean);
    }
    q_std = (q_std / m as f32).sqrt();
    if q_std < 1e-6 {
        q_std = 1.0;
    }

    let mut q_norm = std::mem::take(&mut context.state.sort_buffer);
    q_norm.clear();
    q_norm.reserve(m);
    for &v in q {
        q_norm.push((v - q_mean) / q_std);
    }

    let mut distances = std::mem::take(&mut context.state.diff_buffer);
    distances.clear();
    distances.reserve(n - m + 1);

    for i in 0..=n - m {
        let window = &x[i..i + m];

        let mut w_mean = 0.0;
        for &v in window {
            w_mean += v;
        }
        w_mean /= m as f32;

        let mut w_std = 0.0;
        for &v in window {
            w_std += (v - w_mean) * (v - w_mean);
        }
        w_std = (w_std / m as f32).sqrt();
        if w_std < 1e-6 {
            w_std = 1.0;
        }

        let mut dot = 0.0;
        for j in 0..m {
            let w_norm = (window[j] - w_mean) / w_std;
            dot += q_norm[j] * w_norm;
        }

        let d = (2.0 * m as f32 * (1.0 - dot / m as f32).max(0.0)).sqrt();
        distances.push(d);
    }

context.state.sort_buffer = q_norm;
    context.state.diff_buffer = distances;
    &context.state.diff_buffer
}

/// Matrix profile with an exact dot product per pair (rolling STOMP/MPX
/// updates drift by more than f32 can afford on smooth series). Subsequences
/// are z-normalised once into a transposed `m x l` table, then the upper
/// triangle is scanned row by row, four columns per SIMD step, and each pair
/// updates both ends: `l²/2` dot products with no divisions.
fn stomp_matrix_profile(x: &[f32], m: usize, profile: &mut Vec<f32>, scratch: &mut Vec<f32>) {
    use std::simd::{cmp::SimdPartialOrd as _, f32x4, num::SimdFloat as _};
    profile.clear();
    let n = x.len();
    if n <= m {
        profile.push(0.0);
        return;
    }

    let l = n - m + 1;
    let mf = m as f32;
    // Centre on the series mean first: window means of a large-offset series
    // would otherwise round away most of the signal.
    let mu = x.iter().sum::<f32>() / n as f32;

    // z[t * l + j] = (x[j + t] - mean_j) / std_j
    scratch.clear();
    scratch.resize(m * l, 0.0);
    let z = scratch.as_mut_slice();
    for j in 0..l {
        let window = &x[j..j + m];
        let mean = window.iter().map(|&v| v - mu).sum::<f32>() / mf;
        let var = window
            .iter()
            .map(|&v| (v - mu - mean) * (v - mu - mean))
            .sum::<f32>()
            / mf;
        let mut std = var.sqrt();
        if std < 1e-6 {
            std = 1.0; // avoid div by zero, standardize flat line
        }
        let inv = 1.0 / std;
        for (t, &v) in window.iter().enumerate() {
            z[t * l + j] = (v - mu - mean) * inv;
        }
    }

    // Squared distances 2m - 2 dot while scanning; sqrt once at the end.
    profile.resize(l, f32::INFINITY);
    // stumpy / matrixprofile: neighbours within ceil(m / 4) are trivial matches.
    let exclusion_zone = m.div_ceil(4);
    let two_m = 2.0 * mf;
    let two_m_v = f32x4::splat(two_m);
    let zero = f32x4::splat(0.0);

    for i in 0..l {
        let mut j = i + exclusion_zone + 1;
        if j >= l {
            continue;
        }
        let mut row_min = f32x4::splat(f32::INFINITY);
        while j + 4 <= l {
            let mut dot = zero;
            for t in 0..m {
                let row = &z[t * l..(t + 1) * l];
                dot += f32x4::splat(row[i]) * f32x4::from_slice(&row[j..j + 4]);
            }
            let d2 = (two_m_v - dot - dot).simd_max(zero);
            row_min = row_min.simd_min(d2);
            let col = f32x4::from_slice(&profile[j..j + 4]).simd_min(d2);
            col.copy_to_slice(&mut profile[j..j + 4]);
            j += 4;
        }
        let mut best = row_min.reduce_min().min(profile[i]);
        for j in j..l {
            let dot: f32 = (0..m).map(|t| z[t * l + i] * z[t * l + j]).sum();
            let d2 = (two_m - 2.0 * dot).max(0.0);
            best = best.min(d2);
            if d2 < profile[j] {
                profile[j] = d2;
            }
        }
        profile[i] = best;
    }

    for p in profile.iter_mut() {
        *p = p.sqrt();
    }
}
