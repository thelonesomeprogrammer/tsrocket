use crate::types::Feature;

#[inline(always)]
pub fn eval_min_max(feat: &Feature, context: &mut crate::context::FeatureContext) -> Option<f32> {
    let values = context.values;
    let state = &mut *context.state;
    let n = context.n;
    let _mean = context.mean;
    let _m2 = context.m2;
    let _m3 = context.m3;
    let _m4 = context.m4;
    let _var = context.var;
    let _std_dev = context.std_dev;
    let _mac_sum = context.mac_sum;
    let _mc_sum = context.mc_sum;
    let first_max_idx = context.first_max_idx;
    let last_max_idx = context.last_max_idx;
    let first_min_idx = context.first_min_idx;
    let last_min_idx = context.last_min_idx;
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
        Feature::Min => state.min_value,
        Feature::PkPkDistance => (state.max_value - state.min_value).abs(),
        Feature::Max => state.max_value,
        Feature::AbsMax => state.abs_max,
        Feature::FirstLocMax => first_max_idx as f32 / n,
        Feature::LastLocMax => (last_max_idx + 1) as f32 / n,
        Feature::FirstLocMin => first_min_idx as f32 / n,
        Feature::LastLocMin => (last_min_idx + 1) as f32 / n,
        Feature::MeanNAbsoluteMax(n_max) => {
            let mut abs_vals: Vec<f32> = values.iter().map(|v| v.abs()).collect();
            let count = (*n_max as usize).min(abs_vals.len());
            if count > 0 {
                if count < abs_vals.len() {
                    abs_vals.select_nth_unstable_by(count, |a, b| {
                        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                abs_vals.iter().take(count).sum::<f32>() / count as f32
            } else {
                f32::NAN
            }
        }
        Feature::HasDuplicateMax => {
            let mut count = 0;
            for &v in values {
                if v == state.max_value {
                    count += 1;
                    if count > 1 {
                        break;
                    }
                }
            }
            if count > 1 { 1.0 } else { 0.0 }
        }
        Feature::HasDuplicateMin => {
            let mut count = 0;
            for &v in values {
                if v == state.min_value {
                    count += 1;
                    if count > 1 {
                        break;
                    }
                }
            }
            if count > 1 { 1.0 } else { 0.0 }
        }
        Feature::HasDuplicate => {
            let mut unique = rustc_hash::FxHashSet::default();
            let mut has_dup = false;
            for &v in values {
                if !unique.insert(crate::common::value_key(v)) {
                    has_dup = true;
                    break;
                }
            }
            if has_dup { 1.0 } else { 0.0 }
        }
        _ => return None,
    };
    Some(res)
}
