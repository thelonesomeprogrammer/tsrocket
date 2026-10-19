use crate::types::Feature;

#[inline(always)]
pub fn eval_transform(feat: &Feature, context: &mut crate::context::FeatureContext) -> Option<f32> {
    let values = context.values;
    let state = &mut *context.state;
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
    let freq_centroid = context.freq_centroid;
    let spectral_decrease = context.spectral_decrease;
    let spectral_slope = context.spectral_slope;
    let spectral_spread = context.spectral_spread;
    let spectral_entropy = context.spectral_entropy;
    let spectral_roll_on = context.spectral_roll_on;
    let spectral_roll_off = context.spectral_roll_off;
    let spectral_spread = context.spectral_spread;
    let spectral_skewness = context.spectral_skewness;
    let spectral_kurtosis = context.spectral_kurtosis;
    let max_frequency = context.max_frequency;
    let median_frequency = context.median_frequency;
    let fundamental_frequency = context.fundamental_frequency;
    let power_bandwidth = context.power_bandwidth;
    let spectral_positive_turning = context.spectral_positive_turning;
    let spectral_variation = context.spectral_variation;
    let _fft_autocorr = context.fft_autocorr;
    let mfcc = context.mfcc;
    let lpcc = context.lpcc;
    let cwt_energy = context.cwt_energy;
    let cwt_entropy = context.cwt_entropy;
    let cwt_abs_mean = context.cwt_abs_mean;
    let cwt_std = context.cwt_std;
    let cwt_var = context.cwt_var;
    let fft_complex = context.fft_complex;
    let spectrum = context.spectrum;
    let unique_c3_lags = context.unique_c3_lags;
    let _unique_paa_totals = context.unique_paa_totals;
    let _paa_boundaries = context.paa_boundaries;

    let res = match feat {
        Feature::C3(lag) => {
            let l_idx = if let Some(idx) = unique_c3_lags.iter().position(|&l| l == *lag) {
                idx
            } else {
                return Some(0.0);
            };
            let l = *lag as usize;
            if values.len() > 2 * l {
                state.c3_sums[l_idx] / (values.len() - 2 * l) as f32
            } else {
                0.0
            }
        }
        // Mean of segment `index` of `total` equal-width segments; segment i
        // covers [i * n / total, (i + 1) * n / total). Same in every engine.
        Feature::Paa(total, index) => {
            let n_vals = values.len();
            if *total == 0 {
                return None;
            }
            let start = *index as usize * n_vals / *total as usize;
            let end = (*index as usize + 1) * n_vals / *total as usize;
            if start < end && end <= values.len() {
                values[start..end].iter().sum::<f32>() / (end - start) as f32
            } else {
                0.0
            }
        }

        Feature::SpktWelchDensity(coeff) => {
            let k = *coeff as usize;
            if k < context.welch_density.len() {
                context.welch_density[k]
            } else {
                f32::NAN // tsfresh: NaN past the last Welch frequency
            }
        }
        // tsfresh: pywt.cwt(x, widths, "mexh")[widths.index(w), coeff].
        Feature::CwtCoefficients(_widths, _len, coeff, w) => {
            let c = *coeff as usize;
            if c >= values.len() {
                return Some(f32::NAN);
            }
            super::cwt::mexh_cwt_coeff(values, *w as u16, c)
        }
        Feature::NumberCwtPeaks(n_val) => {
            super::cwt::number_cwt_peaks(values, *n_val as usize) as f32
        }
        // TSFEL: max(scipy.signal.welch(x / std(x), fs, nperseg=len(x))[1]).
        Feature::MaxPowerSpectrum => {
            let n_vals = values.len() as f64;
            let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n_vals;
            let std = (values
                .iter()
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / n_vals)
                .sqrt();
            let mut scaled = std::mem::take(&mut state.sort_buffer);
            scaled.clear();
            if std > 0.0 {
                scaled.extend(values.iter().map(|&v| (v as f64 / std) as f32));
            } else {
                scaled.extend_from_slice(values);
            }
            let fs = state.fs;
            let max = state
                .welch_full
                .get_or_insert_default()
                .psd(&scaled, scaled.len(), fs)
                .map_or(0.0, |psd| psd.iter().copied().fold(0.0, f32::max));
            state.sort_buffer = scaled;
            max
        }
        // tsfresh: NaN past the last rfft coefficient (n / 2).
        Feature::FftCoefficient(coeff, _) if *coeff as usize > context.values.len() / 2 => f32::NAN,
        Feature::FftCoefficient(coeff, attr) => {
            let k = *coeff as usize;
            let (re, im) = if !fft_complex.is_empty() && k < fft_complex.len() {
                (fft_complex[k].re, fft_complex[k].im)
            } else {
                (0.0, 0.0) // Fallback if no FFT computed
            };
            match attr {
                crate::types::FftAttr::Real => re,
                crate::types::FftAttr::Imag => im,
                crate::types::FftAttr::Abs => (re * re + im * im).sqrt(),
                // A real negative coefficient is 180 in numpy; f32 rounding can
                // leave im = -0.0, which atan2 turns into -180.
                crate::types::FftAttr::Angle => match im.atan2(re).to_degrees() {
                    a if a == -180.0 => 180.0,
                    a => a,
                },
            }
        }
        Feature::SpectralCentroid => freq_centroid,
        // TSFEL: sum(linspace(0, cumsum[-1], len) - cumsum(|X|)).
        Feature::SpectralDistance => {
            let len = spectrum.len();
            if len > 1 {
                let total = spectrum.iter().fold(0.0f64, |acc, &s| acc + s as f64);
                let mut cum = 0.0f64;
                spectrum
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| {
                        cum += s as f64;
                        total * i as f64 / (len - 1) as f64 - cum
                    })
                    .sum::<f64>() as f32
            } else {
                0.0
            }
        }
        Feature::SpectralDecrease => spectral_decrease,
        Feature::SpectralSlope => spectral_slope,
        Feature::SpectralRollOn => spectral_roll_on,
        Feature::SpectralRollOff => spectral_roll_off,
        Feature::MaxFrequency => max_frequency,
        Feature::MedianFrequency => median_frequency,
        Feature::FundamentalFrequency => fundamental_frequency,
        Feature::PowerBandwidth => power_bandwidth,
        Feature::SpectralPositiveTurning => spectral_positive_turning,
        Feature::SpectralVariation => spectral_variation,
        Feature::SpectralSpread => spectral_spread,
        Feature::SpectralSkewness => spectral_skewness,
        Feature::SpectralKurtosis => spectral_kurtosis,
        Feature::SpectrogramCoefficients(_, f_bits) => {
            if !spectrum.is_empty() {
                let target_freq = f32::from_bits(*f_bits);
                let freq_step = state.fs / context.dft_len as f32;
                let idx = (target_freq / freq_step).round() as usize;
                let idx = idx.min(spectrum.len() - 1);
                spectrum[idx]
            } else {
                0.0
            }
        }
        Feature::SpectrogramMeanCoeff(coeff, bins) => {
            return state
                .spectrogram
                .coefficient(values, *coeff, *bins, state.fs);
        }
        Feature::SpectralEntropy => spectral_entropy,
        Feature::WaveletFeatures(_w_bits, f_type) => {
            if values.len() >= 2 {
                let mut sum = 0.0;
                for i in (0..values.len() - 1).step_by(2) {
                    if *f_type == 0 {
                        sum += (values[i] - values[i + 1]).abs();
                    } else {
                        sum += (values[i] - values[i + 1]).powi(2);
                    }
                }
                if *f_type == 0 {
                    sum / (values.len() / 2) as f32
                } else {
                    (sum / (values.len() / 2) as f32).sqrt()
                }
            } else {
                0.0
            }
        }
        Feature::CalcCentroid(fs_bits) => {
            let fs = fs_bits.map_or(state.fs, f32::from_bits);
            if state.energy == 0.0 || fs == 0.0 {
                0.0
            } else {
                (state.t_energy / fs) / state.energy
            }
        }

        Feature::Mfcc(idx) => {
            if !mfcc.is_empty() && (*idx as usize) < mfcc.len() {
                mfcc[*idx as usize]
            } else {
                0.0
            }
        }
        Feature::Lpcc(idx) => {
            if !lpcc.is_empty() && (*idx as usize) < lpcc.len() {
                lpcc[*idx as usize]
            } else {
                0.0
            }
        }
        Feature::WaveletEnergy(idx) => {
            if !cwt_energy.is_empty() && (*idx as usize) < cwt_energy.len() {
                cwt_energy[*idx as usize]
            } else {
                0.0
            }
        }
        Feature::WaveletEntropy => cwt_entropy,
        Feature::WaveletAbsMean(idx) => {
            if !cwt_abs_mean.is_empty() && (*idx as usize) < cwt_abs_mean.len() {
                cwt_abs_mean[*idx as usize]
            } else {
                0.0
            }
        }
        Feature::WaveletStd(idx) => {
            if !cwt_std.is_empty() && (*idx as usize) < cwt_std.len() {
                cwt_std[*idx as usize]
            } else {
                0.0
            }
        }
        Feature::WaveletVar(idx) => {
            if !cwt_var.is_empty() && (*idx as usize) < cwt_var.len() {
                cwt_var[*idx as usize]
            } else {
                0.0
            }
        }
        // tsfresh: binned_entropy(pxx / max(pxx), bins) where pxx is the Welch
        // PSD (same nperseg=min(n,256), fs=1 config as spkt_welch_density).
        // Dividing by max(pxx) only rescales the histogram's bin edges, not the
        // relative bin membership, so binning pxx directly gives the same result
        // -- except when pxx is all zero (a perfectly flat window), where tsfresh
        // divides 0/0 into NaNs and binned_entropy short-circuits to NaN.
        Feature::FourierEntropy(max_bins) => {
            let max_bins = *max_bins as usize;
            let pxx = context.welch_density;
            if max_bins == 0 || pxx.is_empty() {
                return Some(f32::NAN);
            }
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for &v in pxx {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
            if min_val == max_val {
                return Some(if max_val == 0.0 { f32::NAN } else { 0.0 });
            }

            let mut hist = std::mem::take(&mut state.binned_entropy_buffer);
            hist.clear();
            hist.resize(max_bins, 0.0);

            let bin_width = (max_val - min_val) / max_bins as f32;
            for &v in pxx {
                let mut bin = ((v - min_val) / bin_width).floor() as usize;
                if bin >= max_bins {
                    bin = max_bins - 1;
                }
                hist[bin] += 1.0;
            }

            let n_pxx = pxx.len() as f32;
            let mut entropy = 0.0f32;
            for &count in &hist {
                if count > 0.0 {
                    let p = count / n_pxx;
                    entropy -= p * p.ln();
                }
            }
            state.binned_entropy_buffer = hist;
            entropy
        }
        // tsfresh fft_aggregated: (non-central) moments of np.abs(np.fft.rfft(x))
        // treated as a distribution over its bin index.
        Feature::FftAggregated(agg) => {
            let y = spectrum;
            let sum: f64 = y.iter().map(|&v| v as f64).sum();
            if y.is_empty() || sum == 0.0 {
                return Some(f32::NAN);
            }
            let moment = |p: i32| -> f64 {
                y.iter()
                    .enumerate()
                    .map(|(i, &v)| (i as f64).powi(p) * v as f64)
                    .sum::<f64>()
                    / sum
            };
            let centroid = moment(1);
            let variance = moment(2) - centroid * centroid;
            let res = match agg {
                crate::types::FftAggType::Centroid => centroid,
                crate::types::FftAggType::Variance => variance,
                crate::types::FftAggType::Skew => {
                    if variance < 0.5 {
                        f64::NAN
                    } else {
                        (moment(3) - 3.0 * centroid * variance - centroid.powi(3))
                            / variance.powf(1.5)
                    }
                }
                crate::types::FftAggType::Kurtosis => {
                    if variance < 0.5 {
                        f64::NAN
                    } else {
                        (moment(4) - 4.0 * centroid * moment(3)
                            + 6.0 * moment(2) * centroid.powi(2)
                            - 3.0 * centroid)
                            / variance.powi(2)
                    }
                }
            };
            res as f32
        }
        _ => return None,
    };
    Some(res)
}
