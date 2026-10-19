//! tsfresh and TSFEL output column names, translated to tsrocket names.
//!
//! A model trained on tsfresh or TSFEL output knows its inputs by those
//! libraries' column names, so accepting them unchanged makes the deployed
//! feature list just `df.columns`. Each column name is translated to the
//! canonical tsrocket name and then parsed by the normal path, so parameter
//! validation lives in one place (`parse.rs`).
//!
//! - tsfresh: `<kind>__<calculator>[__<param>_<value>...]`, e.g.
//!   `value__fft_coefficient__attr_"abs"__coeff_3`. The kind is optional.
//! - TSFEL: `<channel>_<Feature name>[_<index or frequency>]`, e.g.
//!   `0_Spectral centroid`, `0_LPCC_3`, `0_Wavelet energy_12.5Hz`. The channel
//!   is optional. Frequency-suffixed names depend on the sampling frequency,
//!   so they only parse when the engine's `fs` is known.

/// The canonical tsrocket name for a tsfresh or TSFEL column name, if `s` is one.
pub(super) fn translate(s: &str, fs: Option<f32>) -> Option<String> {
    tsfresh(s).or_else(|| tsfel(s, fs))
}

// ─── tsfresh ────────────────────────────────────────────────────────────────

fn tsfresh(s: &str) -> Option<String> {
    let parts: Vec<&str> = s.split("__").collect();
    // The kind (the value column's name) is everything before the calculator.
    (0..parts.len()).find_map(|i| tsfresh_calculator(parts[i], &parts[i + 1..]))
}

/// Every parameter name tsfresh uses. Longest match wins, so `max_bins_10` is
/// `max_bins`, not `max`.
const TSFRESH_PARAMS: &[&str] = &[
    "aggtype",
    "attr",
    "autolag",
    "bins",
    "chunk_len",
    "coeff",
    "dimension",
    "f_agg",
    "feature",
    "isabs",
    "k",
    "lag",
    "m",
    "max",
    "max_bins",
    "maxlag",
    "min",
    "n",
    "normalize",
    "num_segments",
    "number_of_maxima",
    "q",
    "qh",
    "ql",
    "query",
    "r",
    "segment_focus",
    "t",
    "tau",
    "threshold",
    "value",
    "w",
    "widths",
];

/// `param_value` segments of one tsfresh column name. Every parameter must be
/// consumed with `take`, so a name with unexpected parameters is rejected.
struct Params<'a>(Vec<(&'static str, &'a str)>);

impl<'a> Params<'a> {
    fn new(segments: &[&'a str]) -> Option<Self> {
        let mut out = Vec::with_capacity(segments.len());
        for seg in segments {
            let key = TSFRESH_PARAMS
                .iter()
                .filter(|k| {
                    seg.len() > k.len() && seg.starts_with(*k) && seg.as_bytes()[k.len()] == b'_'
                })
                .max_by_key(|k| k.len())?;
            // String values are quoted: attr_"abs".
            let value = seg[key.len() + 1..].trim_matches('"');
            out.push((*key, value));
        }
        Some(Self(out))
    }

    fn take(&mut self, key: &str) -> Option<&'a str> {
        let i = self.0.iter().position(|(k, _)| *k == key)?;
        Some(self.0.swap_remove(i).1)
    }
}

fn tsfresh_calculator(name: &str, segments: &[&str]) -> Option<String> {
    let mut p = Params::new(segments)?;
    let out = match name {
        "sum_values" => "total_sum".into(),
        "abs_energy" => "energy".into(),
        "standard_deviation" => "std_dev".into(),
        "maximum" => "max_value".into(),
        "minimum" => "min_value".into(),
        "absolute_maximum" => "abs_max".into(),
        "absolute_sum_of_changes" => "abs_sum_change".into(),
        "first_location_of_maximum" => "first_loc_max".into(),
        "last_location_of_maximum" => "last_loc_max".into(),
        "first_location_of_minimum" => "first_loc_min".into(),
        "last_location_of_minimum" => "last_loc_min".into(),
        "sample_entropy" => "tsfresh_sample_entropy".into(),
        "mean"
        | "median"
        | "variance"
        | "skewness"
        | "kurtosis"
        | "length"
        | "mean_abs_change"
        | "mean_change"
        | "mean_second_derivative_central"
        | "variation_coefficient"
        | "root_mean_square"
        | "count_above_mean"
        | "count_below_mean"
        | "longest_strike_above_mean"
        | "longest_strike_below_mean"
        | "has_duplicate"
        | "has_duplicate_max"
        | "has_duplicate_min"
        | "benford_correlation"
        | "variance_larger_than_standard_deviation"
        | "sum_of_reoccurring_values"
        | "sum_of_reoccurring_data_points"
        | "percentage_of_reoccurring_values_to_all_values"
        | "percentage_of_reoccurring_datapoints_to_all_datapoints"
        | "ratio_value_number_to_time_series_length" => name.into(),
        "cid_ce" => match p.take("normalize")? {
            "True" => "cid_ce_normalized".into(),
            "False" => "cid_ce".into(),
            _ => return None,
        },
        "time_reversal_asymmetry_statistic" => {
            format!("time_reversal_asymmetry-{}", p.take("lag")?)
        }
        "c3" => format!("c3-{}", p.take("lag")?),
        "symmetry_looking" => format!("symmetry_looking-{}", p.take("r")?),
        "large_standard_deviation" => format!("large_standard_deviation-{}", p.take("r")?),
        "ratio_beyond_r_sigma" => format!("ratio_beyond_r_sigma-{}", p.take("r")?),
        "quantile" => format!("quantile-{}", p.take("q")?),
        "index_mass_quantile" => format!("index_mass_quantile-{}", p.take("q")?),
        "autocorrelation" => format!("autocorr-{}", p.take("lag")?),
        "partial_autocorrelation" => format!("partial_autocorr-{}", p.take("lag")?),
        "agg_autocorrelation" => {
            format!(
                "agg_autocorrelation-{}-{}",
                p.take("f_agg")?,
                p.take("maxlag")?
            )
        }
        "number_peaks" => format!("number_peaks__n_{}", p.take("n")?),
        "number_cwt_peaks" => format!("number_cwt_peaks__n_{}", p.take("n")?),
        "number_crossing_m" => format!("number_crossing_m__m_{}", p.take("m")?),
        "binned_entropy" => format!("binned_entropy__max_bins_{}", p.take("max_bins")?),
        "spkt_welch_density" => format!("spkt_welch_density__coeff_{}", p.take("coeff")?),
        "cwt_coefficients" => format!(
            "cwt_coefficients__coeff_{}__w_{}__widths_{}",
            p.take("coeff")?,
            p.take("w")?,
            p.take("widths")?
        ),
        "ar_coefficient" => format!("ar_coefficient-{}-{}", p.take("k")?, p.take("coeff")?),
        "change_quantiles" => format!(
            "change_quantiles-{}-{}-{}-{}",
            p.take("ql")?,
            p.take("qh")?,
            p.take("isabs")?,
            p.take("f_agg")?
        ),
        "fft_coefficient" => format!("fft_coeff-{}-{}", p.take("coeff")?, p.take("attr")?),
        "fft_aggregated" => format!("fft_aggregated-{}", p.take("aggtype")?),
        "value_count" => format!("value_count-{}", p.take("value")?),
        "range_count" => format!("range_count-{}-{}", p.take("min")?, p.take("max")?),
        "count_above" => format!("count_above-{}", p.take("t")?),
        "count_below" => format!("count_below-{}", p.take("t")?),
        "approximate_entropy" => format!("approx_entropy-{}-{}", p.take("m")?, p.take("r")?),
        "friedrich_coefficients" => format!(
            "friedrich_coefficients-{}-{}-{}",
            p.take("m")?,
            p.take("r")?,
            p.take("coeff")?
        ),
        "max_langevin_fixed_point" => {
            format!("max_langevin_fixed_point-{}-{}", p.take("m")?, p.take("r")?)
        }
        "linear_trend" => format!("linear_trend-{}", p.take("attr")?),
        "agg_linear_trend" => format!(
            "agg_linear_trend-{}-{}-{}",
            p.take("attr")?,
            p.take("chunk_len")?,
            p.take("f_agg")?
        ),
        "augmented_dickey_fuller" => {
            // tsrocket implements tsfresh's default lag search only.
            if p.take("autolag").is_some_and(|a| a != "AIC") {
                return None;
            }
            format!("augmented_dickey_fuller-{}", p.take("attr")?)
        }
        "energy_ratio_by_chunks" => format!(
            "energy_ratio_by_chunks_num_segments_{}__segment_focus_{}",
            p.take("num_segments")?,
            p.take("segment_focus")?
        ),
        "lempel_ziv_complexity" => format!("lempel_ziv_complexity-{}", p.take("bins")?),
        "fourier_entropy" => format!("fourier_entropy-{}", p.take("bins")?),
        "permutation_entropy" => {
            format!(
                "permutation_entropy-{}-{}",
                p.take("tau")?,
                p.take("dimension")?
            )
        }
        "mean_n_absolute_max" => {
            format!("mean_n_absolute_max-{}", p.take("number_of_maxima")?)
        }
        // tsfresh's default query is None, which always gives NaN; tsrocket's
        // zero-length query does the same. A real query isn't in the name.
        "query_similarity_count" => match p.take("query")? {
            "None" => format!("query_similarity_count-0-{}", p.take("threshold")?),
            _ => return None,
        },
        // Not translatable: tsfresh's matrix_profile picks its own window, and
        // linear_trend_timewise's name lacks the sample period tsrocket needs.
        _ => return None,
    };
    p.0.is_empty().then_some(out)
}

// ─── TSFEL ──────────────────────────────────────────────────────────────────

/// TSFEL features with one output column, at TSFEL's default parameters.
const TSFEL_SINGLE: &[(&str, &str)] = &[
    ("Absolute energy", "energy"),
    ("Average power", "average_power"),
    ("Entropy", "entropy"),
    ("Histogram mode", "hist_mode-10"),
    ("Interquartile range", "iqr"),
    ("Kurtosis", "biased_fisher_kurtosis"),
    ("Max", "max_value"),
    ("Mean", "mean"),
    ("Mean absolute deviation", "mad"),
    ("Median", "median"),
    ("Median absolute deviation", "median_abs_deviation"),
    ("Min", "min_value"),
    ("Peak to peak distance", "pk_pk_distance"),
    ("Root mean square", "rms"),
    ("Skewness", "biased_skewness"),
    ("Standard deviation", "std_dev"),
    ("Variance", "variance"),
    ("Area under the curve", "auc"),
    ("Autocorrelation", "autocorrelation"),
    ("Centroid", "calc_centroid"),
    ("Lempel-Ziv complexity", "lempel_ziv"),
    ("Mean absolute diff", "mean_abs_change"),
    ("Mean diff", "mean_change"),
    ("Median absolute diff", "median_abs_diff"),
    ("Median diff", "median_diff"),
    ("Negative turning points", "negative_turning"),
    ("Neighbourhood peaks", "neighbourhood_peaks-10"),
    ("Positive turning points", "positive_turning"),
    ("Signal distance", "signal_distance"),
    ("Slope", "slope"),
    ("Sum absolute diff", "abs_sum_change"),
    ("Zero crossing rate", "zero_cross"),
    ("Fundamental frequency", "fundamental_frequency"),
    ("Human range energy", "human_range_energy"),
    ("Max power spectrum", "max_power_spectrum"),
    ("Maximum frequency", "max_frequency"),
    ("Median frequency", "median_frequency"),
    ("Power bandwidth", "power_bandwidth"),
    ("Spectral centroid", "spectral_centroid"),
    ("Spectral decrease", "spectral_decrease"),
    ("Spectral distance", "spectral_distance"),
    ("Spectral entropy", "spectral_entropy"),
    ("Spectral kurtosis", "spectral_kurtosis"),
    (
        "Spectral positive turning points",
        "spectral_positive_turning",
    ),
    ("Spectral roll-off", "spectral_roll_off"),
    ("Spectral roll-on", "spectral_roll_on"),
    ("Spectral skewness", "spectral_skewness"),
    ("Spectral slope", "spectral_slope"),
    ("Spectral spread", "spectral_spread"),
    ("Spectral variation", "spectral_variation"),
    ("Wavelet entropy", "wavelet_entropy"),
    ("Detrended fluctuation analysis", "dfa"),
    ("Higuchi fractal dimension", "higuchi_fd"),
    ("Hurst exponent", "hurst_exponent"),
    ("Maximum fractal length", "maximum_fractal_length"),
    ("Petrosian fractal dimension", "petrosian_fractal_dimension"),
    ("Multiscale entropy", "mse-3"),
];

/// TSFEL's default `percentile` for the ECDF Percentile features.
const TSFEL_PERCENTILES: [&str; 2] = ["0.2", "0.8"];
/// TSFEL's default `max_width`: wavelet scales 1..=9.
const TSFEL_WAVELET_SCALES: usize = 9;
/// TSFEL's default spectrogram `bins`.
const TSFEL_SPECTROGRAM_BINS: usize = 32;

fn tsfel(s: &str, fs: Option<f32>) -> Option<String> {
    // The channel prefix is optional and may itself contain '_'.
    std::iter::once(s)
        .chain(s.match_indices('_').map(|(i, _)| &s[i + 1..]))
        .find_map(|rest| tsfel_feature(rest, fs))
}

fn tsfel_feature(s: &str, fs: Option<f32>) -> Option<String> {
    if let Some(&(_, name)) = TSFEL_SINGLE.iter().find(|(tsfel, _)| *tsfel == s) {
        return Some(name.into());
    }
    let (feature, suffix) = s.rsplit_once('_')?;
    if let Some(hz) = suffix.strip_suffix("Hz") {
        let (fs, hz) = (fs?, hz.parse::<f32>().ok()?);
        return match feature {
            "Wavelet absolute mean" => Some(format!("wavelet_abs_mean-{}", wavelet_index(hz, fs)?)),
            "Wavelet energy" => Some(format!("wavelet_energy-{}", wavelet_index(hz, fs)?)),
            "Wavelet standard deviation" => Some(format!("wavelet_std-{}", wavelet_index(hz, fs)?)),
            "Wavelet variance" => Some(format!("wavelet_var-{}", wavelet_index(hz, fs)?)),
            "Spectrogram mean coefficient" => Some(format!(
                "spectrogram_mean_coeff-{}",
                spectrogram_index(hz, fs)?
            )),
            _ => None,
        };
    }
    let i: usize = suffix.parse().ok()?;
    match feature {
        "ECDF" => Some(format!("ecdf-{}", i + 1)),
        "ECDF Percentile" => Some(format!("ecdf_percentile-{}", TSFEL_PERCENTILES.get(i)?)),
        "ECDF Percentile Count" => Some(format!(
            "ecdf_percentile_count-{}",
            TSFEL_PERCENTILES.get(i)?
        )),
        "LPCC" => Some(format!("lpcc-{i}")),
        "MFCC" => Some(format!("mfcc-{i}")),
        _ => None,
    }
}

/// TSFEL labels a frequency rounded to 2 decimals; accept it only if it is
/// within that rounding of `exact`.
fn matches_label(exact: f32, label: f32) -> bool {
    (exact - label).abs() <= 0.0051
}

/// Index of the mexh wavelet scale whose centre frequency TSFEL labels `hz`:
/// scale k = index + 1 has centre frequency fs / (4 k).
fn wavelet_index(hz: f32, fs: f32) -> Option<usize> {
    (0..TSFEL_WAVELET_SCALES).find(|&i| matches_label(fs / (4.0 * (i + 1) as f32), hz))
}

/// Index of the spectrogram bin TSFEL labels `hz`: bin i sits at
/// i * fs / (2 (bins - 1)).
fn spectrogram_index(hz: f32, fs: f32) -> Option<usize> {
    let step = fs / (2 * (TSFEL_SPECTROGRAM_BINS - 1)) as f32;
    (0..TSFEL_SPECTROGRAM_BINS).find(|&i| matches_label(i as f32 * step, hz))
}
