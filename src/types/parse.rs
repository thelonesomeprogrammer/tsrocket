use super::feature::{AdfAttr, AggAttr, AggFunc, Feature, FftAggType, FftAttr};

// ─── Unit features: one line each ──────────────────────────────────────────
//
// `Variant => "canonical_name", ["alias", ...];`
// The canonical name is the output column name and is always accepted when
// parsing; aliases are additionally accepted (tsfresh/TSFEL spellings).
// Features with parameters are parsed in `from_str` / named in `name()` below.

macro_rules! unit_features {
    ($($variant:ident => $name:literal $(, [$($alias:literal),* $(,)?])?;)*) => {
        fn parse_unit(s: &str) -> Option<Feature> {
            match s {
                $($name $($(| $alias)*)? => Some(Feature::$variant),)*
                _ => None,
            }
        }

        /// Canonical name of every unit variant, for "did you mean" hints.
        const UNIT_NAMES: &[&str] = &[$($name),*];

        /// Every unit variant, for the coverage/round-trip tests.
        #[cfg(test)]
        pub(crate) const UNIT_FEATURES: &[Feature] = &[$(Feature::$variant),*];

        impl Feature {
            fn unit_name(&self) -> Option<&'static str> {
                match self {
                    $(Feature::$variant => Some($name),)*
                    _ => None,
                }
            }
        }
    };
}

unit_features! {
    TotalSum => "total_sum", ["value__sum_values"];
    Mean => "mean", ["value__mean"];
    Variance => "variance", ["value__variance"];
    Std => "std_dev", ["std", "value__standard_deviation"];
    Min => "min_value", ["min", "value__minimum"];
    Max => "max_value", ["max", "value__maximum"];
    Median => "median", ["value__median"];
    MedianAbsDeviation => "median_abs_deviation";
    Skew => "skewness", ["skew", "value__skewness"];
    BiasedSkew => "biased_skewness";
    UnbiasedFisherKurtosis => "kurtosis", ["value__kurtosis", "unbiased_fisher_kurtosis"];
    BiasedFisherKurtosis => "biased_fisher_kurtosis";
    Mad => "mad", ["mean_abs_deviation"];
    Iqr => "iqr";
    Entropy => "entropy";
    SampleEntropy => "sample_entropy";
    TsfreshSampleEntropy => "tsfresh_sample_entropy", ["value__sample_entropy"];
    HiguchiFd => "higuchi_fd";
    Dfa => "dfa";
    HurstExponent => "hurst_exponent";
    MaximumFractalLength => "maximum_fractal_length";
    PetrosianFractalDimension => "petrosian_fractal_dimension";
    Energy => "energy";
    Rms => "rms";
    RootMeanSquare => "root_mean_square";
    LempelZiv => "lempel_ziv";
    ZeroCrossingRate => "zero_crossing_rate";
    PeakCount => "peak_count";
    NegativeTurning => "negative_turning";
    PositiveTurning => "positive_turning";
    AutocorrLag1 => "autocorr_lag1", ["centered_autocorr_lag1"];
    AutocorrFirst1e => "autocorrelation", ["autocorr_first_1e"];
    MeanAbsChange => "mean_abs_change";
    MeanChange => "mean_change", ["mean_diff"];
    MedianDiff => "median_diff";
    MedianAbsDiff => "median_abs_diff";
    CidCe => "cid_ce";
    CidCeNormalized => "cid_ce_normalized";
    Slope => "slope";
    Intercept => "intercept";
    AbsSumChange => "abs_sum_change";
    CountAboveMean => "count_above_mean";
    CountBelowMean => "count_below_mean";
    LongestStrikeAboveMean => "longest_strike_above_mean";
    LongestStrikeBelowMean => "longest_strike_below_mean";
    VariationCoefficient => "variation_coefficient";
    Auc => "auc";
    SlopeSignChange => "slope_sign_change";
    TurningPoints => "turning_points";
    ZeroCrossingMean => "zero_crossing_mean";
    ZeroCrossingStd => "zero_crossing_std";
    AbsMax => "abs_max", ["value__absolute_maximum"];
    FirstLocMax => "first_loc_max", ["value__first_location_of_maximum"];
    LastLocMax => "last_loc_max", ["value__last_location_of_maximum"];
    FirstLocMin => "first_loc_min", ["value__first_location_of_minimum"];
    LastLocMin => "last_loc_min", ["value__last_location_of_minimum"];
    BenfordCorrelation => "benford_correlation", ["value__benford_correlation"];
    SumOfReoccurringValues => "sum_of_reoccurring_values", ["value__sum_of_reoccurring_values"];
    SumOfReoccurringDataPoints => "sum_of_reoccurring_data_points", ["value__sum_of_reoccurring_data_points"];
    PercentageOfReoccurringDatapointsToAllDatapoints => "percentage_of_reoccurring_datapoints_to_all_datapoints", ["value__percentage_of_reoccurring_datapoints_to_all_datapoints"];
    PercentageOfReoccurringValuesToAllValues => "percentage_of_reoccurring_values_to_all_values", ["value__percentage_of_reoccurring_values_to_all_values"];
    RatioValueNumberToTimeSeriesLength => "ratio_value_number_to_time_series_length", ["value__ratio_value_number_to_time_series_length"];
    Length => "length", ["value__length"];
    VarianceLargerThanStandardDeviation => "variance_larger_than_standard_deviation", ["value__variance_larger_than_standard_deviation"];
    SpectralCentroid => "spectral_centroid";
    SpectralDistance => "spectral_distance";
    SpectralDecrease => "spectral_decrease";
    SpectralSlope => "spectral_slope";
    SpectralSpread => "spectral_spread";
    SpectralEntropy => "spectral_entropy";
    SpectralRollOn => "spectral_roll_on";
    SpectralRollOff => "spectral_roll_off";
    MaxFrequency => "max_frequency";
    MedianFrequency => "median_frequency";
    FundamentalFrequency => "fundamental_frequency";
    PowerBandwidth => "power_bandwidth";
    SpectralPositiveTurning => "spectral_positive_turning";
    SpectralVariation => "spectral_variation";
    SpectralSkewness => "spectral_skewness";
    SpectralKurtosis => "spectral_kurtosis";
    SignalDistance => "signal_distance";
    PkPkDistance => "pk_pk_distance";
    ZeroCross => "zero_cross";
    MaxPowerSpectrum => "max_power_spectrum";
    MeanSecondDerivativeCentral => "mean_second_derivative_central";
    HasDuplicateMax => "has_duplicate_max";
    HasDuplicateMin => "has_duplicate_min";
    HasDuplicate => "has_duplicate";
    WaveletEntropy => "wavelet_entropy";
}

// ─── Meta features ──────────────────────────────────────────────────────────

/// Options given in the feature list that configure the engine instead of
/// producing an output column. Every engine accepts them (so one list works
/// everywhere) and ignores the ones that do not apply to it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetaFeatures {
    /// `fresh-N`: the sliding engine rebuilds its sliding DFT from a fresh FFT
    /// at least every N samples (default: every `window_size`). Repeated:
    /// the smallest N wins. Ignored by the static and expanding engines,
    /// which always run a fresh FFT.
    pub fresh_fft_every: Option<usize>,
}

/// Split meta features (see `MetaFeatures`) off a feature list, returning the
/// remaining feature names in order. Errors on a malformed meta feature.
pub fn split_meta(names: Vec<String>) -> Result<(Vec<String>, MetaFeatures), String> {
    let mut meta = MetaFeatures::default();
    let mut rest = Vec::with_capacity(names.len());
    for name in names {
        if let Some(arg) = name.strip_prefix("fresh-") {
            let n = arg
                .parse::<usize>()
                .ok()
                .filter(|&n| n >= 1)
                .ok_or_else(|| format!("{name}: expected fresh-N with N >= 1"))?;
            meta.fresh_fft_every = Some(meta.fresh_fft_every.map_or(n, |m| m.min(n)));
        } else {
            rest.push(name);
        }
    }
    Ok((rest, meta))
}

// ─── FromStr ────────────────────────────────────────────────────────────────

impl std::str::FromStr for Feature {
    type Err = String;
    /// A tsrocket, tsfresh or TSFEL feature name. TSFEL names with a
    /// frequency suffix need the sampling frequency: see `parse_with_fs`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s, None)
    }
}

impl Feature {
    /// Like `FromStr`, and also accepts TSFEL names whose suffix is a
    /// frequency (`0_Wavelet energy_12.5Hz`), which depend on `fs`.
    pub fn parse_with_fs(s: &str, fs: f32) -> Result<Self, String> {
        Self::parse(s, Some(fs))
    }

    fn parse(s: &str, fs: Option<f32>) -> Result<Self, String> {
        parse_native(s)
            .or_else(|| super::compat::translate(s, fs).and_then(|name| parse_native(&name)))
            .ok_or_else(|| unknown_feature(s))
    }
}

/// A tsrocket feature name (canonical or alias), without tsfresh/TSFEL
/// translation.
fn parse_native(s: &str) -> Option<Feature> {
    {
        // 1. Unit features (table below)
        if let Some(f) = parse_unit(s) {
            return Some(f);
        }

        // 2. Exact matches with fixed parameters
        match s {
            name if name.starts_with("energy_ratio_by_chunks_num_segments_") => {
                let parts: Vec<&str> = name.split("__").collect();
                if parts.len() == 2 {
                    let p1 = parts[0].strip_prefix("energy_ratio_by_chunks_num_segments_");
                    let p2 = parts[1].strip_prefix("segment_focus_");
                    if let (Some(num_s), Some(focus_s)) = (p1, p2) {
                        if let (Ok(num), Ok(focus)) = (num_s.parse::<u16>(), focus_s.parse::<u16>())
                        {
                            if num > 0 && focus < num {
                                return Some(Feature::EnergyRatioByChunks(num, focus));
                            }
                        }
                    }
                }
            }
            "augmented_dickey_fuller-teststat" => {
                return Some(Feature::AugmentedDickeyFuller(AdfAttr::TestStat));
            }
            "augmented_dickey_fuller-pvalue" => {
                return Some(Feature::AugmentedDickeyFuller(AdfAttr::PValue));
            }
            "augmented_dickey_fuller-usedlag" => {
                return Some(Feature::AugmentedDickeyFuller(AdfAttr::UsedLag));
            }
            "human_range_energy" => {
                return Some(Feature::HumanRangeEnergy(None));
            }
            "hist_mode" => {
                return Some(Feature::HistMode(10)); // TSFEL default nbins=10
            }
            // TSFEL's neighbourhood_peaks is tsfresh's number_peaks under another name.
            "neighbourhood_peaks" => {
                return Some(Feature::NumberPeaks(10)); // TSFEL default n=10
            }
            "average_power" => {
                return Some(Feature::AveragePower(None));
            }
            "calc_centroid" => {
                return Some(Feature::CalcCentroid(None));
            }
            _ => {}
        }

        // 3. Parameterized features (prefix-based)
        if let Some(f) = parse_parameterized(s) {
            return Some(f);
        }

        // 4. Legacy tsfresh format
        if let Some(f) = parse_legacy_format(s) {
            return Some(f);
        }

        if s.starts_with("query_similarity_count-") {
            let parts: Vec<&str> = s.split("-").collect();
            if parts.len() == 3 {
                if let (Ok(l), Ok(t)) = (parts[1].parse::<u16>(), parts[2].parse::<f32>()) {
                    return Some(Feature::QuerySimilarityCount(l, t.to_bits()));
                }
            }
        }
        if s.starts_with("matrix_profile-") {
            let parts: Vec<&str> = s.split("-").collect();
            if parts.len() == 3 {
                if let Ok(l) = parts[1].parse::<u16>() {
                    if let Some(agg) = parse_agg_func(parts[2]) {
                        return Some(Feature::MatrixProfile(l, agg));
                    }
                }
            }
        }
        None
    }
}

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Prefixes of the `name-param...` features, for "did you mean" hints.
const PARAMETERIZED_NAMES: &[&str] = &[
    "agg_autocorrelation",
    "agg_linear_trend",
    "approx_entropy",
    "ar_coefficient",
    "augmented_dickey_fuller",
    "autocorr",
    "average_power",
    "c3",
    "calc_centroid",
    "change_quantiles",
    "count_above",
    "count_below",
    "ecdf",
    "ecdf_percentile",
    "ecdf_percentile_count",
    "ecdf_slope",
    "fft_aggregated",
    "fft_coeff",
    "fourier_entropy",
    "friedrich_coefficients",
    "hist_mode",
    "human_range_energy",
    "index_mass_quantile",
    "large_standard_deviation",
    "lempel_ziv_complexity",
    "linear_trend",
    "linear_trend_timewise",
    "lpcc",
    "matrix_profile",
    "max_langevin_fixed_point",
    "mean_n_absolute_max",
    "mfcc",
    "mse",
    "neighbourhood_peaks",
    "paa",
    "partial_autocorr",
    "permutation_entropy",
    "quantile",
    "query_similarity_count",
    "range_count",
    "ratio_beyond_r_sigma",
    "spectrogram",
    "spectrogram_mean_coeff",
    "symmetry_looking",
    "time_reversal_asymmetry",
    "value_count",
    "wavelet",
    "wavelet_abs_mean",
    "wavelet_energy",
    "wavelet_std",
    "wavelet_var",
];

const FEATURE_LIST_URL: &str =
    "https://github.com/thelonesomeprogrammer/tsrocket/blob/master/docs/features.md";

fn unknown_feature(s: &str) -> String {
    let head = s.split('-').next().unwrap_or(s);
    let best = UNIT_NAMES
        .iter()
        .map(|&n| (edit_distance(s, n), n.to_string()))
        .chain(
            PARAMETERIZED_NAMES
                .iter()
                .map(|&n| (edit_distance(head, n), format!("{n}-..."))),
        )
        .min();
    match best {
        Some((d, name)) if d > 0 && d <= (s.len() / 4).max(2) => {
            format!(
                "Unknown feature: {s} (did you mean `{name}`?). Valid names: {FEATURE_LIST_URL}"
            )
        }
        _ => format!("Unknown feature: {s}. Valid names: {FEATURE_LIST_URL}"),
    }
}

/// Levenshtein distance between two short ASCII-ish names.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diag = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let next = (diag + usize::from(ca != cb))
                .min(row[j] + 1)
                .min(row[j + 1] + 1);
            diag = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}

// ─── Parameterized parsers ──────────────────────────────────────────────────

fn parse_parameterized(s: &str) -> Option<Feature> {
    if let Some(arg) = s.strip_prefix("mse-") {
        let parts: Vec<&str> = arg.split('-').collect();
        if parts.is_empty() || parts.len() > 2 {
            return None;
        }
        let m: u8 = parts[0].parse().ok()?;
        if m < 1 {
            return None;
        }
        let maxscale: u16 = if parts.len() == 2 {
            let val: u16 = parts[1].parse().ok()?;
            if val < 1 {
                return None;
            }
            val
        } else {
            0
        };
        return Some(Feature::Mse(m, maxscale));
    }
    if let Some(arg) = s.strip_prefix("human_range_energy-") {
        return Some(Feature::HumanRangeEnergy(Some(parse_fs(arg)?)));
    }
    if let Some(arg) = s.strip_prefix("average_power-") {
        return Some(Feature::AveragePower(Some(parse_fs(arg)?)));
    }
    if let Some(arg) = s.strip_prefix("lempel_ziv_complexity-") {
        let bins: u16 = arg.trim().parse().ok()?;
        // Symbols are stored as u8.
        if (2..=256).contains(&bins) {
            return Some(Feature::LempelZivComplexity(bins));
        } else {
            return None;
        }
    }
    if let Some(arg) = s.strip_prefix("ecdf_slope-") {
        let (p1, p2) = arg.split_once('-')?;
        let p_init: f32 = p1.parse().ok()?;
        let p_end: f32 = p2.parse().ok()?;
        if p_init > 0.0 && p_init < p_end && p_end <= 1.0 {
            return Some(Feature::EcdfSlope(p_init.to_bits(), p_end.to_bits()));
        } else {
            return None;
        }
    }
    if let Some(arg) = s.strip_prefix("hist_mode-") {
        let nbins: u16 = arg.parse().ok()?;
        if nbins == 0 {
            return None;
        }
        return Some(Feature::HistMode(nbins));
    }
    if let Some(arg) = s.strip_prefix("neighbourhood_peaks-") {
        let n: u16 = arg.parse().ok()?;
        if n == 0 {
            return None;
        }
        return Some(Feature::NumberPeaks(n));
    }
    if let Some(arg) = s.strip_prefix("ecdf_percentile_count-") {
        let p: f32 = arg.parse().ok()?;
        if p > 0.0 && p <= 1.0 {
            return Some(Feature::EcdfPercentileCount(p.to_bits()));
        } else {
            return None;
        }
    }
    if let Some(arg) = s.strip_prefix("ecdf_percentile-") {
        let p: f32 = arg.parse().ok()?;
        if p > 0.0 && p <= 1.0 {
            return Some(Feature::EcdfPercentile(p.to_bits()));
        } else {
            return None;
        }
    }
    if let Some(arg) = s.strip_prefix("lpcc-") {
        return Some(Feature::Lpcc(arg.trim().parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("mfcc-") {
        return Some(Feature::Mfcc(arg.trim().parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("wavelet_energy-") {
        return Some(Feature::WaveletEnergy(arg.trim().parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("wavelet_abs_mean-") {
        return Some(Feature::WaveletAbsMean(arg.trim().parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("wavelet_std-") {
        return Some(Feature::WaveletStd(arg.trim().parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("wavelet_var-") {
        return Some(Feature::WaveletVar(arg.trim().parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("paa-") {
        let (n, m) = arg.split_once('-')?;
        let total = n.parse::<u16>().ok()?;
        let index = m.parse::<u16>().ok()?;
        if index >= total {
            return None;
        }
        return Some(Feature::Paa(total, index));
    }
    if let Some(arg) = s.strip_prefix("c3-") {
        return Some(Feature::C3(arg.trim().parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("autocorr-") {
        return Some(Feature::Autocorr(arg.parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("agg_autocorrelation-") {
        let parts: Vec<&str> = arg.split('-').collect();
        if parts.len() != 2 {
            return None;
        }
        let func = parse_agg_func(parts[0])?;
        let maxlag: u16 = parts[1].parse().ok()?;
        return Some(Feature::AggAutocorrelation(func, maxlag));
    }
    if let Some(arg) = s.strip_prefix("partial_autocorr-") {
        return Some(Feature::PartialAutocorr(arg.parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("time_reversal_asymmetry-") {
        return Some(Feature::TimeReversalAsymmetry(arg.parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("fft_coeff-") {
        let (coeff_s, attr_s) = arg.split_once('-')?;
        let coeff: u16 = coeff_s.parse().ok()?;
        let attr = match attr_s {
            "real" => FftAttr::Real,
            "imag" => FftAttr::Imag,
            "abs" => FftAttr::Abs,
            "angle" => FftAttr::Angle,
            _ => return None,
        };
        return Some(Feature::FftCoefficient(coeff, attr));
    }
    if let Some(arg) = s.strip_prefix("approx_entropy-") {
        let (m_s, r_s) = arg.split_once('-')?;
        let m: u8 = m_s.parse().ok()?;
        let r: f32 = r_s.parse().ok()?;
        return Some(Feature::ApproxEntropy(m, r.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("linear_trend_timewise-") {
        let (attr_s, period_s) = arg.split_once('-')?;
        let attr = parse_agg_attr(attr_s)?;
        let period: f32 = period_s.parse().ok()?;
        if !(period.is_finite() && period > 0.0) {
            return None;
        }
        return Some(Feature::LinearTrendTimewise(attr, period.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("linear_trend-") {
        let attr = parse_agg_attr(arg)?;
        return Some(Feature::LinearTrend(attr));
    }
    if let Some(arg) = s.strip_prefix("agg_linear_trend-") {
        let parts: Vec<&str> = arg.split('-').collect();
        if parts.len() != 3 {
            return None;
        }
        let attr = parse_agg_attr(parts[0])?;
        let chunk_len: u16 = parts[1].parse().ok()?;
        if chunk_len == 0 {
            return None;
        }
        let func = parse_agg_func(parts[2])?;
        return Some(Feature::AggLinearTrend(attr, chunk_len, func));
    }
    if let Some(arg) = s.strip_prefix("quantile-") {
        let q: f32 = arg.parse().ok()?;
        return Some(Feature::Quantile(q.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("index_mass_quantile-") {
        let q: f32 = arg.parse().ok()?;
        return Some(Feature::IndexMassQuantile(q.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("ar_coefficient-") {
        let (k_s, p_s) = arg.split_once('-')?;
        // ar_coefficient-<order k>-<coefficient p>; p = 0 is the intercept.
        let k: u16 = k_s.parse().ok()?;
        let p: u16 = p_s.parse().ok()?;
        if k == 0 || p > k {
            return None;
        }
        return Some(Feature::ArCoefficient(k, p));
    }
    if let Some(arg) = s.strip_prefix("friedrich_coefficients-") {
        let parts: Vec<&str> = arg.split('-').collect();
        if parts.len() == 3 {
            let m: u8 = parts[0].parse().ok()?;
            let r: f32 = parts[1].parse().ok()?;
            let coeff: u16 = parts[2].parse().ok()?;
            return Some(Feature::FriedrichCoefficients(m, r.to_bits(), coeff));
        }
    }
    if let Some(arg) = s.strip_prefix("max_langevin_fixed_point-") {
        let (m_s, r_s) = arg.split_once('-')?;
        let m: u8 = m_s.parse().ok()?;
        let r: f32 = r_s.parse().ok()?;
        return Some(Feature::MaxLangevinFixedPoint(m, r.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("mean_n_absolute_max-") {
        return Some(Feature::MeanNAbsoluteMax(arg.parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("wavelet-") {
        let (w_s, f_s) = arg.split_once('-')?;
        let w: f32 = w_s.parse().ok()?;
        let f: u16 = f_s.parse().ok()?;
        return Some(Feature::WaveletFeatures(w.to_bits(), f));
    }
    if let Some(arg) = s.strip_prefix("spectrogram_mean_coeff-") {
        // spectrogram_mean_coeff-COEFF[-BINS], TSFEL default bins=32.
        let (c_s, b_s) = arg.split_once('-').unwrap_or((arg, "32"));
        let coeff: u16 = c_s.parse().ok()?;
        let bins: u16 = b_s.parse().ok()?;
        if bins < 2 || coeff >= bins {
            return None;
        }
        return Some(Feature::SpectrogramMeanCoeff(coeff, bins));
    }
    if let Some(arg) = s.strip_prefix("spectrogram-") {
        let (t_s, f_s) = arg.split_once('-')?;
        let t: u16 = t_s.parse().ok()?;
        let f: f32 = f_s.parse().ok()?;
        return Some(Feature::SpectrogramCoefficients(t, f.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("large_standard_deviation-") {
        let r: f32 = arg.parse().ok()?;
        return Some(Feature::LargeStandardDeviation(r.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("symmetry_looking-") {
        let r: f32 = arg.parse().ok()?;
        return Some(Feature::SymmetryLooking(r.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("ecdf-") {
        let d: u32 = arg.parse().ok()?;
        return Some(Feature::Ecdf(d));
    }
    if let Some(arg) = s.strip_prefix("calc_centroid-") {
        return Some(Feature::CalcCentroid(Some(parse_fs(arg)?)));
    }
    if let Some(arg) = s.strip_prefix("binned_entropy__max_bins_")
        && let Ok(bins) = arg.parse::<u32>()
    {
        if bins > 0 {
            return Some(Feature::BinnedEntropy(bins));
        } else {
            return None;
        }
    }
    if let Some(arg) = s.strip_prefix("fourier_entropy-")
        && let Ok(bins) = arg.parse::<u32>()
    {
        if bins > 0 {
            return Some(Feature::FourierEntropy(bins));
        } else {
            return None;
        }
    }
    if let Some(arg) = s.strip_prefix("fft_aggregated-") {
        let agg = match arg {
            "centroid" => FftAggType::Centroid,
            "variance" => FftAggType::Variance,
            "skew" => FftAggType::Skew,
            "kurtosis" => FftAggType::Kurtosis,
            _ => return None,
        };
        return Some(Feature::FftAggregated(agg));
    }
    if let Some(arg) = s.strip_prefix("ratio_beyond_r_sigma-") {
        let r: f32 = arg.parse().ok()?;
        return Some(Feature::RatioBeyondRSigma(r.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("count_above-") {
        let t: f32 = arg.parse().ok()?;
        if t.is_nan() {
            return None;
        }
        return Some(Feature::CountAbove(t.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("count_below-") {
        let t: f32 = arg.parse().ok()?;
        if t.is_nan() {
            return None;
        }
        return Some(Feature::CountBelow(t.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("range_count-") {
        // min and max are both floats and either may be negative, so the '-'
        // separating them can't be found with a plain split: try each '-' in
        // turn (skipping index 0, min's own sign) until both sides parse.
        for (idx, _) in arg.match_indices('-') {
            if idx == 0 {
                continue;
            }
            let (min_s, max_s) = (&arg[..idx], &arg[idx + 1..]);
            if let (Ok(min_v), Ok(max_v)) = (min_s.parse::<f32>(), max_s.parse::<f32>()) {
                if min_v.is_nan() || max_v.is_nan() || min_v > max_v {
                    return None;
                }
                return Some(Feature::RangeCount(min_v.to_bits(), max_v.to_bits()));
            }
        }
        return None;
    }
    None
}

// ─── Legacy tsfresh format parsers ──────────────────────────────────────────

fn parse_legacy_format(s: &str) -> Option<Feature> {
    if s.contains("number_crossing_m__m_") {
        let pos = s.rfind("__m_")?;
        let m: f32 = s[pos + 4..].parse().ok()?;
        return Some(Feature::NumberCrossingM(m.to_bits()));
    }
    if s.contains("number_peaks__n_") {
        let pos = s.find("n_")?;
        let n: u16 = s[pos + 2..].parse().ok()?;
        if n == 0 {
            return None;
        }
        return Some(Feature::NumberPeaks(n));
    }
    if s.starts_with("change_quantiles-") {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 5 {
            return None;
        }
        let ql = parts[1].parse::<f32>().ok()?.to_bits();
        let qh = parts[2].parse::<f32>().ok()?.to_bits();
        let isabs = match parts[3] {
            "True" | "true" => true,
            "False" | "false" => false,
            _ => return None,
        };
        let func = parse_agg_func(parts[4])?;
        return Some(Feature::ChangeQuantiles(ql, qh, isabs, func));
    }
    if let Some(arg) = s.strip_prefix("spkt_welch_density__coeff_") {
        let coeff: u16 = arg.parse().ok()?;
        return Some(Feature::SpktWelchDensity(coeff));
    }
    if let Some(attr_str) = s.strip_prefix("augmented_dickey_fuller-") {
        return match attr_str {
            "teststat" => Some(Feature::AugmentedDickeyFuller(AdfAttr::TestStat)),
            "pvalue" => Some(Feature::AugmentedDickeyFuller(AdfAttr::PValue)),
            "usedlag" => Some(Feature::AugmentedDickeyFuller(AdfAttr::UsedLag)),
            _ => None,
        };
    }

    if let Some(arg) = s.strip_prefix("number_cwt_peaks__n_") {
        let n: u16 = arg.parse().ok()?;
        if n == 0 {
            return None;
        }
        return Some(Feature::NumberCwtPeaks(n));
    }
    if s.starts_with("cwt_coefficients__coeff_") {
        if let Some(coeff_end) = s.find("__w_") {
            let coeff_str = &s[24..coeff_end];
            let coeff = coeff_str.parse::<u16>().ok()?;
            if let Some(w_end) = s.find("__widths_") {
                let w_str = &s[coeff_end + 4..w_end];
                let w = w_str.parse::<u16>().ok()?;
                let widths_str = &s[w_end + 9..];
                let cleaned = widths_str
                    .replace("(", "")
                    .replace(")", "")
                    .replace(" ", "");
                let parts: Vec<&str> = cleaned.split(',').collect();
                let mut widths = [0u16; 8];
                let len = parts.len().min(8) as u8;
                for i in 0..(len as usize) {
                    if let Ok(val) = parts[i].parse::<u16>() {
                        widths[i] = val;
                    }
                }
                return Some(Feature::CwtCoefficients(widths, len, coeff, w));
            }
        }
    }

    if let Some(arg) = s.strip_prefix("permutation_entropy-") {
        let parts: Vec<&str> = arg.split('-').collect();
        if parts.len() == 2 {
            if let (Ok(tau), Ok(dim)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                return Some(Feature::PermutationEntropy(tau, dim));
            }
        }
    }

    if let Some(arg) = s.strip_prefix("value_count-") {
        if let Ok(v) = arg.parse::<f32>() {
            return Some(Feature::ValueCount(v.to_bits()));
        }
    }
    if let Some(arg) = s.strip_prefix("large_standard_deviation__r_") {
        let r: f32 = arg.parse().ok()?;
        return Some(Feature::LargeStandardDeviation(r.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("symmetry_looking__r_") {
        let r: f32 = arg.parse().ok()?;
        return Some(Feature::SymmetryLooking(r.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("binned_entropy__max_bins_")
        && let Ok(bins) = arg.parse::<u32>()
    {
        if bins > 0 {
            return Some(Feature::BinnedEntropy(bins));
        } else {
            return None;
        }
    }
    None
}

// ─── Shared sub-enum parsers ────────────────────────────────────────────────

fn parse_agg_attr(s: &str) -> Option<AggAttr> {
    match s {
        "slope" => Some(AggAttr::Slope),
        "intercept" => Some(AggAttr::Intercept),
        "stderr" => Some(AggAttr::Stderr),
        "rvalue" => Some(AggAttr::RValue),
        "pvalue" => Some(AggAttr::PValue),
        _ => None,
    }
}

fn parse_agg_func(s: &str) -> Option<AggFunc> {
    match s {
        "max" => Some(AggFunc::Max),
        "min" => Some(AggFunc::Min),
        "mean" => Some(AggFunc::Mean),
        "var" => Some(AggFunc::Var),
        "median" => Some(AggFunc::Median),
        _ => None,
    }
}

// ─── Feature::name() ───────────────────────────────────────────────────────

impl Feature {
    pub fn name(&self) -> String {
        if let Some(name) = self.unit_name() {
            return name.to_string();
        }
        match self {
            Feature::Mse(m, maxscale) => {
                if *maxscale == 0 {
                    format!("mse-{m}")
                } else {
                    format!("mse-{m}-{maxscale}")
                }
            }
            Feature::EcdfPercentile(p) => format!("ecdf_percentile-{}", f32::from_bits(*p)),
            Feature::EcdfPercentileCount(p) => {
                format!("ecdf_percentile_count-{}", f32::from_bits(*p))
            }
            Feature::EcdfSlope(p_init, p_end) => format!(
                "ecdf_slope-{}-{}",
                f32::from_bits(*p_init),
                f32::from_bits(*p_end)
            ),
            Feature::EnergyRatioByChunks(num, focus) => format!(
                "energy_ratio_by_chunks_num_segments_{}__segment_focus_{}",
                num, focus
            ),
            Feature::NumberCrossingM(m) => format!("number_crossing_m__m_{}", f32::from_bits(*m)),
            Feature::NumberPeaks(n) => format!("number_peaks__n_{}", n),
            Feature::C3(lag) => format!("c3-{}", lag),
            Feature::Paa(total, index) => format!("paa-{}-{}", total, index),
            Feature::Autocorr(lag) => format!("autocorr-{}", lag),
            Feature::AggAutocorrelation(func, maxlag) => {
                let func_str = match func {
                    AggFunc::Max => "max",
                    AggFunc::Min => "min",
                    AggFunc::Mean => "mean",
                    AggFunc::Var => "var",
                    AggFunc::Median => "median",
                };
                format!("agg_autocorrelation-{}-{}", func_str, maxlag)
            }
            Feature::PartialAutocorr(lag) => format!("partial_autocorr-{}", lag),
            Feature::TimeReversalAsymmetry(lag) => format!("time_reversal_asymmetry-{}", lag),
            Feature::LempelZivComplexity(bins) => format!("lempel_ziv_complexity-{}", bins),
            Feature::FftCoefficient(coeff, attr) => {
                let attr_str = match attr {
                    FftAttr::Real => "real",
                    FftAttr::Imag => "imag",
                    FftAttr::Abs => "abs",
                    FftAttr::Angle => "angle",
                };
                format!("fft_coeff-{}-{}", coeff, attr_str)
            }
            Feature::BinnedEntropy(bins) => format!("binned_entropy__max_bins_{}", bins),
            Feature::FourierEntropy(bins) => format!("fourier_entropy-{}", bins),
            Feature::FftAggregated(agg) => {
                let agg_str = match agg {
                    FftAggType::Centroid => "centroid",
                    FftAggType::Variance => "variance",
                    FftAggType::Skew => "skew",
                    FftAggType::Kurtosis => "kurtosis",
                };
                format!("fft_aggregated-{}", agg_str)
            }
            Feature::ApproxEntropy(m, r_bits) => {
                format!("approx_entropy-{}-{}", m, f32::from_bits(*r_bits))
            }
            Feature::PermutationEntropy(tau, dim) => {
                format!("permutation_entropy-{}-{}", tau, dim)
            }
            Feature::ValueCount(val_bits) => {
                format!("value_count-{}", f32::from_bits(*val_bits))
            }
            Feature::LinearTrend(attr) => {
                let attr_str = match attr {
                    AggAttr::Slope => "slope",
                    AggAttr::Intercept => "intercept",
                    AggAttr::Stderr => "stderr",
                    AggAttr::RValue => "rvalue",
                    AggAttr::PValue => "pvalue",
                };
                format!("linear_trend-{}", attr_str)
            }
            Feature::LinearTrendTimewise(attr, period_bits) => {
                let attr_str = match attr {
                    AggAttr::Slope => "slope",
                    AggAttr::Intercept => "intercept",
                    AggAttr::Stderr => "stderr",
                    AggAttr::RValue => "rvalue",
                    AggAttr::PValue => "pvalue",
                };
                format!(
                    "linear_trend_timewise-{}-{}",
                    attr_str,
                    f32::from_bits(*period_bits)
                )
            }
            Feature::AggLinearTrend(attr, chunk_len, func) => {
                let attr_str = match attr {
                    AggAttr::Slope => "slope",
                    AggAttr::Intercept => "intercept",
                    AggAttr::Stderr => "stderr",
                    AggAttr::RValue => "rvalue",
                    AggAttr::PValue => "pvalue",
                };
                let func_str = match func {
                    AggFunc::Max => "max",
                    AggFunc::Min => "min",
                    AggFunc::Mean => "mean",
                    AggFunc::Var => "var",
                    AggFunc::Median => "median",
                };
                format!("agg_linear_trend-{}-{}-{}", attr_str, chunk_len, func_str)
            }
            Feature::ChangeQuantiles(ql_bits, qh_bits, isabs, func) => {
                let func_str = match func {
                    AggFunc::Max => "max",
                    AggFunc::Min => "min",
                    AggFunc::Mean => "mean",
                    AggFunc::Var => "var",
                    AggFunc::Median => "median",
                };
                format!(
                    "change_quantiles-{}-{}-{}-{}",
                    f32::from_bits(*ql_bits),
                    f32::from_bits(*qh_bits),
                    if *isabs { "True" } else { "False" },
                    func_str
                )
            }
            Feature::Quantile(q_bits) => format!("quantile-{}", f32::from_bits(*q_bits)),
            Feature::IndexMassQuantile(q_bits) => {
                format!("index_mass_quantile-{}", f32::from_bits(*q_bits))
            }
            Feature::MaxLangevinFixedPoint(m, r_bits) => {
                format!("max_langevin_fixed_point-{}-{}", m, f32::from_bits(*r_bits))
            }
            Feature::QuerySimilarityCount(l, t) => {
                format!("query_similarity_count-{}-{}", l, f32::from_bits(*t))
            }
            Feature::MatrixProfile(l, agg) => {
                let agg_str = match agg {
                    AggFunc::Min => "min",
                    AggFunc::Max => "max",
                    AggFunc::Mean => "mean",
                    AggFunc::Var => "var",
                    AggFunc::Median => "median",
                };
                format!("matrix_profile-{}-{}", l, agg_str)
            }
            Feature::MeanNAbsoluteMax(n) => format!("mean_n_absolute_max-{}", n),
            Feature::HistMode(nbins) => format!("hist_mode-{}", nbins),
            Feature::HumanRangeEnergy(fs) => with_fs("human_range_energy", *fs),
            Feature::AveragePower(fs) => with_fs("average_power", *fs),
            Feature::WaveletFeatures(w_bits, f) => {
                format!("wavelet-{}-{}", f32::from_bits(*w_bits), f)
            }
            Feature::SpectrogramMeanCoeff(coeff, 32) => format!("spectrogram_mean_coeff-{}", coeff),
            Feature::SpectrogramMeanCoeff(coeff, bins) => {
                format!("spectrogram_mean_coeff-{}-{}", coeff, bins)
            }
            Feature::SpectrogramCoefficients(t, f_bits) => {
                format!("spectrogram-{}-{}", t, f32::from_bits(*f_bits))
            }
            Feature::LargeStandardDeviation(r_bits) => {
                format!("large_standard_deviation-{}", f32::from_bits(*r_bits))
            }
            Feature::SymmetryLooking(r_bits) => {
                format!("symmetry_looking-{}", f32::from_bits(*r_bits))
            }
            Feature::RatioBeyondRSigma(r_bits) => {
                format!("ratio_beyond_r_sigma-{}", f32::from_bits(*r_bits))
            }
            Feature::CountAbove(t_bits) => format!("count_above-{}", f32::from_bits(*t_bits)),
            Feature::CountBelow(t_bits) => format!("count_below-{}", f32::from_bits(*t_bits)),
            Feature::RangeCount(min_bits, max_bits) => format!(
                "range_count-{}-{}",
                f32::from_bits(*min_bits),
                f32::from_bits(*max_bits)
            ),
            Feature::Lpcc(idx) => format!("lpcc-{}", idx),
            Feature::Ecdf(d) => format!("ecdf-{}", d),
            Feature::CalcCentroid(fs) => with_fs("calc_centroid", *fs),
            Feature::Mfcc(idx) => format!("mfcc-{}", idx),
            Feature::WaveletEnergy(idx) => format!("wavelet_energy-{}", idx),
            Feature::WaveletAbsMean(idx) => format!("wavelet_abs_mean-{}", idx),
            Feature::WaveletStd(idx) => format!("wavelet_std-{}", idx),
            Feature::WaveletVar(idx) => format!("wavelet_var-{}", idx),

            Feature::SpktWelchDensity(coeff) => format!("spkt_welch_density__coeff_{}", coeff),
            Feature::CwtCoefficients(widths, len, coeff, w) => {
                let mut w_str = String::from("(");
                for i in 0..*len as usize {
                    if i > 0 {
                        w_str.push_str(", ");
                    }
                    w_str.push_str(&widths[i].to_string());
                }
                w_str.push(')');
                format!(
                    "cwt_coefficients__coeff_{}__w_{}__widths_{}",
                    coeff, w, w_str
                )
            }
            Feature::NumberCwtPeaks(n) => format!("number_cwt_peaks__n_{}", n),
            Feature::AugmentedDickeyFuller(attr) => match attr {
                AdfAttr::TestStat => "augmented_dickey_fuller-teststat".to_string(),
                AdfAttr::PValue => "augmented_dickey_fuller-pvalue".to_string(),
                AdfAttr::UsedLag => "augmented_dickey_fuller-usedlag".to_string(),
            },
            Feature::ArCoefficient(k, p) => format!("ar_coefficient-{}-{}", k, p),
            Feature::FriedrichCoefficients(m, r_bits, coeff) => {
                format!(
                    "friedrich_coefficients-{}-{}-{}",
                    m,
                    f32::from_bits(*r_bits),
                    coeff
                )
            }
            _ => unreachable!("unit variant {self:?} missing from unit_features! table"),
        }
    }
}

/// A sampling frequency name parameter (`human_range_energy-<fs>`): positive
/// and finite, as f32 bits.
fn parse_fs(arg: &str) -> Option<u32> {
    let fs: f32 = arg.trim().parse().ok()?;
    (fs.is_finite() && fs > 0.0).then(|| fs.to_bits())
}

/// `name-<fs>` for a pinned sampling frequency, bare `name` for the extractor's.
fn with_fs(name: &str, fs: Option<u32>) -> String {
    match fs {
        Some(bits) => format!("{}-{}", name, f32::from_bits(bits)),
        None => name.to_string(),
    }
}
