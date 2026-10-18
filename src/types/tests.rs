use std::collections::HashSet;

use strum::IntoEnumIterator;

use super::feature::FeatureDiscriminants;
use super::parse::UNIT_FEATURES;
use super::{AdfAttr, AggAttr, AggFunc, Feature, FftAggType, FftAttr};

fn f(x: f32) -> u32 {
    x.to_bits()
}

/// One valid instance of every parameterized variant. Unit variants come from
/// the `unit_features!` table. Adding a `Feature` variant without a sample here
/// fails `every_variant_has_a_sample`.
fn parameterized_samples() -> Vec<Feature> {
    use Feature as F;
    vec![
        F::LempelZivComplexity(3),
        F::Mse(3, 0),
        F::Mse(2, 10),
        F::BinnedEntropy(5),
        F::EnergyRatioByChunks(3, 1),
        F::NumberCrossingM(f(0.5)),
        F::NumberPeaks(3),
        F::Paa(4, 1),
        F::C3(2),
        F::Autocorr(3),
        F::AggAutocorrelation(AggFunc::Mean, 10),
        F::PartialAutocorr(2),
        F::TimeReversalAsymmetry(2),
        F::FftCoefficient(3, FftAttr::Abs),
        F::ApproxEntropy(2, f(0.1)),
        F::LinearTrend(AggAttr::Slope),
        F::LinearTrendTimewise(AggAttr::Stderr, f(0.5)),
        F::AggLinearTrend(AggAttr::Intercept, 5, AggFunc::Max),
        F::Quantile(f(0.25)),
        F::ChangeQuantiles(f(0.2), f(0.8), true, AggFunc::Mean),
        F::IndexMassQuantile(f(0.5)),
        F::EcdfPercentile(f(0.5)),
        F::EcdfPercentileCount(f(0.5)),
        F::HistMode(10),
        F::EcdfSlope(f(0.2), f(0.5)),
        F::MaxLangevinFixedPoint(3, f(30.0)),
        F::ArCoefficient(10, 1),
        F::FriedrichCoefficients(3, f(30.0), 0),
        F::MeanNAbsoluteMax(7),
        F::QuerySimilarityCount(10, f(0.5)),
        F::MatrixProfile(10, AggFunc::Min),
        F::HumanRangeEnergy(Some(f(100.0))),
        F::HumanRangeEnergy(None),
        F::AveragePower(Some(f(100.0))),
        F::AveragePower(None),
        F::WaveletFeatures(f(0.5), 1),
        F::SpectrogramCoefficients(2, f(0.5)),
        F::SpectrogramMeanCoeff(3, 32),
        F::SpectrogramMeanCoeff(0, 5),
        F::LargeStandardDeviation(f(0.05)),
        F::SymmetryLooking(f(0.05)),
        F::RatioBeyondRSigma(f(1.5)),
        F::Ecdf(10),
        F::PermutationEntropy(1, 3),
        F::ValueCount(f(1.0)),
        F::CalcCentroid(Some(f(100.0))),
        F::CalcCentroid(None),
        F::Mfcc(3),
        F::Lpcc(3),
        F::WaveletEnergy(3),
        F::WaveletAbsMean(3),
        F::WaveletStd(3),
        F::WaveletVar(3),
        F::SpktWelchDensity(2),
        F::CwtCoefficients([2, 5, 10, 20, 0, 0, 0, 0], 4, 3, 5),
        F::NumberCwtPeaks(5),
        F::AugmentedDickeyFuller(AdfAttr::PValue),
        F::CountAbove(f(0.5)),
        F::CountBelow(f(0.5)),
        F::RangeCount(f(-1.0), f(1.0)),
        F::FourierEntropy(5),
        F::FftAggregated(FftAggType::Centroid),
        F::FftAggregated(FftAggType::Variance),
        F::FftAggregated(FftAggType::Skew),
        F::FftAggregated(FftAggType::Kurtosis),
    ]
}

fn samples() -> Vec<Feature> {
    UNIT_FEATURES
        .iter()
        .copied()
        .chain(parameterized_samples())
        .collect()
}

#[test]
fn every_variant_has_a_sample() {
    let covered: HashSet<FeatureDiscriminants> = samples()
        .into_iter()
        .map(FeatureDiscriminants::from)
        .collect();
    let missing: Vec<_> = FeatureDiscriminants::iter()
        .filter(|d| !covered.contains(d))
        .collect();
    assert!(
        missing.is_empty(),
        "add these to unit_features! (parse.rs) or parameterized_samples() (types/tests.rs): {missing:?}"
    );
}

#[test]
fn name_round_trips_through_parse() {
    let failures: Vec<String> = samples()
        .into_iter()
        .filter_map(|feat| {
            let name = feat.name();
            match name.parse::<Feature>() {
                Ok(parsed) if parsed == feat => None,
                other => Some(format!("{feat:?} -> {name:?} -> {other:?}")),
            }
        })
        .collect();
    assert!(
        failures.is_empty(),
        "name() doesn't parse back:\n{}",
        failures.join("\n")
    );
}

#[test]
fn invalid_parameters_are_rejected() {
    for s in [
        "paa-2-2",
        "mse-0",
        "hist_mode-0",
        "spectrogram_mean_coeff-32",
        "spectrogram_mean_coeff-0-1",
        "spectrogram_mean_coeff-1-2-3",
        "neighbourhood_peaks-0",
        "mse--1",
        "ar_coefficient-1-10",
        "ar_coefficient-0-0",
        "agg_linear_trend-slope-0-mean",
        "linear_trend_timewise-slope-0",
        "linear_trend_timewise-slope--1",
        "linear_trend_timewise-slope-inf",
        "linear_trend_timewise-slope-nan",
        "linear_trend_timewise-slope",
        "linear_trend_timewise-foo-1",
        "energy_ratio_by_chunks_num_segments_3__segment_focus_3",
        "human_range_energy-0",
        "average_power-0",
        "average_power--1",
        "calc_centroid-0",
        "calc_centroid--1",
        "calc_centroid-nan",
        "calc_centroid-inf",
        "ecdf_percentile-0.0",
        "ecdf_percentile-1.1",
        "ecdf_percentile_count-0.0",
        "ecdf_percentile_count-1.1",
        "ecdf_slope-0.5-0.5",
        "ecdf_slope-0.6-0.5",
        "ecdf_slope-0.0-0.5",
        "ecdf_slope-0.5-1.1",
        "count_above-nan",
        "count_below-nan",
        "range_count-nan-5",
        "range_count-5-nan",
        "range_count-5-0",
        "fourier_entropy-0",
        "fft_aggregated-mode",
        "not_a_feature",
        "lempel_ziv_complexity-1",
        "lempel_ziv_complexity-257",
    ] {
        assert!(s.parse::<Feature>().is_err(), "{s:?} should be rejected");
    }
}

#[test]
fn compute_flags_have_distinct_single_bits() {
    use super::Compute;

    let names: Vec<_> = Compute::all().iter_names().collect();
    for (name, flag) in &names {
        assert_eq!(flag.bits().count_ones(), 1, "{name} should be a single bit");
    }
    assert_eq!(
        Compute::all().bits().count_ones() as usize,
        names.len(),
        "two Compute flags share a bit"
    );
}

#[test]
fn python_feature_samples_cover_every_variant() {
    // tests/feature_samples.txt drives the cross-engine Python tests.
    let listed: HashSet<FeatureDiscriminants> = include_str!("../../tests/feature_samples.txt")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            l.parse::<Feature>()
                .unwrap_or_else(|e| panic!("tests/feature_samples.txt: {e}"))
                .into()
        })
        .collect();
    let missing: Vec<_> = FeatureDiscriminants::iter()
        .filter(|d| !listed.contains(d))
        .collect();
    assert!(
        missing.is_empty(),
        "add these to tests/feature_samples.txt: {missing:?}"
    );
}

#[test]
fn meta_features_are_split_off() {
    use super::{MetaFeatures, split_meta};
    let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();

    let (rest, meta) =
        split_meta(names(&["mean", "fresh-8", "spectral_entropy", "fresh-3"])).unwrap();
    assert_eq!(rest, names(&["mean", "spectral_entropy"]));
    assert_eq!(meta.fresh_fft_every, Some(3), "the smallest N wins");

    let (rest, meta) = split_meta(names(&["mean"])).unwrap();
    assert_eq!(rest, names(&["mean"]));
    assert_eq!(meta, MetaFeatures::default());

    for bad in ["fresh-0", "fresh-", "fresh-x", "fresh--1", "fresh-1.5"] {
        assert!(
            split_meta(names(&[bad])).is_err(),
            "{bad} should be rejected"
        );
    }
    // A meta feature is not a feature: it never parses as one.
    assert!("fresh-1".parse::<Feature>().is_err());
}

#[test]
fn tsfresh_and_tsfel_column_names_translate() {
    // A model trained on tsfresh/TSFEL output is deployed with its column names.
    for (external, canonical) in [
        ("value__mean", "mean"),
        ("value__abs_energy", "energy"),
        ("value__cid_ce__normalize_True", "cid_ce_normalized"),
        (
            r#"value__fft_coefficient__attr_"abs"__coeff_3"#,
            "fft_coeff-3-abs",
        ),
        (r#"fft_coefficient__attr_"abs"__coeff_3"#, "fft_coeff-3-abs"),
        (
            r#"a__b__fft_coefficient__attr_"abs"__coeff_3"#,
            "fft_coeff-3-abs",
        ),
        (
            r#"value__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"median""#,
            "agg_linear_trend-slope-5-median",
        ),
        (
            r#"value__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.2"#,
            "change_quantiles-0.2-0.8-True-var",
        ),
        ("value__range_count__max_1__min_-1", "range_count--1-1"),
        (
            "value__binned_entropy__max_bins_10",
            "binned_entropy__max_bins_10",
        ),
        (
            r#"value__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC""#,
            "augmented_dickey_fuller-pvalue",
        ),
        ("0_Spectral centroid", "spectral_centroid"),
        ("Kurtosis", "biased_fisher_kurtosis"),
        ("acc_x_ECDF Percentile_1", "ecdf_percentile-0.8"),
        ("0_ECDF_0", "ecdf-1"),
        ("0_MFCC_11", "mfcc-11"),
        (
            "value__query_similarity_count__query_None__threshold_0.0",
            "query_similarity_count-0-0",
        ),
    ] {
        let f: Feature = external
            .parse()
            .unwrap_or_else(|e| panic!("{external:?} should parse: {e}"));
        assert_eq!(f.name(), canonical, "{external:?}");
    }
}

#[test]
fn tsfel_frequency_names_need_fs() {
    let name = "0_Wavelet energy_12.5Hz";
    assert!(
        name.parse::<Feature>().is_err(),
        "no fs, no frequency mapping"
    );
    assert_eq!(
        Feature::parse_with_fs(name, 100.0).unwrap().name(),
        "wavelet_energy-1"
    );
    assert_eq!(
        Feature::parse_with_fs(name, 50.0).unwrap().name(),
        "wavelet_energy-0"
    );
    let spec = "0_Spectrogram mean coefficient_1.61Hz";
    assert_eq!(
        Feature::parse_with_fs(spec, 100.0).unwrap().name(),
        "spectrogram_mean_coeff-1"
    );
    assert!(Feature::parse_with_fs("0_Wavelet energy_13Hz", 100.0).is_err());
}

#[test]
fn bad_tsfresh_names_are_rejected() {
    for s in [
        // Unknown parameter, bad values, unsupported options.
        r#"value__fft_coefficient__attr_"abs"__coeff_3__bogus_1"#,
        r#"value__fft_coefficient__attr_"phase"__coeff_3"#,
        r#"value__agg_linear_trend__attr_"slope"__chunk_len_0__f_agg_"mean""#,
        r#"value__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mode""#,
        r#"value__change_quantiles__f_agg_"mean"__isabs_Maybe__qh_0.2__ql_0.0"#,
        r#"value__augmented_dickey_fuller__attr_"teststat"__autolag_"BIC""#,
        "value__cid_ce__normalize_maybe",
        "value__number_peaks__n_0",
        "value__linear_trend_timewise__attr_\"slope\"",
        "0_ECDF Percentile_2",
    ] {
        assert!(s.parse::<Feature>().is_err(), "{s:?} should be rejected");
    }
}

#[test]
fn unknown_names_suggest_the_closest_feature() {
    let err = "spectral_centriod".parse::<Feature>().unwrap_err();
    assert!(err.contains("did you mean `spectral_centroid`"), "{err}");
    let err = "fft_coef-3-abs".parse::<Feature>().unwrap_err();
    assert!(err.contains("did you mean `fft_coeff-...`"), "{err}");
    let err = "zzzzzzzzzzzzzzzz".parse::<Feature>().unwrap_err();
    assert!(!err.contains("did you mean"), "{err}");
}
