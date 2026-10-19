pub mod autocorrelation;
pub mod changes;
pub mod complexity;
pub mod crossings_peaks;
pub mod cwt;
pub mod distribution;
pub mod dynamic;
pub mod energy;
pub mod lpc;
pub mod min_max;
pub mod misc;
pub mod moments;
pub mod runs;
pub mod stationarity;
pub mod subsequence;
pub mod transform;

use crate::context::FeatureContext;
use crate::types::Feature;

/// Median of `v` (numpy's convention: mean of the two middle values for an
/// even length), reordering `v` in place. NaN for an empty slice.
pub(crate) fn median_in_place(v: &mut [f32]) -> f32 {
    let n = v.len();
    if n == 0 {
        return f32::NAN;
    }
    let (lo, &mut hi, _) = v.select_nth_unstable_by(n / 2, f32::total_cmp);
    if n % 2 == 1 {
        hi
    } else {
        0.5 * (lo.iter().copied().fold(f32::NEG_INFINITY, f32::max) + hi)
    }
}

/// How a value on a bin edge is binned.
#[derive(Clone, Copy)]
pub(crate) enum EdgeSide {
    /// `np.histogram`: bins are [e_k, e_k+1), the last one closed; a value on
    /// an inner edge goes up.
    Histogram,
    /// `np.searchsorted(edges[1..], v, "left")`: bins are (e_k, e_k+1]; a value
    /// on an inner edge goes down (tsfresh's lempel_ziv_complexity).
    SearchsortedLeft,
}

/// Bin thresholds for numpy's binning of f32 values over [lo, hi]. numpy's
/// edges are `np.linspace(lo, hi, bins + 1)` in f64; each is stored as the f32
/// that makes an f32 comparison exact (`v >= e` iff `v >= ceil32(e)`,
/// `v > e` iff `v > floor32(e)`), so `numpy_bin` stays in f32. Returns the
/// scale for the initial guess.
pub(crate) fn numpy_bin_edges(
    lo: f32,
    hi: f32,
    bins: usize,
    side: EdgeSide,
    out: &mut Vec<f32>,
) -> f32 {
    let (lo64, hi64) = (lo as f64, hi as f64);
    let step = (hi64 - lo64) / bins as f64;
    out.clear();
    out.extend((0..=bins).map(|k| {
        let e = if k == bins {
            hi64
        } else {
            k as f64 * step + lo64
        };
        let f = e as f32;
        match side {
            EdgeSide::Histogram if (f as f64) < e => f.next_up(),
            EdgeSide::SearchsortedLeft if (f as f64) > e => f.next_down(),
            _ => f,
        }
    }));
    (bins as f64 / (hi64 - lo64)) as f32
}

/// numpy's bin for `v` given `numpy_bin_edges(lo, .., side, edges)`: a scaled
/// guess, at most one bin off, then one branchless correction each way (the
/// data decides them, so branches would mispredict).
#[inline(always)]
pub(crate) fn numpy_bin(v: f32, lo: f32, norm: f32, edges: &[f32], side: EdgeSide) -> usize {
    let last = edges.len() as i32 - 2;
    let mut k = (((v - lo) * norm) as i32).clamp(0, last) as usize;
    let (below, above) = match side {
        EdgeSide::Histogram => (v < edges[k], v >= edges[k + 1]),
        EdgeSide::SearchsortedLeft => (v <= edges[k], v > edges[k + 1]),
    };
    k -= ((k > 0) & below) as usize;
    k += ((k < last as usize) & above & !below) as usize;
    k
}

/// The decimal a user wrote for an f32 parameter (0.1 is stored as
/// 0.1000000015): `p` rounded to the 7 significant digits an f32 holds, so
/// comparisons against it match the reference libraries' f64 parameter.
pub(crate) fn decimal_param(p: f32) -> f64 {
    let p = p as f64;
    if p == 0.0 || !p.is_finite() {
        return p;
    }
    let scale = 10f64.powi(6 - p.abs().log10().floor() as i32);
    (p * scale).round() / scale
}

/// Evaluates one feature. Every engine calls this.
///
/// The match is exhaustive on purpose: adding a `Feature` variant without
/// routing it to an `eval_*` function here is a compile error.
pub fn eval(feat: &Feature, ctx: &mut FeatureContext) -> f32 {
    use Feature as F;
    let res = match feat {
        F::HurstExponent => complexity::eval_complexity(feat, ctx),
        F::MaximumFractalLength => complexity::eval_complexity(feat, ctx),
        F::BiasedFisherKurtosis
        | F::Iqr
        | F::Mad
        | F::Mean
        | F::Skew
        | F::BiasedSkew
        | F::Std
        | F::TotalSum
        | F::UnbiasedFisherKurtosis
        | F::Variance
        | F::VariationCoefficient
        | F::LargeStandardDeviation(..) => moments::eval_moments(feat, ctx),
        F::AbsMax
        | F::FirstLocMax
        | F::FirstLocMin
        | F::HasDuplicate
        | F::HasDuplicateMax
        | F::HasDuplicateMin
        | F::LastLocMax
        | F::LastLocMin
        | F::Max
        | F::MeanNAbsoluteMax(..)
        | F::Min
        | F::PkPkDistance => min_max::eval_min_max(feat, ctx),
        F::BenfordCorrelation
        | F::BinnedEntropy(..)
        | F::ChangeQuantiles(..)
        | F::Ecdf(..)
        | F::EcdfPercentile(..)
        | F::EcdfPercentileCount(..)
        | F::HistMode(..)
        | F::EcdfSlope(..)
        | F::Entropy
        | F::IndexMassQuantile(..)
        | F::Median
        | F::MedianAbsDeviation
        | F::Quantile(..)
        | F::RatioBeyondRSigma(..)
        | F::SumOfReoccurringDataPoints
        | F::SumOfReoccurringValues
        | F::ValueCount(..)
        | F::SymmetryLooking(..) => distribution::eval_distribution(feat, ctx),
        F::Energy
        | F::EnergyRatioByChunks(..)
        | F::HumanRangeEnergy(..)
        | F::AveragePower(..)
        | F::Rms
        | F::RootMeanSquare => energy::eval_energy(feat, ctx),
        F::NegativeTurning
        | F::NumberCrossingM(..)
        | F::NumberPeaks(..)
        | F::PeakCount
        | F::PositiveTurning
        | F::ZeroCross
        | F::ZeroCrossingMean
        | F::ZeroCrossingRate
        | F::ZeroCrossingStd
        | F::SlopeSignChange
        | F::TurningPoints => crossings_peaks::eval_crossings_peaks(feat, ctx),
        F::AggAutocorrelation(..)
        | F::Autocorr(..)
        | F::AutocorrFirst1e
        | F::AutocorrLag1
        | F::PartialAutocorr(..)
        | F::TimeReversalAsymmetry(..) => autocorrelation::eval_autocorrelation(feat, ctx),
        F::AbsSumChange
        | F::AggLinearTrend(..)
        | F::Auc
        | F::CidCe
        | F::CidCeNormalized
        | F::Intercept
        | F::LinearTrend(..)
        | F::LinearTrendTimewise(..)
        | F::MeanAbsChange
        | F::MeanChange
        | F::MedianAbsDiff
        | F::MedianDiff
        | F::SignalDistance
        | F::Slope
        | F::MeanSecondDerivativeCentral => changes::eval_changes(feat, ctx),
        F::CountAboveMean
        | F::CountBelowMean
        | F::CountAbove(..)
        | F::CountBelow(..)
        | F::RangeCount(..)
        | F::LongestStrikeAboveMean
        | F::LongestStrikeBelowMean => runs::eval_runs(feat, ctx),
        F::MatrixProfile(..) | F::QuerySimilarityCount(..) => {
            subsequence::eval_subsequence(feat, ctx)
        }
        F::C3(..)
        | F::CalcCentroid(..)
        | F::CwtCoefficients(..)
        | F::FftCoefficient(..)
        | F::Lpcc(..)
        | F::MaxPowerSpectrum
        | F::Mfcc(..)
        | F::NumberCwtPeaks(..)
        | F::Paa(..)
        | F::SpectralCentroid
        | F::SpectralDecrease
        | F::SpectralDistance
        | F::SpectralEntropy
        | F::SpectralKurtosis
        | F::SpectralRollOff
        | F::SpectralRollOn
        | F::SpectralSkewness
        | F::SpectralSlope
        | F::SpectralSpread
        | F::SpectrogramCoefficients(..)
        | F::SpectrogramMeanCoeff(..)
        | F::SpktWelchDensity(..)
        | F::WaveletEnergy(..)
        | F::WaveletEntropy
        | F::WaveletAbsMean(..)
        | F::WaveletStd(..)
        | F::WaveletVar(..)
        | F::MaxFrequency
        | F::MedianFrequency
        | F::FundamentalFrequency
        | F::PowerBandwidth
        | F::SpectralPositiveTurning
        | F::SpectralVariation
        | F::FourierEntropy(..)
        | F::FftAggregated(..)
        | F::WaveletFeatures(..) => transform::eval_transform(feat, ctx),
        F::ApproxEntropy(..)
        | F::HiguchiFd
        | F::LempelZiv
        | F::LempelZivComplexity(..)
        | F::PermutationEntropy(..)
        | F::SampleEntropy
        | F::TsfreshSampleEntropy
        | F::Dfa
        | F::HurstExponent
        | F::MaximumFractalLength
        | F::PetrosianFractalDimension
        | F::Mse(..) => complexity::eval_complexity(feat, ctx),
        F::ArCoefficient(..) | F::FriedrichCoefficients(..) | F::MaxLangevinFixedPoint(..) => {
            dynamic::eval_dynamic(feat, ctx)
        }
        F::AugmentedDickeyFuller(..) => stationarity::eval_stationarity(feat, ctx),
        F::Length
        | F::PercentageOfReoccurringDatapointsToAllDatapoints
        | F::PercentageOfReoccurringValuesToAllValues
        | F::RatioValueNumberToTimeSeriesLength
        | F::VarianceLargerThanStandardDeviation => misc::eval_misc(feat, ctx),
        F::Dfa => complexity::eval_complexity(feat, ctx),
    };
    // `None` means "not computable for this window" (e.g. zero variance, window
    // shorter than a lag); engines have always reported that as 0.0.
    res.unwrap_or(0.0)
}
