use super::compute::Compute;

// ─── Feature enum ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy, strum::EnumDiscriminants)]
#[strum_discriminants(derive(Hash, strum::EnumIter))]
pub enum Feature {
    TotalSum,
    Mean,
    Variance,
    Std,
    Min,
    Max,
    Median,
    MedianAbsDeviation,
    Skew,
    BiasedSkew,
    UnbiasedFisherKurtosis, // tsfresh default
    BiasedFisherKurtosis,   // tsfel default
    Mad,
    Iqr,
    Entropy,
    SampleEntropy,        // tsfel definition
    TsfreshSampleEntropy, // tsfresh definition
    Mse(u8, u16),
    HiguchiFd,
    Dfa,
    HurstExponent,
    MaximumFractalLength,
    BinnedEntropy(u32),
    Energy,
    EnergyRatioByChunks(u16, u16),
    Rms,
    RootMeanSquare,
    ZeroCrossingRate,
    PeakCount,
    NegativeTurning,
    PositiveTurning,
    NumberCrossingM(u32),
    NumberPeaks(u16),
    AutocorrLag1,    // Centered (tsfresh default)
    AutocorrFirst1e, // tsfel 'Autocorrelation' feature
    MeanAbsChange,
    MeanChange,
    MedianDiff,
    MedianAbsDiff,
    CidCe,
    CidCeNormalized,
    Slope,
    Intercept,
    Paa(u16, u16),
    AbsSumChange,
    CountAboveMean,
    CountBelowMean,
    CountAbove(u32),      // t encoded as f32 bits
    CountBelow(u32),      // t encoded as f32 bits
    RangeCount(u32, u32), // min, max encoded as f32 bits
    LongestStrikeAboveMean,
    LongestStrikeBelowMean,
    VariationCoefficient,
    C3(u16),
    Auc,
    SlopeSignChange,
    TurningPoints,
    ZeroCrossingMean,
    ZeroCrossingStd,
    AbsMax,
    FirstLocMax,
    LastLocMax,
    FirstLocMin,
    LastLocMin,
    Autocorr(u16),
    AggAutocorrelation(AggFunc, u16),
    PartialAutocorr(u16),
    TimeReversalAsymmetry(u16),
    LempelZivComplexity(u16),
    LempelZiv,
    FftCoefficient(u16, FftAttr),
    ApproxEntropy(u8, u32), // r is encoded as u32 (fixed point or bitcast)
    LinearTrend(AggAttr),
    LinearTrendTimewise(AggAttr, u32), // sampling period in seconds as f32 bits
    AggLinearTrend(AggAttr, u16, AggFunc),
    Quantile(u32),                            // q encoded as u32 bits
    ChangeQuantiles(u32, u32, bool, AggFunc), // ql, qh, isabs, f_agg
    IndexMassQuantile(u32),                   // q encoded as u32 bits
    BenfordCorrelation,
    MaxLangevinFixedPoint(u8, u32), // m, r as bits
    ArCoefficient(u16, u16),
    FriedrichCoefficients(u8, u32, u16),
    SumOfReoccurringValues,
    SumOfReoccurringDataPoints,
    PercentageOfReoccurringDatapointsToAllDatapoints,
    PercentageOfReoccurringValuesToAllValues,
    RatioValueNumberToTimeSeriesLength,
    MeanNAbsoluteMax(u16),
    Length,
    VarianceLargerThanStandardDeviation,
    QuerySimilarityCount(u16, u32),
    MatrixProfile(u16, crate::types::AggFunc),
    HumanRangeEnergy(Option<u32>), // fs as bits; None = the extractor's fs
    SpectralCentroid,
    SpectralDistance,
    SpectralDecrease,
    SpectralSlope,
    SpectralSpread,
    SpectralEntropy,
    SpectralRollOn,
    SpectralRollOff,
    SpectralSkewness,
    SpectralKurtosis,
    SignalDistance,
    WaveletFeatures(u32, u16), // mother wavelet (freq stored as f32 bits), feature type
    SpectrogramCoefficients(u16, u32), // time, freq stored as f32 bits
    SpectrogramMeanCoeff(u16, u16), // coefficient index, bins
    PkPkDistance,
    ZeroCross,
    MaxPowerSpectrum,
    MeanSecondDerivativeCentral,
    LargeStandardDeviation(u32),
    SymmetryLooking(u32),
    RatioBeyondRSigma(u32), // r encoded as u32 bits
    HasDuplicateMax,
    HasDuplicateMin,
    HasDuplicate,
    Ecdf(u32),
    EcdfPercentile(u32),
    EcdfPercentileCount(u32),
    HistMode(u16), // nbins
    EcdfSlope(u32, u32),
    PermutationEntropy(u32, u32),
    ValueCount(u32),
    CalcCentroid(Option<u32>), // fs as bits; None = the extractor's fs
    Mfcc(u16),
    Lpcc(u16),
    WaveletEnergy(u16),
    WaveletEntropy,
    WaveletAbsMean(u16),
    WaveletStd(u16),
    WaveletVar(u16),
    SpktWelchDensity(u16),
    CwtCoefficients([u16; 8], u8, u16, u16),
    NumberCwtPeaks(u16),
    AugmentedDickeyFuller(AdfAttr),
    MaxFrequency,
    MedianFrequency,
    FundamentalFrequency,
    PowerBandwidth,
    SpectralPositiveTurning,
    SpectralVariation,
    FourierEntropy(u32),
    FftAggregated(FftAggType),
    PetrosianFractalDimension,
    AveragePower(Option<u32>), // fs as bits; None = the extractor's fs
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub enum FftAttr {
    Real,
    Imag,
    Abs,
    Angle,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub enum AdfAttr {
    TestStat,
    PValue,
    UsedLag,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub enum AggAttr {
    Slope,
    Intercept,
    Stderr,
    RValue,
    PValue,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub enum AggFunc {
    Max,
    Min,
    Mean,
    Var,
    Median,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub enum FftAggType {
    Centroid,
    Variance,
    Skew,
    Kurtosis,
}

// ─── Feature → Compute dependency mapping ───────────────────────────────────

impl Feature {
    /// Returns the set of computation flags this feature requires.
    pub fn required_compute(&self) -> Compute {
        use Compute as C;
        match self {
            Self::TotalSum => C::SUM,
            Self::Mean => C::SUM | C::MEAN,
            Self::Variance => C::SUM | C::MEAN | C::VARIANCE | C::ENERGY | C::NEEDS_SORT,
            Self::Std => C::SUM | C::MEAN | C::VARIANCE | C::STD | C::ENERGY | C::NEEDS_SORT,
            Self::Min => C::MIN,
            Self::Max => C::MAX,
            Self::Median => C::MEDIAN | C::NEEDS_SORT,
            Self::MedianAbsDeviation => C::MEDIAN | C::MEDIAN_ABS_DEV | C::NEEDS_SORT,
            Self::Skew | Self::BiasedSkew => {
                C::SUM | C::MEAN | C::VARIANCE | C::SKEW | C::ENERGY | C::NEEDS_SORT
            }
            // The 4th central moment needs the sum of cubes, accumulated under SKEW.
            Self::UnbiasedFisherKurtosis | Self::BiasedFisherKurtosis => {
                C::SUM | C::MEAN | C::VARIANCE | C::SKEW | C::KURTOSIS | C::ENERGY | C::NEEDS_SORT
            }
            Self::Mad => C::SUM | C::MEAN | C::MAD | C::NEEDS_SORT,
            Self::Iqr => C::MIN | C::MAX | C::MEDIAN | C::IQR | C::NEEDS_SORT,
            Self::Entropy => C::empty(),
            Self::SampleEntropy => {
                C::SUM | C::MEAN | C::VARIANCE | C::STD | C::ENERGY | C::SAMP_ENT | C::NEEDS_SORT
            }
            Self::TsfreshSampleEntropy => C::empty(),
            Self::Mse(..) => C::SUM | C::MEAN | C::VARIANCE | C::STD | C::ENERGY | C::NEEDS_SORT,
            Self::BinnedEntropy(_) => C::MIN | C::MAX | C::BINNED_ENT,
            Self::Energy => C::ENERGY,
            Self::EnergyRatioByChunks(_, _) => C::ENERGY,
            Self::Rms => C::ENERGY | C::RMS,
            Self::RootMeanSquare => C::ENERGY | C::ROOT_MEAN_SQ,
            Self::ZeroCrossingRate => C::ZERO_CROSS,
            Self::PeakCount => C::PEAKS,
            Self::NegativeTurning => C::TROUGHS,
            Self::PositiveTurning => C::PEAKS,
            Self::NumberCrossingM(_) => C::NUMBER_PEAKS_CROSSINGS,
            Self::NumberPeaks(_) => C::NUMBER_PEAKS_CROSSINGS,
            Self::AutocorrLag1 => C::SUM | C::MEAN | C::ENERGY,
            Self::AutocorrFirst1e => {
                C::SUM | C::MEAN | C::ENERGY | C::FULL_AUTOCORR | C::NEEDS_SORT
            }
            Self::MeanAbsChange => C::SUM | C::MEAN | C::MAC,
            Self::MeanChange => C::SUM | C::MEAN | C::MC,
            Self::MedianDiff => C::empty(),
            Self::MedianAbsDiff => C::empty(),
            Self::CidCe => C::SUM | C::MEAN | C::CID_CE,
            Self::CidCeNormalized => {
                C::SUM | C::MEAN | C::VARIANCE | C::STD | C::ENERGY | C::CID_CE | C::MIN | C::MAX
            }
            Self::Slope => C::SUM | C::MEAN | C::SLOPE,
            Self::Intercept => C::SUM | C::MEAN | C::SLOPE | C::INTERCEPT,
            Self::Paa(_, _) => C::empty(),
            Self::AbsSumChange => C::MAC,
            Self::CountAboveMean => C::SUM | C::MEAN | C::CNT_ABOVE_MEAN | C::NEEDS_SORT,
            Self::CountBelowMean => C::SUM | C::MEAN | C::CNT_BELOW_MEAN | C::NEEDS_SORT,
            Self::CountAbove(_) => C::COUNT_ABOVE,
            Self::CountBelow(_) => C::COUNT_BELOW,
            Self::RangeCount(_, _) => C::RANGE_COUNT,
            Self::LongestStrikeAboveMean => C::SUM | C::MEAN | C::STRIKE_ABOVE | C::NEEDS_SORT,
            Self::LongestStrikeBelowMean => C::SUM | C::MEAN | C::STRIKE_BELOW | C::NEEDS_SORT,
            Self::VariationCoefficient => {
                C::SUM | C::MEAN | C::VARIANCE | C::STD | C::ENERGY | C::VAR_COEFF | C::NEEDS_SORT
            }
            Self::C3(_) => C::C3,
            Self::Auc => C::empty(),
            Self::SlopeSignChange => C::empty(),
            Self::TurningPoints => C::PEAKS | C::TROUGHS,
            Self::ZeroCrossingMean => {
                C::SUM | C::MEAN | C::ZERO_CROSS | C::ZC_STATS | C::ZC_INDICES | C::NEEDS_SORT
            }
            Self::ZeroCrossingStd => {
                C::SUM
                    | C::MEAN
                    | C::ZERO_CROSS
                    | C::ZC_STATS
                    | C::ZC_STD
                    | C::ZC_INDICES
                    | C::NEEDS_SORT
            }
            Self::AbsMax => C::ABS_MAX,
            Self::FirstLocMax => C::MAX | C::NEEDS_SORT | C::FIRST_LOC_MAX,
            Self::LastLocMax => C::MAX | C::NEEDS_SORT | C::LAST_LOC_MAX,
            Self::FirstLocMin => C::MIN | C::NEEDS_SORT | C::FIRST_LOC_MIN,
            Self::LastLocMin => C::MIN | C::NEEDS_SORT | C::LAST_LOC_MIN,
            Self::Autocorr(_) => {
                C::SUM | C::MEAN | C::VARIANCE | C::ENERGY | C::FULL_AUTOCORR | C::NEEDS_SORT
            }
            Self::AggAutocorrelation(_, _) => {
                C::SUM | C::MEAN | C::VARIANCE | C::ENERGY | C::FULL_AUTOCORR | C::NEEDS_SORT
            }
            Self::PartialAutocorr(_) => {
                C::SUM | C::MEAN | C::VARIANCE | C::ENERGY | C::PACF | C::NEEDS_SORT
            }
            Self::TimeReversalAsymmetry(_) => C::TRA | C::NEEDS_SORT,
            Self::LempelZiv | Self::LempelZivComplexity(_) => {
                C::SUM | C::MEAN | C::MIN | C::MAX | C::NEEDS_SORT
            }
            Self::FftCoefficient(_, _) => C::FFT_COEFF | C::NEEDS_SORT,
            Self::ApproxEntropy(_, _) => C::APPROX_ENT | C::NEEDS_SORT,
            Self::LinearTrend(_) | Self::LinearTrendTimewise(..) => {
                C::SUM | C::MEAN | C::SLOPE | C::VARIANCE | C::ENERGY
            }
            Self::AggLinearTrend(_, _, _) => C::AGG_LIN_TREND | C::NEEDS_SORT,
            Self::Quantile(_) => C::QUANTILE | C::NEEDS_SORT,
            Self::ChangeQuantiles(_, _, _, _) => C::QUANTILE | C::NEEDS_SORT,
            Self::IndexMassQuantile(_) => C::empty(),
            Self::BenfordCorrelation => C::BENFORD | C::NEEDS_SORT,
            Self::MaxLangevinFixedPoint(_, _) => C::LANGEVIN | C::NEEDS_SORT,
            Self::ArCoefficient(_, _) => C::AR_COEFF | C::NEEDS_SORT,
            Self::FriedrichCoefficients(_, _, _) => C::FRIEDRICH | C::NEEDS_SORT,
            Self::SumOfReoccurringValues => C::REOCCUR_VAL | C::NEEDS_SORT,
            Self::SumOfReoccurringDataPoints => C::REOCCUR_DP | C::NEEDS_SORT,
            Self::PercentageOfReoccurringDatapointsToAllDatapoints => {
                C::REOCCUR_RATIOS | C::NEEDS_SORT
            }
            Self::PercentageOfReoccurringValuesToAllValues => C::REOCCUR_RATIOS | C::NEEDS_SORT,
            Self::RatioValueNumberToTimeSeriesLength => C::REOCCUR_RATIOS | C::NEEDS_SORT,
            Self::MeanNAbsoluteMax(_) => C::MEAN_N_ABS_MAX | C::NEEDS_SORT,
            Self::Length => C::LENGTH | C::NEEDS_SORT,
            Self::VarianceLargerThanStandardDeviation => {
                C::SUM | C::MEAN | C::VARIANCE | C::ENERGY | C::NEEDS_SORT | C::VAR_GT_STD
            }
            Self::HumanRangeEnergy(_) => C::HUMAN_RANGE_E | C::NEEDS_SORT,
            Self::SpectralCentroid => C::SPEC_CENTROID | C::NEEDS_SORT,
            Self::SpectralDistance => C::SPEC_DISTANCE | C::NEEDS_SORT,
            Self::SpectralDecrease => C::SPEC_DECREASE | C::NEEDS_SORT,
            Self::SpectralSlope => C::SPEC_SLOPE | C::NEEDS_SORT,
            Self::SpectralSpread => C::SPEC_SPREAD | C::NEEDS_SORT,
            Self::SpectralEntropy => C::SPEC_ENTROPY | C::NEEDS_SORT,
            Self::SpectralRollOn => C::SPEC_ROLLON | C::NEEDS_SORT,
            Self::SpectralRollOff => C::SPEC_ROLLOFF | C::NEEDS_SORT,
            Self::SpectralSkewness => C::SPEC_SKEWNESS | C::NEEDS_SORT,
            Self::SpectralKurtosis => C::SPEC_KURTOSIS | C::NEEDS_SORT,
            Self::SignalDistance => C::SIG_DISTANCE | C::NEEDS_SORT,
            Self::WaveletFeatures(_, _) => C::WAVELET | C::NEEDS_SORT,
            Self::QuerySimilarityCount(_, _) => C::QUERY_SIMILARITY | C::NEEDS_SORT,
            Self::MatrixProfile(_, _) => C::MATRIX_PROFILE | C::NEEDS_SORT,
            Self::SpectrogramCoefficients(_, _) => C::SPECTROGRAM | C::NEEDS_SORT,
            Self::SpectrogramMeanCoeff(_, _) => C::empty(),
            Self::MeanSecondDerivativeCentral => C::empty(),
            Self::LargeStandardDeviation(_) => {
                C::SUM | C::MEAN | C::VARIANCE | C::MIN | C::MAX | C::ENERGY | C::NEEDS_SORT
            }
            Self::SymmetryLooking(_) => {
                C::SUM | C::MEAN | C::MIN | C::MAX | C::MEDIAN | C::NEEDS_SORT
            }
            Self::RatioBeyondRSigma(_) => {
                C::SUM | C::MEAN | C::VARIANCE | C::ENERGY | C::NEEDS_SORT | C::RATIO_BEYOND_R_SIGMA
            }
            Self::HasDuplicateMax => C::MAX | C::HAS_DUP_MAX | C::NEEDS_SORT,
            Self::HasDuplicateMin => C::MIN | C::HAS_DUP_MIN | C::NEEDS_SORT,
            Self::HasDuplicate => C::HAS_DUPLICATE | C::NEEDS_SORT,
            Self::PkPkDistance => C::MIN | C::MAX,
            Self::ZeroCross => C::ZERO_CROSS,
            Self::MaxPowerSpectrum => C::empty(),
            Self::Ecdf(_) => C::LENGTH,
            Self::EcdfPercentile(_) => C::MIN | C::MAX,
            Self::EcdfPercentileCount(_) => C::MIN | C::MAX,
            Self::HistMode(_) => C::MIN | C::MAX,
            Self::EcdfSlope(_, _) => C::MIN | C::MAX,
            Self::PermutationEntropy(_, _) => C::empty(),
            Self::ValueCount(_) => C::empty(),
            Self::CalcCentroid(_) => C::ENERGY | C::CALC_CENTROID,
            Self::Mfcc(_) => C::MFCC,
            Self::Lpcc(_) => C::LPCC,
            Self::WaveletEnergy(_) => C::CWT_MEXH,
            Self::WaveletEntropy => C::CWT_MEXH,
            Self::WaveletAbsMean(_) => C::CWT_MEXH,
            Self::WaveletStd(_) => C::CWT_MEXH,
            Self::WaveletVar(_) => C::CWT_MEXH,
            Self::SpktWelchDensity(_) => C::WELCH,
            Self::CwtCoefficients(_, _, _, _) => C::CWT,
            Self::NumberCwtPeaks(_) => C::CWT,
            Self::AugmentedDickeyFuller(_) => C::ADF | C::NEEDS_SORT,
            Self::MaxFrequency => C::MAX_FREQ,
            Self::MedianFrequency => C::MEDIAN_FREQ,
            Self::FundamentalFrequency => C::FUNDAMENTAL_FREQ,
            Self::PowerBandwidth => C::POWER_BANDWIDTH,
            Self::SpectralPositiveTurning => C::SPEC_POS_TURN,
            Self::SpectralVariation => C::SPEC_VARIATION,
            Self::FourierEntropy(_) => C::WELCH,
            Self::FftAggregated(_) => C::FFT_AGGREGATED,
            Self::Dfa | Self::HurstExponent => C::empty(),
            Self::HiguchiFd => C::HIGUCHI,
            Self::MaximumFractalLength => C::HIGUCHI,
            Self::PetrosianFractalDimension => C::empty(),
            Self::AveragePower(_) => C::ENERGY,
        }
    }
}

// ─── Feature → Compute aggregation ─────────────────────────────────────────

/// Fold a feature list into a single `Compute` bitflag set.
pub fn compute_flags(features: &[Feature]) -> Compute {
    features
        .iter()
        .fold(Compute::empty(), |acc, f| acc | f.required_compute())
}
