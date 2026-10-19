pub struct BaseMetrics {
    pub mean: f32,
    pub m2: f32,
    pub m3: f32,
    pub m4: f32,
    pub var: f32,
    pub std_dev: f32,
}

pub struct SortMetrics {
    pub first_max_idx: usize,
    pub last_max_idx: usize,
    pub first_min_idx: usize,
    pub last_min_idx: usize,
    pub median: f32,
    pub median_abs_dev: f32,
    pub iqr: f32,
}

pub struct MeanMetrics {
    pub mad_sum: f32,
    pub count_a: usize,
    pub count_b: usize,
    pub max_strike_a: usize,
    pub max_strike_b: usize,
}

pub struct ZcMetrics {
    pub zc_mean: f32,
    pub zc_std: f32,
}

#[derive(Default)]
pub struct FftResult {
    pub dft_len: usize,
    pub fft_complex: Vec<realfft::num_complex::Complex<f32>>,
    pub spectrum: Vec<f32>,
    pub freq_centroid: f32,
    pub spectral_decrease: f32,
    pub spectral_slope: f32,
    pub spectral_spread: f32,
    pub spectral_entropy: f32,
    pub spectral_roll_on: f32,
    pub spectral_roll_off: f32,
    pub spectral_skewness: f32,
    pub spectral_kurtosis: f32,
    pub max_frequency: f32,
    pub median_frequency: f32,
    pub fundamental_frequency: f32,
    pub power_bandwidth: f32,
    pub spectral_positive_turning: f32,
    pub spectral_variation: f32,
    pub fft_autocorr: Vec<f32>,
    pub mfcc: [f32; 12],
    pub lpcc: [f32; crate::features::lpc::N_LPCC],
    /// TSFEL wavelet features over pywt scales 1..10.
    pub cwt_energy: [f32; 9],
    pub cwt_entropy: f32,
    pub cwt_abs_mean: [f32; 9],
    pub cwt_std: [f32; 9],
    pub cwt_var: [f32; 9],
    pub cwt_peaks: u16,
    pub welch_density: Vec<f32>,
}

impl FftResult {
    /// Hand this window's buffers back to `state` for the next window. The
    /// DFT itself (`fft_complex`) goes back to wherever the engine keeps it.
    pub fn recycle(&mut self, state: &mut crate::common::ColumnState) {
        state.spectrum_buffer = std::mem::take(&mut self.spectrum);
        state.fft_autocorr_buffer = std::mem::take(&mut self.fft_autocorr);
        state.welch_density = std::mem::take(&mut self.welch_density);
    }
}
