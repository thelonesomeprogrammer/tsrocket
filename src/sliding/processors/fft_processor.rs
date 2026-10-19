use std::sync::Arc;

use realfft::RealToComplex;

use crate::common::{ColumnState, SlidingDFT};
use crate::metrics::FftResult;
use crate::types::Compute;

pub struct FftProcessor;

impl FftProcessor {
    /// Sliding windows: the DFT is updated in O(window) per step by the sliding
    /// DFT (see `SlidingDFT::update` in sliding.rs) instead of an O(n log n)
    /// FFT. It is rebuilt from a fresh FFT on the first window and after every
    /// `rebuild_every` updates (`window` unless a `fresh-N` meta feature lowers
    /// it), which bounds the f32 drift of repeated twiddle products.
    pub fn finalize(
        compute: Compute,
        values: &[f32],
        n: f32,
        mean: f32,
        m2: f32,
        r2c: &Option<Arc<dyn RealToComplex<f32>>>,
        rebuild_every: usize,
        state: &mut ColumnState,
    ) -> Result<FftResult, String> {
        let mut fft_complex = Vec::new();
        if let Some(r2c) = r2c
            .as_ref()
            .filter(|_| compute.intersects(Compute::ANY_FFT))
        {
            let stale = state
                .sliding_dft
                .as_ref()
                .is_none_or(|s| s.updates >= rebuild_every);
            if stale {
                // Keep the twiddles; the old bins become the FFT's output buffer.
                if let Some(sdft) = state.sliding_dft.as_mut() {
                    state.fft_complex_buffer = std::mem::take(&mut sdft.bins);
                }
                let bins = crate::spectral::rfft(values, r2c.as_ref(), state)?;
                match state.sliding_dft.as_mut() {
                    Some(sdft) if sdft.n == values.len() => {
                        sdft.bins = bins;
                        sdft.updates = 0;
                    }
                    _ => state.sliding_dft = Some(SlidingDFT::from_fft(bins, values.len())),
                }
            }
            // Lent to the FftResult for this window; the engine puts it back
            // (`SlidingEngine::finalize_results`) before the next update.
            if let Some(sdft) = state.sliding_dft.as_mut() {
                fft_complex = std::mem::take(&mut sdft.bins);
            }
        }
        let res = crate::spectral::finalize(
            compute,
            values,
            n,
            mean,
            m2,
            fft_complex,
            values.len(),
            state,
        );
        if res.is_err() {
            // The bins went down with the result: rebuild from a fresh FFT.
            state.sliding_dft = None;
        }
        res
    }
}
