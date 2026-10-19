use crate::common::ColumnState;
use crate::common::LANES;
use crate::types::{Compute, Feature};
use std::sync::Arc;

use super::processors::complexity_processor::ComplexityProcessor;
use super::processors::diff_processor::DiffProcessor;
use super::processors::fft_processor::FftProcessor;
use super::processors::mean_processor::MeanProcessor;
use super::processors::queue_processor::QueueProcessor;
use super::processors::sort_processor::SortProcessor;
use super::processors::stats_processor::StatsProcessor;
use super::processors::threshold_processor::ThresholdProcessor;
use super::processors::trend_processor::TrendProcessor;

pub(crate) struct SlidingEngine<'a> {
    pub(crate) compute: Compute,
    pub(crate) features: &'a [Feature],
    pub(crate) unique_paa_totals: &'a [u16],
    pub(crate) unique_c3_lags: &'a [u16],
    pub(crate) unique_tra_lags: &'a [u16],
    pub(crate) unique_count_above_thresholds: &'a [u32],
    pub(crate) unique_count_below_thresholds: &'a [u32],
    pub(crate) unique_range_counts: &'a [(u32, u32)],
    pub(crate) paa_boundaries: &'a [Vec<usize>],
    pub(crate) r2c: Option<Arc<dyn realfft::RealToComplex<f32>>>,
    pub(crate) fft_rebuild_every: usize,
}

impl<'a> SlidingEngine<'a> {
    #[inline(always)]
    pub(crate) fn update_batch(
        &self,
        old_slice: &[f32],
        new_slice: &[f32],
        value_before_old: Option<f32>,
        value_after_old: f32,
        value_before_new: f32,
        global_start_idx: usize,
        window_size: usize,
        state: &mut ColumnState,
    ) {
        let y = old_slice.len();
        if y == 0 {
            return;
        }

        StatsProcessor::update_batch(self.compute, old_slice, new_slice, state);
        ThresholdProcessor::update_batch(
            self.compute,
            old_slice,
            new_slice,
            self.unique_count_above_thresholds,
            self.unique_count_below_thresholds,
            self.unique_range_counts,
            state,
        );
        DiffProcessor::update_batch(
            self.compute,
            old_slice,
            new_slice,
            value_before_old,
            value_after_old,
            value_before_new,
            state,
        );

        TrendProcessor::update_batch(self.compute, old_slice, new_slice, window_size, state);

        if self
            .compute
            .intersects(crate::types::Compute::REOCCUR_RATIOS)
        {
            for &val in new_slice {
                let bits = crate::common::value_key(val);
                let count = state.value_counts.entry(bits).or_insert(0);
                *count += 1;
                if *count == 2 {
                    state.reoccurring_values += 1;
                    state.reoccurring_datapoints += 2;
                } else if *count > 2 {
                    state.reoccurring_datapoints += 1;
                }
            }

            for &val in old_slice {
                let old_bits = crate::common::value_key(val);
                let count = state.value_counts.entry(old_bits).or_insert(0);
                if *count == 2 {
                    state.reoccurring_values -= 1;
                    state.reoccurring_datapoints -= 2;
                } else if *count > 2 {
                    state.reoccurring_datapoints -= 1;
                }
                *count -= 1;
                if *count == 0 {
                    state.value_counts.remove(&old_bits);
                }
            }
        }

        QueueProcessor::update_batch(
            self.compute,
            new_slice,
            global_start_idx,
            window_size,
            state,
        );
    }

    #[inline(always)]
    pub(crate) fn update_incremental(
        &self,
        old_val: f32,
        new_val: f32,
        old_val_next: f32,
        old_last: f32,
        global_idx: usize,
        window_size: usize,
        state: &mut ColumnState,
    ) {
        StatsProcessor::update_incremental(self.compute, old_val, new_val, state);
        ThresholdProcessor::update_incremental(
            self.compute,
            old_val,
            new_val,
            self.unique_count_above_thresholds,
            self.unique_count_below_thresholds,
            self.unique_range_counts,
            state,
        );
        DiffProcessor::update_incremental(
            self.compute,
            old_val,
            new_val,
            old_val_next,
            old_last,
            state,
        );

        TrendProcessor::update_incremental(self.compute, old_val, new_val, window_size, state);

        if self
            .compute
            .intersects(crate::types::Compute::REOCCUR_RATIOS)
        {
            let bits = crate::common::value_key(new_val);
            let count = state.value_counts.entry(bits).or_insert(0);
            *count += 1;
            if *count == 2 {
                state.reoccurring_values += 1;
                state.reoccurring_datapoints += 2;
            } else if *count > 2 {
                state.reoccurring_datapoints += 1;
            }

            let old_bits = crate::common::value_key(old_val);
            let count = state.value_counts.entry(old_bits).or_insert(0);
            if *count == 2 {
                state.reoccurring_values -= 1;
                state.reoccurring_datapoints -= 2;
            } else if *count > 2 {
                state.reoccurring_datapoints -= 1;
            }
            *count -= 1;
            if *count == 0 {
                state.value_counts.remove(&old_bits);
            }
        }
        QueueProcessor::update_incremental(self.compute, new_val, global_idx, window_size, state);
    }

    /// Appends this window's features to `out`.
    #[inline(always)]
    pub(crate) fn process_column(
        &self,
        values: &[f32],
        state: &mut ColumnState,
        is_incremental: bool,
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        let n = values.len() as f32;
        if n == 0.0 {
            out.resize(out.len() + self.features.len(), 0.0);
            return Ok(());
        }

        StatsProcessor::reset_state(state, is_incremental);
        DiffProcessor::reset_state(state, values, is_incremental);
        ComplexityProcessor::reset_state(state);
        if !is_incremental {
            ThresholdProcessor::reset_state(state);
        }
        // The min/max queues are maintained incrementally by update_batch /
        // update_incremental; only rebuild them on a full recompute.
        if !is_incremental {
            QueueProcessor::reset_state(state, values.len());
        }
        // The SIMD/remainder passes skip stats on incremental windows, and a
        // maximum can't be updated as values leave, so rescan the window.
        if is_incremental && self.compute.contains(crate::types::Compute::ABS_MAX) {
            state.abs_max = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        }

        if !is_incremental
            && self
                .compute
                .intersects(crate::types::Compute::REOCCUR_RATIOS)
        {
            state.value_counts.clear();
            state.reoccurring_datapoints = 0;
            state.reoccurring_values = 0;
        }

        // SIMD Pass 1
        let rem_start = self.process_simd_chunks(values, state, is_incremental);

        // Remainder Pass 1
        self.process_remainder(values, rem_start, state, is_incremental);

        // Post-processing and Pass 2
        self.finalize_results(values, n, state, out)
    }
    #[inline(always)]
    fn process_simd_chunks(
        &self,
        values: &[f32],
        state: &mut ColumnState,
        is_incremental: bool,
    ) -> usize {
        let chunks = values.as_chunks::<LANES>();
        let mut i = 0;
        for chunk_slice in chunks.0.iter() {
            let chunk = std::simd::f32x4::from_slice(chunk_slice);
            let global_idx = i;

            if !is_incremental {
                StatsProcessor::process_simd(self.compute, chunk, global_idx, values.len(), state);
                ThresholdProcessor::process_simd(
                    self.compute,
                    chunk,
                    self.unique_count_above_thresholds,
                    self.unique_count_below_thresholds,
                    self.unique_range_counts,
                    state,
                );
                TrendProcessor::process_simd(self.compute, chunk, global_idx, state);

                if self
                    .compute
                    .intersects(crate::types::Compute::REOCCUR_RATIOS)
                {
                    let arr = chunk.to_array();
                    for &val in &arr {
                        let bits = crate::common::value_key(val);
                        let count = state.value_counts.entry(bits).or_insert(0);
                        *count += 1;
                        if *count == 2 {
                            state.reoccurring_values += 1;
                            state.reoccurring_datapoints += 2;
                        } else if *count > 2 {
                            state.reoccurring_datapoints += 1;
                        }
                    }
                }
            }

            DiffProcessor::process_simd(
                self.compute,
                chunk,
                values,
                global_idx,
                is_incremental,
                state,
            );

            ComplexityProcessor::process_simd(
                self.compute,
                chunk,
                global_idx,
                values,
                self.unique_paa_totals,
                self.paa_boundaries,
                self.unique_c3_lags,
                self.unique_tra_lags,
                state,
            );

            i += crate::common::LANES;
        }

        i
    }

    #[inline(always)]
    fn process_remainder(
        &self,
        values: &[f32],
        rem_start: usize,
        state: &mut ColumnState,
        is_incremental: bool,
    ) {
        let window_size = values.len();

        for i in rem_start..values.len() {
            let val = values[i];

            if !is_incremental {
                StatsProcessor::process_remainder(self.compute, val, state);
                ThresholdProcessor::process_remainder(
                    self.compute,
                    val,
                    self.unique_count_above_thresholds,
                    self.unique_count_below_thresholds,
                    self.unique_range_counts,
                    state,
                );
                TrendProcessor::process_remainder(self.compute, val, i, state);
            }

            if !is_incremental
                && self
                    .compute
                    .intersects(crate::types::Compute::REOCCUR_RATIOS)
            {
                let bits = crate::common::value_key(val);
                let count = state.value_counts.entry(bits).or_insert(0);
                *count += 1;
                if *count == 2 {
                    state.reoccurring_values += 1;
                    state.reoccurring_datapoints += 2;
                } else if *count > 2 {
                    state.reoccurring_datapoints += 1;
                }
            }

            if !is_incremental {
                QueueProcessor::process_remainder(self.compute, val, i, window_size, state);
            }

            if i > 0 {
                let prev = values[i - 1];
                DiffProcessor::process_remainder(
                    self.compute,
                    val,
                    prev,
                    i as f32,
                    is_incremental,
                    state,
                );
            } else {
                DiffProcessor::process_remainder(
                    self.compute,
                    val,
                    state.prev_last,
                    i as f32,
                    is_incremental,
                    state,
                );
            }
            state.prev_last = val;

            ComplexityProcessor::process_remainder(
                self.compute,
                val,
                i,
                values,
                self.unique_paa_totals,
                self.paa_boundaries,
                self.unique_c3_lags,
                self.unique_tra_lags,
                state,
            );
        }
    }

    #[inline(always)]
    pub(crate) fn finalize_results(
        &self,
        values: &[f32],
        n: f32,
        state: &mut ColumnState,
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        StatsProcessor::finalize_simd(self.compute, state);
        DiffProcessor::finalize_simd(self.compute, state);
        TrendProcessor::finalize_simd(self.compute, state);
        ComplexityProcessor::finalize_simd(self.compute, state);
        let base_metrics = StatsProcessor::finalize_base_metrics(state, n);
        let mut fft_res = FftProcessor::finalize(
            self.compute,
            values,
            n,
            base_metrics.mean,
            base_metrics.m2,
            &self.r2c,
            self.fft_rebuild_every,
            state,
        )?;

        let sort_metrics = SortProcessor::finalize(self.compute, values, n, state);
        let mean_metrics = MeanProcessor::finalize(self.compute, values, n, base_metrics.mean);
        let zc_metrics = DiffProcessor::finalize(self.compute, state);

        let mut context = crate::context::FeatureContext::new(
            values,
            None,
            state,
            n,
            base_metrics,
            sort_metrics,
            mean_metrics,
            zc_metrics,
            &fft_res,
            self.unique_c3_lags,
            self.unique_tra_lags,
            self.unique_paa_totals,
            self.paa_boundaries,
            self.unique_count_above_thresholds,
            self.unique_count_below_thresholds,
            self.unique_range_counts,
        );

        out.extend(
            self.features
                .iter()
                .map(|feat| crate::features::eval(feat, &mut context)),
        );

        fft_res.recycle(state);
        if let Some(sdft) = state.sliding_dft.as_mut() {
            sdft.bins = std::mem::take(&mut fft_res.fft_complex);
        }
        Ok(())
    }
}
