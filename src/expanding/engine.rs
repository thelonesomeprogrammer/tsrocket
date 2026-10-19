use crate::common::ColumnState;
use crate::common::LANES;
use crate::types::{Compute, Feature};
use realfft::RealToComplex;
use std::simd::f32x4;
use std::sync::Arc;

use super::processors::diff_processor::DiffProcessor;
use super::processors::fft_processor::FftProcessor;
use super::processors::mean_processor::MeanProcessor;
use super::processors::sort_processor::SortProcessor;
use super::processors::stats_processor::StatsProcessor;
use super::processors::threshold_processor::ThresholdProcessor;
use super::processors::trend_processor::TrendProcessor;

pub(crate) struct ExpandingEngine<'a> {
    pub(crate) compute: Compute,
    pub(crate) features: &'a [Feature],
    pub(crate) unique_paa_totals: &'a [u16],
    pub(crate) unique_c3_lags: &'a [u16],
    pub(crate) unique_autocorr_lags: &'a [u16],
    pub(crate) unique_tra_lags: &'a [u16],
    pub(crate) unique_count_above_thresholds: &'a [u32],
    pub(crate) unique_count_below_thresholds: &'a [u32],
    pub(crate) unique_range_counts: &'a [(u32, u32)],
    pub(crate) paa_boundaries: &'a [Vec<usize>],
    pub(crate) r2c: Option<Arc<dyn RealToComplex<f32>>>,
    pub(crate) fft_update_period: usize,
}

impl<'a> ExpandingEngine<'a> {
    #[inline(always)]
    pub(crate) fn process_expanding(
        &mut self,
        values: &[f32],
        global_idx: usize,
        state: &mut ColumnState,
        full_series: &mut Vec<f32>,
        running_sorted: &mut Vec<f32>,
    ) -> Vec<f32> {
        let rem_start = self.process_simd_chunks(values, global_idx, state, full_series);
        self.process_remainder(values, rem_start, global_idx, state, full_series);

        if self
            .compute
            .intersects(crate::types::Compute::REOCCUR_RATIOS)
        {
            for &val in values {
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

        full_series.extend_from_slice(values);

        if state.prefix_sums.is_empty() && !full_series.is_empty() {
            let mut s = 0.0;
            state.prefix_sums.reserve(full_series.len());
            for &v in full_series.iter() {
                s += v;
                state.prefix_sums.push(s);
            }
        } else {
            let mut s = state.prefix_sums.last().copied().unwrap_or(0.0);
            for &v in values {
                s += v;
                state.prefix_sums.push(s);
            }
        }

        let total_n = full_series.len() as f32;
        self.finalize_results(total_n, state, running_sorted, full_series)
    }

    #[inline(always)]
    fn process_simd_chunks(
        &self,
        values: &[f32],
        global_start_idx: usize,
        state: &mut ColumnState,
        full_series: &[f32],
    ) -> usize {
        let chunks = values.chunks_exact(LANES);
        let rem_start = (values.len() / LANES) * LANES;

        for (c_idx, i) in chunks.enumerate() {
            let offset_usize = global_start_idx + c_idx * LANES;
            let offset = offset_usize as f32;
            let chunk = f32x4::from_slice(i);
            let shifted = f32x4::from_array([state.prev_last, i[0], i[1], i[2]]);

            StatsProcessor::process_simd(self.compute, chunk, i, offset_usize, state);
            ThresholdProcessor::process_simd(
                self.compute,
                chunk,
                self.unique_count_above_thresholds,
                self.unique_count_below_thresholds,
                self.unique_range_counts,
                state,
            );
            DiffProcessor::process_simd(self.compute, chunk, shifted, offset, state);
            TrendProcessor::process_simd(self.compute, chunk, offset, state);

            state.prev_last = i[LANES - 1];
        }

        for (i, &val) in values[..rem_start].iter().enumerate() {
            let global_idx = global_start_idx + i;
            TrendProcessor::process_sequential(
                self.compute,
                val,
                global_idx,
                global_start_idx,
                state,
                full_series,
                values,
                self.unique_c3_lags,
                self.unique_autocorr_lags,
                self.unique_tra_lags,
            );
        }
        rem_start
    }

    #[inline(always)]
    fn process_remainder(
        &self,
        values: &[f32],
        rem_start: usize,
        global_start_idx: usize,
        state: &mut ColumnState,
        full_series: &[f32],
    ) {
        for i in rem_start..values.len() {
            let val = values[i];
            let global_idx = global_start_idx + i;

            StatsProcessor::process_remainder(self.compute, val, global_idx, state);
            ThresholdProcessor::process_remainder(
                self.compute,
                val,
                self.unique_count_above_thresholds,
                self.unique_count_below_thresholds,
                self.unique_range_counts,
                state,
            );

            if global_idx > 0 {
                let prev = if i > 0 {
                    values[i - 1]
                } else {
                    state.prev_last
                };
                DiffProcessor::process_remainder(self.compute, val, prev, global_idx, state);
            }

            TrendProcessor::process_sequential(
                self.compute,
                val,
                global_idx,
                global_start_idx,
                state,
                full_series,
                values,
                self.unique_c3_lags,
                self.unique_autocorr_lags,
                self.unique_tra_lags,
            );
            TrendProcessor::process_remainder(self.compute, val, global_idx, state);

            state.prev_last = val;
        }
    }

    #[inline(always)]
    fn finalize_results(
        &self,
        n: f32,
        state: &mut ColumnState,
        running_sorted: &mut Vec<f32>,
        full_series: &[f32],
    ) -> Vec<f32> {
        StatsProcessor::finalize_simd(self.compute, state);
        DiffProcessor::finalize_simd(self.compute, state);
        TrendProcessor::finalize_simd(self.compute, state);
        TrendProcessor::finalize_paa(
            self.compute,
            self.unique_paa_totals,
            self.paa_boundaries,
            full_series,
            state,
        );

        let base_metrics = StatsProcessor::finalize_base_metrics(state, n);

        let (median, iqr, median_abs_dev) =
            SortProcessor::process_running_sorted(self.compute, full_series, running_sorted, state);

        let mut fft_res = FftProcessor::finalize(
            self.compute,
            full_series,
            base_metrics.mean,
            base_metrics.m2,
            &self.r2c,
            self.fft_update_period,
            state,
        )
        .unwrap_or_default();

        let mean_metrics =
            MeanProcessor::finalize(self.compute, full_series, n, base_metrics.mean, state);

        let zc_metrics = DiffProcessor::finalize(self.compute, state);

        let sort_metrics = crate::metrics::SortMetrics {
            first_max_idx: state.first_max_idx,
            last_max_idx: state.last_max_idx,
            first_min_idx: state.first_min_idx,
            last_min_idx: state.last_min_idx,
            median,
            median_abs_dev,
            iqr,
        };

        let mut context = crate::context::FeatureContext::new(
            full_series,
            Some(running_sorted),
            state,
            n,
            base_metrics,
            sort_metrics,
            mean_metrics,
            zc_metrics,
            &fft_res,
            &self.unique_c3_lags,
            &self.unique_tra_lags,
            &self.unique_paa_totals,
            &self.paa_boundaries,
            self.unique_count_above_thresholds,
            self.unique_count_below_thresholds,
            self.unique_range_counts,
        );

        let mut feats = Vec::with_capacity(self.features.len());
        for feat in self.features {
            let val = crate::features::eval(feat, &mut context);
            feats.push(val);
        }

        fft_res.recycle(state);
        state.last_fft_complex = std::mem::take(&mut fft_res.fft_complex);

        feats
    }
}
