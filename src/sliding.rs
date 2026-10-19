use crate::common::{ColumnState, map_features_to_indices};
use crate::numpy_io;
use crate::types::{Compute, Feature};
use numpy::PyArray3;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use realfft::RealFftPlanner;
use std::sync::{Arc, Mutex};

pub mod engine;
pub mod processors;
use engine::SlidingEngine;

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct SlidingExtractor {
    pub features: Vec<Feature>,
    pub compute: Compute,
    pub unique_paa_totals: Vec<u16>,
    pub unique_c3_lags: Vec<u16>,
    pub unique_autocorr_lags: Vec<u16>,
    pub unique_tra_lags: Vec<u16>,
    pub unique_count_above_thresholds: Vec<u32>,
    pub unique_count_below_thresholds: Vec<u32>,
    pub unique_range_counts: Vec<(u32, u32)>,
    pub window_size: usize,
    pub stride: usize,
    pub fs: f32,
    /// Rebuild the sliding DFT from a fresh FFT every this many samples:
    /// `window_size` by default, lower with a `fresh-N` meta feature.
    pub fft_rebuild_every: usize,
    // State per column
    pub states: Vec<ColumnState>,
    pub histories: Vec<Vec<f32>>,
    pub planner: Arc<Mutex<RealFftPlanner<f32>>>,
}

#[pymethods]
impl SlidingExtractor {
    #[new]
    #[pyo3(signature = (feature_str, n_cols, window_size, stride=1, fs=100.0))]
    pub fn new(
        feature_str: Vec<String>,
        n_cols: usize,
        window_size: usize,
        stride: usize,
        fs: f32,
    ) -> PyResult<Self> {
        if !(fs.is_finite() && fs > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "fs must be a positive, finite sampling frequency, got {fs}"
            )));
        }
        let mut features = Vec::new();
        let mut unique_paa_totals = std::collections::BTreeSet::new();
        let mut unique_c3_lags = std::collections::BTreeSet::new();
        let mut unique_autocorr_lags = std::collections::BTreeSet::new();
        let mut unique_tra_lags = std::collections::BTreeSet::new();
        let mut unique_count_above_thresholds = std::collections::BTreeSet::new();
        let mut unique_count_below_thresholds = std::collections::BTreeSet::new();
        let mut unique_range_counts = std::collections::BTreeSet::new();

        let (feature_str, meta) = crate::types::split_meta(feature_str)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        for i in feature_str {
            let feat = Feature::parse_with_fs(&i, fs)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            match feat {
                Feature::Paa(total, _) => {
                    unique_paa_totals.insert(total);
                }
                Feature::C3(lag) => {
                    unique_c3_lags.insert(lag);
                }
                Feature::Autocorr(lag) => {
                    unique_autocorr_lags.insert(lag);
                }
                Feature::PartialAutocorr(lag) => {
                    for l in 1..=lag {
                        unique_autocorr_lags.insert(l);
                    }
                }
                Feature::TimeReversalAsymmetry(lag) => {
                    unique_tra_lags.insert(lag);
                }
                Feature::CountAbove(t) => {
                    unique_count_above_thresholds.insert(t);
                }
                Feature::CountBelow(t) => {
                    unique_count_below_thresholds.insert(t);
                }
                Feature::RangeCount(min, max) => {
                    unique_range_counts.insert((min, max));
                }
                _ => {}
            }
            features.push(feat);
        }
        let compute = map_features_to_indices(&features);
        let unique_paa_totals: Vec<u16> = unique_paa_totals.into_iter().collect();
        let unique_c3_lags: Vec<u16> = unique_c3_lags.into_iter().collect();
        let unique_autocorr_lags: Vec<u16> = unique_autocorr_lags.into_iter().collect();
        let unique_tra_lags: Vec<u16> = unique_tra_lags.into_iter().collect();
        let unique_count_above_thresholds: Vec<u32> =
            unique_count_above_thresholds.into_iter().collect();
        let unique_count_below_thresholds: Vec<u32> =
            unique_count_below_thresholds.into_iter().collect();
        let unique_range_counts: Vec<(u32, u32)> = unique_range_counts.into_iter().collect();

        let planner = RealFftPlanner::<f32>::new();
        let planner_arc = Arc::new(Mutex::new(planner));

        if compute.intersects(Compute::ANY_FFT) {
            let mut p = planner_arc.lock().unwrap_or_else(|e| e.into_inner());
            p.plan_fft_forward(window_size);
        }

        Ok(Self {
            features,
            compute,
            unique_paa_totals: unique_paa_totals.clone(),
            unique_c3_lags: unique_c3_lags.clone(),
            unique_autocorr_lags: unique_autocorr_lags.clone(),
            unique_tra_lags: unique_tra_lags.clone(),
            unique_count_above_thresholds: unique_count_above_thresholds.clone(),
            unique_count_below_thresholds: unique_count_below_thresholds.clone(),
            unique_range_counts: unique_range_counts.clone(),
            window_size,
            stride,
            fs,
            fft_rebuild_every: meta
                .fresh_fft_every
                .map_or(window_size, |n| n.min(window_size))
                .max(1),
            states: (0..n_cols)
                .map(|_| {
                    ColumnState::new(
                        &unique_paa_totals,
                        &unique_c3_lags,
                        &unique_autocorr_lags,
                        &unique_tra_lags,
                        &unique_count_above_thresholds,
                        &unique_count_below_thresholds,
                        &unique_range_counts,
                        0.0,
                        fs,
                    )
                })
                .collect(),
            histories: vec![Vec::with_capacity(window_size + stride); n_cols],
            planner: planner_arc,
        })
    }

    /// Canonical names of the last output axis, in order.
    #[getter]
    pub fn feature_names(&self) -> Vec<String> {
        self.features.iter().map(Feature::name).collect()
    }

    /// Append `values`, a 2-D float32/float64 numpy array of shape
    /// (n_series, n_new_samples), to each series' window. Returns a float32
    /// array of shape (n_series, n_windows, n_features): the features of every
    /// window completed by these samples, oldest first.
    pub fn update<'py>(
        &mut self,
        py: Python<'py>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let (n_series, n_windows, data) = numpy_io::with_rows(values, |rows, _| {
            let (n_windows, data) = py
                .detach(|| self.update_rows(rows))
                .map_err(PyTypeError::new_err)?;
            Ok((rows.len(), n_windows, data))
        })?;
        numpy_io::to_numpy_3d(py, (n_series, n_windows, self.features.len()), data)
    }
}

impl SlidingExtractor {
    /// Append one row of new samples per series (all the same length). Returns
    /// the number of windows completed per series and their features,
    /// flattened series-major, then window, then feature.
    pub fn update_rows(&mut self, rows: &[&[f32]]) -> Result<(usize, Vec<f32>), String> {
        let n_cols = rows.len();
        if n_cols == 0 || rows[0].is_empty() {
            return Ok((0, Vec::new()));
        }

        if self.states.len() < n_cols {
            for _ in self.states.len()..n_cols {
                self.states.push(ColumnState::new(
                    &self.unique_paa_totals,
                    &self.unique_c3_lags,
                    &self.unique_autocorr_lags,
                    &self.unique_tra_lags,
                    &self.unique_count_above_thresholds,
                    &self.unique_count_below_thresholds,
                    &self.unique_range_counts,
                    0.0,
                    self.fs,
                ));
                self.histories
                    .push(Vec::with_capacity(self.window_size + self.stride));
            }
        }

        let paa_boundaries: Vec<Vec<usize>> = self
            .unique_paa_totals
            .iter()
            .map(|&total| {
                let mut b = Vec::with_capacity(total as usize + 1);
                for i in 0..=total {
                    b.push((i as f32 * self.window_size as f32 / total as f32) as usize);
                }
                b
            })
            .collect();

        let r2c = if self.compute.intersects(Compute::ANY_FFT) {
            let mut p = self.planner.lock().unwrap_or_else(|e| e.into_inner());
            Some(p.plan_fft_forward(self.window_size))
        } else {
            None
        };

        use rayon::prelude::*;

        let window_size = self.window_size;
        let stride = self.stride;

        let column_results = self.states[..n_cols]
            .par_iter_mut()
            .zip(self.histories[..n_cols].par_iter_mut())
            .zip(rows.par_iter())
            .map(
                |((state, history), &values)| -> Result<(usize, Vec<f32>), String> {
                    let engine = SlidingEngine {
                        compute: self.compute,
                        features: &self.features,
                        unique_paa_totals: &self.unique_paa_totals,
                        unique_c3_lags: &self.unique_c3_lags,
                        unique_tra_lags: &self.unique_tra_lags,
                        unique_count_above_thresholds: &self.unique_count_above_thresholds,
                        unique_count_below_thresholds: &self.unique_count_below_thresholds,
                        unique_range_counts: &self.unique_range_counts,
                        paa_boundaries: &paa_boundaries,
                        r2c: r2c.as_ref().cloned(),
                        fft_rebuild_every: self.fft_rebuild_every,
                    };

                    // Windows completed by these values: at most one per stride.
                    let max_windows = (history.len() + values.len()) / stride.max(1) + 1;
                    let mut batch_res = Vec::with_capacity(max_windows * engine.features.len());
                    let mut n_windows = 0;

                    for &val in values {
                        history.push(val);

                        if history.len() == window_size {
                            // First time window is full
                            engine.process_column(history, state, false, &mut batch_res)?;
                            n_windows += 1;
                        } else if history.len() == window_size + stride {
                            // We have reached a stride boundary
                            let old_slice = &history[..stride];
                            let new_slice = &history[window_size..window_size + stride];

                            let value_after_old = history[stride];
                            let value_before_new = history[window_size - 1];

                            engine.update_batch(
                                old_slice,
                                new_slice,
                                None,
                                value_after_old,
                                value_before_new,
                                state.n as usize + window_size,
                                window_size,
                                state,
                            );

                            if let Some(ref mut sdft) = state.sliding_dft {
                                if sdft.updates + stride >= self.fft_rebuild_every {
                                    // Rebuilt from a fresh FFT for this window
                                    // anyway: skip the O(window) updates.
                                    sdft.updates += stride;
                                } else {
                                    for (i, &v) in old_slice.iter().enumerate() {
                                        sdft.update(v, new_slice[i]);
                                    }
                                }
                            }

                            history.drain(..stride);

                            engine.process_column(history, state, true, &mut batch_res)?;
                            n_windows += 1;
                            state.n += stride as f32;
                        }
                    }
                    Ok((n_windows, batch_res))
                },
            )
            .collect::<Result<Vec<(usize, Vec<f32>)>, String>>()?;

        let n_windows = column_results[0].0;
        let mut out = Vec::with_capacity(column_results.iter().map(|(_, r)| r.len()).sum());
        for (_, r) in &column_results {
            out.extend_from_slice(r);
        }
        Ok((n_windows, out))
    }
}
