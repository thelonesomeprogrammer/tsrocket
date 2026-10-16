## 2026-10-02 - Reusing allocations in ColumnState
**Learning:** Repetitive inner-loop allocations (like `Vec::new()` or `Vec::with_capacity()` inside feature calculation blocks like `AggLinearTrend` and `PartialAutocorr`) can be a silent performance drain in this Rust codebase.
**Action:** Instead of allocating new vectors, add them to the persistent `ColumnState` struct, extract them safely during calculation using `std::mem::take()`, `.clear()` them for use, and store them back when done.

## 2024-05-15 - [Performance Insight: FFT memory allocation]
**Learning:** In the static extractor (`src/static_ext/extractor.rs`), the `r2c` processing logic created brand new `indata` and `outdata` vectors for every column processed (e.g. `vec![0.0; self.fft_size]`), causing significant overhead due to inner-loop heap allocations.
**Action:** Reused the `fft_in_buffer` and `fft_out_buffer` available on the `ColumnState` struct via `std::mem::take`, resizing them only when necessary, processing the FFT, and then restoring them back onto the `state` struct. NOTE: It is important to avoid `.to_vec()` on the returned `outdata` because `fft_complex` actually needs to return the values to the caller without stealing `outdata`. Oh wait, `complex_data` returns `Vec<Complex<f32>>` so `.to_vec()` is still currently allocating to give a copy to `fft_complex`.
## 2026-10-02 - Expanding Engine FFT Allocation Fix
Learning: The expanding engine was allocating a full_series.to_vec() every column in the FFT calculation loop for realfft, as well as making duplicate allocations with to_vec on fft_complex.
Action: Reused fft_in_buffer and fft_out_buffer for realfft processing and removed unneeded to_vec() by building the spectrum from the pre-sized output buffer, then cloning it safely at the end.
## 2026-10-03 - Avoiding to_vec() and fill() on reused buffers
**Learning:** When reusing buffers in  (like  via ), calling  to satisfy an ownership requirement (e.g., ) triggers a deep copy allocation anyway, causing a massive performance regression. Furthermore, calling  on a fully expanded buffer size is slower than zeroing only the required padding.
**Action:** Use careful slicing  to interact with processing functions and only explicitly zero-fill the padding index ranges to maintain true zero-allocation pooling.

## 2026-10-03 - Avoiding to_vec() and fill() on reused buffers
**Learning:** When reusing buffers in `ColumnState` (like `fft_out_buffer` via `std::mem::take`), calling `.to_vec()` to satisfy an ownership requirement (e.g., `SlidingDFT::from_fft()`) triggers a deep copy allocation anyway, causing a massive performance regression. Furthermore, calling `.fill(0.0)` on a fully expanded buffer size is slower than zeroing only the required padding.
**Action:** Use careful slicing `[..len]` to interact with processing functions and only explicitly zero-fill the padding index ranges to maintain true zero-allocation pooling.

## 2024-10-25 - [Single-Pass Covariance]
**Learning:** Calculating standard variance/covariance linear regression elements directly via `Σ(x_i * y_i) - n * x̄ * ȳ` dynamically inside a loop avoids allocating massive inner-loop temporary data `Vec`s required by two-pass map-reduce algorithms.
**Action:** Replace sequential linear-regression math with dynamic one-pass sums to save allocations.

## 2026-10-07 - [Performance Insight: Vector capacity preallocation]
**Learning:** For loops that unconditionally append a known amount of items to an empty vector (`Vec::push()`), failing to reserve capacity beforehand forces the allocator to resize the buffer up to $O(\log N)$ times.
**Action:** Always call `.reserve(len)` on empty buffers that are populated in known-length loops to execute a single $O(1)$ allocation, e.g. for features like `MedianDiff`.
## 2024-10-07 - Lempel-Ziv Complexity Bottleneck
**Learning:** The nested loops traversing Lempel-Ziv substrings incremented nodes one-by-one by starting from zero every time a new character was hit instead of maintaining state throughout a string, resulting in O(N^2) behaviour during worst-cases where substrings get long.
**Action:** By storing state (`ind`, `curr_node`) instead of restarting string searches and simply breaking when `ind + inc > n`, complexity is dramatically reduced resulting in near 5x to 10x faster execution for long sequences.
## 2024-10-08 - Inner loop vector allocations in permutation_entropy
**Learning:** Repetitive inner-loop allocations, like `let mut inv_perm = vec![0usize; dim];` inside a loop iterating `num_shifts` times inside `permutation_entropy`, causes significant heap allocation overhead and slows down feature computation.
**Action:** Extract the vector allocation to before the loop and reuse it via overwriting, which leads to ~56% performance improvement.
## 2024-10-25 - Avoid vector allocations in feature finalizers
**Learning:** Functions like `spectral::finalize()` are evaluated sequentially in inner loops. Calling `Vec::with_capacity()` inside these hot-paths introduces significant heap allocation overhead and slows down throughput.
**Action:** Replace `Vec::with_capacity()` with pre-allocated buffer reuse (`std::mem::take`) from the shared `ColumnState` container, making sure to clear them before use and return them once the evaluation completes.
## 2026-10-11 - Unused workspace allocations
**Learning:** Variables maintained in state workspaces (like `workspace_f64_1` in `calc_dfa`) might actually be unused leftovers from earlier drafts or unneeded intermediate states. Writing to these arrays in tight loops costs memory bandwidth and compute for no reason.
**Action:** Always verify if an array written to inside a loop is actually consumed later in the function. If it is purely write-only, safely remove the array and all associated calculations to save memory writes and slightly improve performance.

## 2026-10-18 - Avoiding dynamic allocations in Levinson-Durbin
**Learning:** `PartialAutocorr` calculations dynamically allocated `Vec` and resized heavily inside a Levinson-Durbin implementation using `let mut phi = vec![0.0; stride * stride];`.
**Action:** Replaced with pre-allocated buffer `pacf_buffer` and `diff_buffer` in `ColumnState` and re-sized instead to significantly reduce inner loop allocations, resulting in roughly a 2.5x speedup for the feature processing overhead.
## 2026-10-11 - Unused variable warnings
**Learning:** Some optimizations like "Single-Pass Covariance" introduced unused variables (like `log_k_inv_sum`, `log_lk_sum`, `sum_x_sq`, `sum_xy` in `src/features/complexity.rs`) because the actual calculations using them were commented out, leaving unused mutable declarations. This creates noise and slight overhead.
**Action:** Remove unused variables and commented-out code blocks left over from previous optimization efforts.

## 2026-10-18 - Eliminating intermediate arrays in sliding window evaluation
**Learning:** Initializing intermediate vectors (like `dy = Vec::with_capacity(...)`) for localized sliding difference checks inside window calculations causes redundant heap allocations.
**Action:** Replace `Vec` allocations with lazy iterative evaluations directly indexing into the source array within the inner check loop (`arr[i+1] - arr[i]`) whenever intermediate values are only consumed immediately.
