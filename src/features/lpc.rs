use std::f32::consts::PI;

const ORDER: usize = 11;
pub const N_LPCC: usize = 12;

pub fn compute_lpcc(values: &[f32]) -> [f32; N_LPCC] {
    let order = ORDER;
    let n_coeff = N_LPCC;

    if values.is_empty() {
        return [0.0; N_LPCC];
    }

    // Uncentred autocorrelation (as TSFEL), not the mean-centred FFT one other
    // features share: that would make LPCC depend on which features are requested.
    let mut r = [0.0f32; ORDER + 1];
    for (lag, r_lag) in r.iter_mut().enumerate().take(values.len()) {
        let s: f64 = values[lag..]
            .iter()
            .zip(values)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        *r_lag = s as f32;
    }

    let mut a = [0.0f32; ORDER + 1];
    let mut e = r[0];

    a[0] = 1.0;

    if e.abs() < 1e-9 {
        return [0.0; N_LPCC];
    }

    for i in 1..=order {
        let mut sum = 0.0;
        for j in 1..i {
            sum += a[j] * r[i - j];
        }
        let k = (r[i] - sum) / e;

        let mut a_new = a;
        a_new[i] = k;
        for j in 1..i {
            a_new[j] = a[j] - k * a[i - j];
        }
        a = a_new;
        e *= 1.0 - k * k;
        if e.abs() < 1e-9 {
            break;
        }
    }

    let mut lpc_coeffs = [0.0f32; N_LPCC];
    lpc_coeffs[0] = 1.0;
    for i in 1..n_coeff {
        lpc_coeffs[i] = -a[i];
    }

    let n = lpc_coeffs.len();
    let mut fft_real = [0.0f32; N_LPCC];
    let mut fft_imag = [0.0f32; N_LPCC];

    for k in 0..n {
        for t in 0..n {
            let angle = -2.0 * PI * (k as f32) * (t as f32) / (n as f32);
            fft_real[k] += lpc_coeffs[t] * angle.cos();
            fft_imag[k] += lpc_coeffs[t] * angle.sin();
        }
    }

    let mut log_power = [0.0f32; N_LPCC];
    for k in 0..n {
        let power = fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k];
        log_power[k] = if power > 1e-9 { power.ln() } else { 0.0 };
    }

    let mut ifft_real = [0.0f32; N_LPCC];
    let mut ifft_imag = [0.0f32; N_LPCC];

    for t in 0..n {
        for k in 0..n {
            let angle = 2.0 * PI * (k as f32) * (t as f32) / (n as f32);
            ifft_real[t] += log_power[k] * angle.cos();
            ifft_imag[t] += log_power[k] * angle.sin();
        }
        ifft_real[t] /= n as f32;
        ifft_imag[t] /= n as f32;
    }

    let mut lpcc = [0.0f32; N_LPCC];
    for i in 0..n {
        lpcc[i] = (ifft_real[i] * ifft_real[i] + ifft_imag[i] * ifft_imag[i]).sqrt();
    }
    lpcc
}
