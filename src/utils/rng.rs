#[derive(Debug, Clone)]
pub struct XorShift64 {
    state: u64,
    spare_normal: Option<f32>,
}

impl XorShift64 {
    const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { Self::DEFAULT_SEED } else { seed };
        Self {
            state,
            spare_normal: None,
        }
    }

    pub fn reseed(&mut self, seed: u64) {
        self.state = if seed == 0 { Self::DEFAULT_SEED } else { seed };
        self.spare_normal = None;
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    pub fn uniform_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    pub fn uniform_f32(&mut self) -> f32 {
        self.uniform_f64() as f32
    }

    pub fn gen_range_usize(&mut self, start: usize, end: usize) -> usize {
        assert!(start < end, "invalid range: start must be < end");
        let span = end - start;
        start + (self.uniform_f64() * span as f64).floor() as usize
    }

    pub fn normal(&mut self, mean: f32, std_dev: f32) -> f32 {
        assert!(mean.is_finite(), "mean must be finite");
        assert!(std_dev.is_finite(), "std_dev must be finite");
        assert!(std_dev >= 0.0, "std_dev must be >= 0");

        if std_dev == 0.0 {
            return mean;
        }

        if let Some(z1) = self.spare_normal.take() {
            return mean + std_dev * z1;
        }

        let u1 = self.uniform_f32().max(1e-7);
        let u2 = self.uniform_f32();

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;

        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        self.spare_normal = Some(z1);

        mean + std_dev * z0
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.gen_range_usize(0, i + 1);
            items.swap(i, j);
        }
    }
}

pub fn shuffle<T>(items: &mut [T], rng: &mut XorShift64) {
    rng.shuffle(items);
}

#[cfg(test)]
mod tests {
    use super::{shuffle, XorShift64};

    #[test]
    fn deterministic_for_same_seed() {
        let mut a = XorShift64::new(42);
        let mut b = XorShift64::new(42);

        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_values_are_within_range() {
        let mut rng = XorShift64::new(7);

        for _ in 0..4096 {
            let v = rng.uniform_f32();
            assert!(v >= 0.0);
            assert!(v < 1.0);
        }
    }

    #[test]
    fn gen_range_usize_stays_in_bounds() {
        let mut rng = XorShift64::new(123);

        for _ in 0..4096 {
            let value = rng.gen_range_usize(10, 25);
            assert!((10..25).contains(&value));
        }
    }

    #[test]
    fn shuffle_keeps_same_elements() {
        let mut rng = XorShift64::new(55);
        let mut values = [1, 2, 3, 4, 5, 6, 7, 8];

        shuffle(&mut values, &mut rng);

        let mut sorted = values;
        sorted.sort_unstable();
        assert_eq!(sorted, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn normal_samples_are_finite_and_centered() {
        let mut rng = XorShift64::new(999);
        let n = 5000;
        let mut sum = 0.0f32;

        for _ in 0..n {
            let sample = rng.normal(0.0, 1.0);
            assert!(sample.is_finite());
            sum += sample;
        }

        let mean = sum / n as f32;
        assert!(mean.abs() < 0.25, "mean too far from zero: {mean}");
    }
}
