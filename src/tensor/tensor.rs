#[derive(Debug, Clone, Default)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}
