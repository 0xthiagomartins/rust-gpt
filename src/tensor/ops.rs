use super::tensor::Tensor;

pub fn zeros(shape: &[usize]) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor {
        data: vec![0.0; len],
        shape: shape.to_vec(),
    }
}
