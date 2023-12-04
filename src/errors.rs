#[derive(Debug)]
pub enum NeuralNetError {
    EmptyVector {message: String, line: u32, file: String},
    InvalidDimensions {message: String, line: u32, file: String},
    InvalidReturnLength {message: String, line: u32, file: String},
}