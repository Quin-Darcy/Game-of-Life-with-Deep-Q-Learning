use crate::experience::Experience;
use crate::errors::NeuralNetError;


// Single f32 input sigmoid function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Vectorized sigmoid function
pub fn sigmoid_vec(inputs: &[f32]) -> Result<Vec<f32>, NeuralNetError> {
    if inputs.is_empty() {
        return Err(NeuralNetError::EmptyVector{
            message: "sigmoid_vec received empty inputs vector".to_string(),
            line: line!(),
            file: file!().to_string(),
        });
    }

    Ok(inputs.iter().map(|x| sigmoid(*x)).collect())
}

// Vectorized sigmoid derivative function
pub fn sigmoid_derivative_vec(inputs: &[f32]) -> Result<Vec<f32>, NeuralNetError>  {
    if inputs.is_empty() {
        return Err(NeuralNetError::EmptyVector { 
            message: "sigmoid_derivative_vec received empty inputs vector".to_string(), 
            line: line!(), 
            file: file!().to_string(), 
        });
    }

    Ok(inputs.iter().map(|x| {
        let sig = sigmoid(*x);
        sig * (1.0 - sig)
    }).collect())
}

pub fn sub_vec(vec1: &[f32], vec2: &[f32]) -> Result<Vec<f32>, NeuralNetError> {
    // Check if either vector is empty
    if vec1.is_empty() || vec2.is_empty() {
        return Err(NeuralNetError::EmptyVector {
            message: "sub_vec received one or two empty vectors".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Ensure vec1 and vec2 are of the same length to avoid panics
    if vec1.len() != vec2.len() {
        return Err(NeuralNetError::InvalidDimensions {
            message: "sub_vec received vectors with non-matching lengths".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Create a new vector with the same capacity as vec1
    let mut result = Vec::with_capacity(vec1.len());

    // Iterate over the elements of vec1 and vec2, subtract and push to result
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        result.push(a - b);
    }

    Ok(result)
}

pub fn hadamard_prod_vec(vec1: &[f32], vec2: &[f32]) -> Result<Vec<f32>, NeuralNetError> {
    // Check if arguments are empty
    if vec1.is_empty() || vec2.is_empty() {
        return Err(NeuralNetError::EmptyVector {
            message: "hadamard_prod_vec received one or two empty vectors".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Ensure vec1 and vec2 are the same length
    if vec1.len() != vec2.len() {
        return Err(NeuralNetError::InvalidDimensions {
            message: "hadamard_prod_vec received vectors with non-matching lengths".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Create new vector with capacity equal to size of first vector
    let mut result = Vec::with_capacity(vec1.len());

    // Iterate over the elements and perform element-wise multiplications
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        result.push(a * b);
    }

    Ok(result)
}

pub fn flat_matrix_vector_mult(flat_matrix: &[f32], v: &[f32], columns: usize, rows: usize) -> Result<Vec<f32>, NeuralNetError> {
    // Check if inputs are empty
    if flat_matrix.is_empty() || v.is_empty() {
        return Err(NeuralNetError::EmptyVector {
            message: "flat_matrix_vector_mult received one or two empty vectors".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Check bounds on matrix and vector
    if flat_matrix.len() != columns * rows {
        return Err(NeuralNetError::InvalidDimensions {
            message: "flat_matrix_vector_mult received invalid matrix or columns/rows values".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    if v.len() != rows {
        return Err(NeuralNetError::InvalidDimensions {
            message: "flat_matrix_vector_mult received vector with invalid length".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    let mut result = Vec::with_capacity(rows);
    let mut sum: f32 = 0.0;

    for i in 0..columns {
        for j in 0..rows {
            sum += v[j] * flat_matrix[j * columns + i];
        }
        result.push(sum);
        sum = 0.0;
    }

    // Check the dimensions of the resultant vector
    if result.len() != columns {
        return Err(NeuralNetError::InvalidDimensions {
            message: "flat_matrix_vector_mult calculated vector with incorrect length".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    Ok(result)
}

// This returns a flat matrix
pub fn outer_product(v1: &[f32], v2: &[f32]) -> Result<Vec<f32>, NeuralNetError> {
    // Check if either vector is empty
    if v1.is_empty() || v2.is_empty() {
        return Err(NeuralNetError::EmptyVector {
            message: "outer_product received one or two empty vectors".to_string(),
            line: line!(), 
            file: file!().to_string(),
        })
    }

    let mut result = Vec::with_capacity(v1.len() * v2.len());

    for &a in v1.iter() {
        for &b in v2.iter() {
            result.push(a * b);
        }
    }

    // Verify length of result
    if result.len() != v1.len() * v2.len() {
        return Err(NeuralNetError::InvalidDimensions {
            message: "outer_product calculated vector with incorrect length".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    Ok(result)
}

pub fn get_batches(training_data: &[Experience], num_batches: usize) -> Result<Vec<Vec<Experience>>, NeuralNetError> {
    // Check if training data is empty
    if training_data.is_empty() {
        return Err(NeuralNetError::EmptyVector {
            message: "get_batches received empty set of Experiences".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Check that num_batches is not 0
    if num_batches == 0 {
        return Err(NeuralNetError::InvalidDimensions {
            message: "get_batches received request for 0 batches".to_string(), 
            line: line!(), 
            file: file!().to_string(),
        })
    }

    // Check that num_batches is not greater than the length of training data
    if num_batches > training_data.len() {
        return Err(NeuralNetError::InvalidDimensions {
            message: "get_batches received request for more batches than the size of training data".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    let mut batches: Vec<Vec<Experience>> = Vec::with_capacity(num_batches);

    if training_data.len() % num_batches == 0 {
        // If the number of training data is divisible by the number of batches, 
        // then each batch will have the same number of training data
        let batch_size = training_data.len() / num_batches;
        for i in 0..num_batches {
            batches.push(training_data[i*batch_size..(i+1)*batch_size].to_vec());
        }
    } else {
        // If the number of training data is not divisible by the number of batches, 
        // then each batch will have the same number of training data except for the last batch
        let batch_size = training_data.len() / num_batches;
        for i in 0..num_batches-1 {
            batches.push(training_data[i*batch_size..(i+1)*batch_size].to_vec());
        }
        batches.push(training_data[(num_batches-1)*batch_size..].to_vec());
    }  

    Ok(batches)
}


#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::*;

    #[test]
    fn test_sigmoid() {
        let input1: f32 = 0.5;
        let input2: f32 = 12.0;
        let input3: f32 = -43.2;

        let expected_output1: f32 = 1.0 / (1.0 + (-input1).exp());
        let expected_output2: f32 = 1.0 / (1.0 + (-input2).exp());
        let expected_output3: f32 = 1.0 / (1.0 + (-input3).exp()); 

        assert!((sigmoid(input1) - expected_output1).abs() < f32::EPSILON);
        assert!((sigmoid(input2) - expected_output2).abs() < f32::EPSILON);
        assert!((sigmoid(input3) - expected_output3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sigmoid_vec() {
        let input_vec1: Vec<f32> = vec![0.0f32, 2.0, -2.0];
        let input_vec2: Vec<f32> = vec![0.01];

        let empty_vec: Vec<f32> = Vec::new();

        let expected_outputs1: Vec<f32> = input_vec1.iter().map(|&x| sigmoid(x)).collect::<Vec<f32>>();
        let expected_outputs2: Vec<f32> = input_vec2.iter().map(|&x| sigmoid(x)).collect::<Vec<f32>>();

        // Test successful cases
        assert_eq!(sigmoid_vec(&input_vec1).unwrap(), expected_outputs1);
        assert_eq!(sigmoid_vec(&input_vec2).unwrap(), expected_outputs2);

        // Test error case
        assert!(matches!(sigmoid_vec(&empty_vec), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "sigmoid_vec received empty inputs vector"
        ));
    }

    #[test]
    fn test_sigmoid_derivative_vec() {
        let input_vec1: Vec<f32> = vec![0.0f32, 2.0, -2.0];
        let input_vec2: Vec<f32> = vec![0.01];
        
        let empty_vec: Vec<f32> = Vec::new();

        let expected_outputs1: Vec<f32> = input_vec1.iter().map(|&x| {let s = sigmoid(x); s * (1.0 - s)}).collect::<Vec<f32>>();
        let expected_outputs2: Vec<f32> = input_vec2.iter().map(|&x| {let s = sigmoid(x); s * (1.0 - s)}).collect::<Vec<f32>>();

        // Test successful cases
        assert_eq!(sigmoid_derivative_vec(&input_vec1).unwrap(), expected_outputs1);
        assert_eq!(sigmoid_derivative_vec(&input_vec2).unwrap(), expected_outputs2);

        // Test error case
        assert!(matches!(sigmoid_derivative_vec(&empty_vec), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "sigmoid_derivative_vec received empty inputs vector"
        ));
    }

    #[test]
    fn test_sub_vec() {
        let input1_vec1: Vec<f32> = vec![1.2];
        let input1_vec2: Vec<f32> = vec![0.1];

        let input2_vec1: Vec<f32> = vec![3.5, 944.2, 13.0, -0.0001, 0.55234];
        let input2_vec2: Vec<f32> = vec![44.0098, 0.0, -1200.03404, 3.3, 1.01];

        let empty_vec: Vec<f32> = Vec::new();

        let mut expected_output1: Vec<f32> = Vec::with_capacity(input1_vec1.len());
        for (&a, &b) in input1_vec1.iter().zip(input1_vec2.iter()) {
            expected_output1.push(a - b);
        }

        let mut expected_output2: Vec<f32> = Vec::with_capacity(input2_vec1.len());
        for (&a, &b) in input2_vec1.iter().zip(input2_vec2.iter()) {
            expected_output2.push(a - b);
        }

        // Test successful cases
        assert_eq!(sub_vec(&input1_vec1, &input1_vec2).unwrap(), expected_output1);
        assert_eq!(sub_vec(&input2_vec1, &input2_vec2).unwrap(), expected_output2);

        // Test error cases
        assert!(matches!(sub_vec(&empty_vec, &input2_vec1), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "sub_vec received one or two empty vectors"
        ));

        assert!(matches!(sub_vec(&input2_vec1, &empty_vec), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "sub_vec received one or two empty vectors"
        ));

        assert!(matches!(sub_vec(&empty_vec, &empty_vec), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "sub_vec received one or two empty vectors"
        ));

        assert!(matches!(sub_vec(&input1_vec1, &input2_vec1), Err(NeuralNetError::InvalidDimensions{message, line: _, file: _})
            if message == "sub_vec received vectors with non-matching lengths"
        ));

    }

    #[test]
    fn test_hadamard_prod_vec() {
        let input1_vec1: Vec<f32> = vec![-3.0];
        let input1_vec2: Vec<f32> = vec![5.001];

        let input2_vec1: Vec<f32> = vec![1.0, 2.3, -34.00989, 0.0032];
        let input2_vec2: Vec<f32> = vec![-90.1, 4.0, 1.0, 22.0999];

        let empty_vec: Vec<f32> = Vec::new();

        let mut expected_output1: Vec<f32> = Vec::with_capacity(input1_vec1.len());
        for (&a, &b) in input1_vec1.iter().zip(input1_vec2.iter()) {
            expected_output1.push(a * b);
        } 

        let mut expected_output2: Vec<f32> = Vec::with_capacity(input2_vec1.len());
        for (&a, &b) in input2_vec1.iter().zip(input2_vec2.iter()) {
            expected_output2.push(a * b);
        }

        // Test successful cases
        assert_eq!(hadamard_prod_vec(&input1_vec1, &input1_vec2).unwrap(), expected_output1);
        assert_eq!(hadamard_prod_vec(&input2_vec1, &input2_vec2).unwrap(), expected_output2);

        // Test error cases
        assert!(matches!(hadamard_prod_vec(&empty_vec, &input2_vec1), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "hadamard_prod_vec received one or two empty vectors"
        ));

        assert!(matches!(hadamard_prod_vec(&input2_vec1, &empty_vec), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "hadamard_prod_vec received one or two empty vectors"
        ));

        assert!(matches!(hadamard_prod_vec(&empty_vec, &empty_vec), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "hadamard_prod_vec received one or two empty vectors"
        ));

        assert!(matches!(hadamard_prod_vec(&input1_vec1, &input2_vec1), Err(NeuralNetError::InvalidDimensions{message, line: _, file: _})
            if message == "hadamard_prod_vec received vectors with non-matching lengths"
        ));

    }

    #[test]
    fn test_flat_matrix_vector_mult() {
        let input1_vec1: Vec<f32> = vec![1.0, 3.0, 4.0, 0.0, 1.0, 2.0];
        let input1_vec2: Vec<f32> = vec![2.0, 5.0];
        let input1_cols: usize = input1_vec1.len() / input1_vec2.len();
        let input1_rows: usize = input1_vec2.len();

        let input2_vec1: Vec<f32> = vec![1.0, 0.0, 3.0, 1.0, 4.0, 2.0];
        let input2_vec2: Vec<f32> = vec![1.0, 3.0, 2.0];
        let input2_cols: usize = input2_vec1.len() / input2_vec2.len();
        let input2_rows: usize = input2_vec2.len();

        let empty_vec: Vec<f32> = Vec::new();

        let expected_output1: Vec::<f32> = vec![2.0, 11.0, 18.0];
        let expected_output2: Vec::<f32> = vec![18.0, 7.0];

        // Test successful cases
        assert_eq!(flat_matrix_vector_mult(&input1_vec1, &input1_vec2, input1_cols, input1_rows).unwrap(), expected_output1);
        assert_eq!(flat_matrix_vector_mult(&input2_vec1, &input2_vec2, input2_cols, input2_rows).unwrap(), expected_output2);

        // Test error cases
        assert!(matches!(flat_matrix_vector_mult(&empty_vec, &input1_vec2, input1_cols, input1_rows), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "flat_matrix_vector_mult received one or two empty vectors"
        ));

        assert!(matches!(flat_matrix_vector_mult(&input1_vec1, &empty_vec, input1_cols, input1_rows), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "flat_matrix_vector_mult received one or two empty vectors"
        ));

        assert!(matches!(flat_matrix_vector_mult(&empty_vec, &empty_vec, input1_cols, input1_rows), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "flat_matrix_vector_mult received one or two empty vectors"
        ));

        assert!(matches!(flat_matrix_vector_mult(&input1_vec1, &input1_vec2, input1_rows, input1_rows), Err(NeuralNetError::InvalidDimensions{message, line: _, file: _})
            if message == "flat_matrix_vector_mult received invalid matrix or columns/rows values"
        ));

        assert!(matches!(flat_matrix_vector_mult(&input1_vec1, &input1_vec2, input1_cols, input1_cols), Err(NeuralNetError::InvalidDimensions{message, line: _, file: _})
            if message == "flat_matrix_vector_mult received invalid matrix or columns/rows values"
        ));

        assert!(matches!(flat_matrix_vector_mult(&input1_vec2, &input1_vec2, input1_cols, input1_rows), Err(NeuralNetError::InvalidDimensions{message, line: _, file: _})
            if message == "flat_matrix_vector_mult received invalid matrix or columns/rows values"
        ));

        assert!(matches!(flat_matrix_vector_mult(&input1_vec2, &input1_vec1, input1_rows, input1_cols), Err(NeuralNetError::InvalidDimensions{message, line: _, file: _})
            if message == "flat_matrix_vector_mult received invalid matrix or columns/rows values"
        ));

        assert!(matches!(flat_matrix_vector_mult(&input1_vec1, &input1_vec1, input1_rows, input1_cols), Err(NeuralNetError::InvalidDimensions{message, line: _, file: _})
            if message == "flat_matrix_vector_mult received vector with invalid length"
        ));
    }

    #[test]
    fn test_outer_product() {
        let input1_vec1: Vec<f32> = vec![3.0, 1.0, 4.0, 0.0, 2.0];
        let input1_vec2: Vec<f32> = vec![2.0, 0.0, 1.0];

        let input2_vec1: Vec<f32> = vec![2.0, 0.0, 1.0];
        let input2_vec2: Vec<f32> = vec![3.0, 1.0, 4.0, 0.0, 2.0];

        let empty_vec: Vec<f32> = Vec::new();

        let expected_output1: Vec<f32> = vec![6.0, 0.0, 3.0, 2.0, 0.0, 1.0, 8.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0];
        let expected_output2: Vec<f32> = vec![6.0, 2.0, 8.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 4.0, 0.0, 2.0];

        // Test successful cases
        assert_eq!(outer_product(&input1_vec1, &input1_vec2).unwrap(), expected_output1);
        assert_eq!(outer_product(&input2_vec1, &input2_vec2).unwrap(), expected_output2);

        // Test error cases
        assert!(matches!(outer_product(&empty_vec, &input2_vec1), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
            if message == "outer_product received one or two empty vectors"
        ));
    }

    // 
    // fn test_get_batches() {
    //     let state1 = bitvec![1, 0, 1];
    //     let new_state1 = bitvec![0, 1, 0];
    //     let experience1 = Experience::new(state1, new_state1);

    //     let state2 = bitvec![1, 1, 0];
    //     let new_state2 = bitvec![0, 0, 1];
    //     let experience2 = Experience::new(state2, new_state2);

    //     let training_data = vec![experience1.clone(), experience2.clone(), experience1.clone()];

    //     // Test successful case with even distribution
    //     let result_even = get_batches(&training_data, 2).unwrap();
    //     assert_eq!(result_even.len(), 2);
    //     assert_eq!(result_even[0].len(), 1);
    //     assert_eq!(result_even[1].len(), 2);

    //     // Test successful case with uneven distribution
    //     let result_uneven = get_batches(&training_data, 3).unwrap();
    //     assert_eq!(result_uneven.len(), 3);
    //     assert_eq!(result_uneven[0].len(), 1);
    //     assert_eq!(result_uneven[1].len(), 1);
    //     assert_eq!(result_uneven[2].len(), 1);

    //     // Test empty training data
    //     let empty_training_data: Vec<Experience> = Vec::new();
    //     assert!(matches!(get_batches(&empty_training_data, 1), Err(NeuralNetError::EmptyVector{message, line: _, file: _})
    //         if message == "get_batches received empty set of Experiences"
    //     ));

    //     // Test request for 0 batches
    //     assert!(matches!(get_batches(&training_data, 0), Err(NeuralNetError::InvalidDimensions{message, line: _, file: _})
    //         if message == "get_batches received request for 0 batches"
    //     ));

    //     // Test request for more batches than the size of training data
    //     assert!(matches!(get_batches(&training_data, 4), Err(NeuralNetError::InvalidDimensions{message, line: _, file: _})
    //         if message == "get_batches received request for more batches than the size of training data"
    //     ));
    // }
}