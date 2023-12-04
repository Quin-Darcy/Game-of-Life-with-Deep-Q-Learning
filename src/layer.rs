use rand::Rng;

use crate::utility::{sigmoid_vec, sub_vec, hadamard_prod_vec, sigmoid_derivative_vec, flat_matrix_vector_mult, outer_product};
use crate::errors::NeuralNetError;

#[derive(Debug, Clone)]
pub struct Layer {
    num_neurons: usize,
    num_inputs: usize,
    layer_index: usize,

    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    weighted_sums: Vec<f32>,
    pub output: Vec<f32>,
    neuron_errors: Vec<f32>,
}

impl Layer {
    // Generate a new layer with random weights and biases
    pub fn new(input_size: usize, output_size: usize, layer_index: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize layer information
        let num_neurons = output_size;
        let num_inputs = input_size;

        // Initialize the weights using Xavier initialization
        let weight_scale = (6.0 / (input_size + output_size) as f32).sqrt();

        let weights = (0..input_size * output_size)
            .map(|_| rng.gen_range(-weight_scale..weight_scale))
            .collect();
        
        // Initialize the biases to zero
        let biases = vec![0.0; num_neurons];

        // Initialize the weighted sums, output, neuron errors, and weight gradient terms to zero
        let weighted_sums = vec![0.0; num_neurons];
        let output = vec![0.0; num_neurons];
        let neuron_errors = vec![0.0; num_neurons];


        Self { num_neurons, num_inputs, layer_index, weights, biases, weighted_sums, output, neuron_errors }
    }

    // Feed the input of the previous layer into the current layer and return its output
    pub fn feed_forward(&mut self, input: &Vec<f32>) -> Result<Vec<f32>, NeuralNetError> {
        // Check for mismatched input lengths
        if input.len() != self.num_inputs {
            return Err(NeuralNetError::InvalidDimensions{
                message: format!("feed_forward received input with invalid length. Expected: {}, Received: {}", self.num_inputs, input.len()),
                line: line!(),
                file: file!().to_string(),
            });
        }

        let mut pre_activations = Vec::with_capacity(self.num_neurons);

        // Calculate weighted sum neuron by neuron
        for i in 0..self.num_neurons {
            // The bias is added first
            let mut neuron_output = self.biases[i];

            // This signifies the index for the group of weights corresponding to the current neuron
            let weight_base_index = i * self.num_inputs;

            // Create the linear combination of weights and inputs
            for j in 0..self.num_inputs {
                neuron_output += input[j] * self.weights[weight_base_index + j];
            }
            pre_activations.push(neuron_output);
        }
        
        // Set the pre-activation values (weighted_sums) and store the output for later reference
        self.weighted_sums = pre_activations;

        self.output = sigmoid_vec(&self.weighted_sums)?;

        return Ok(self.output.clone());
    }

    // This calculates the error in each neuron defined by the partial derivative of the cost function wrt. the weighted sum for that neuron
    // This gives back a vector of floats representing the magnitude of contribution each neuron had to the error in the cost function
    // It takes in the weights and neuron errors from the layer in front of it (closer to the output layer)
    pub fn calculate_neuron_errors(
        &mut self, 
        target: &[f32], 
        next_layer_weights: &[f32], 
        next_layer_error_terms: &[f32],
        num_layers: usize,
    ) -> Result<Vec<f32>, NeuralNetError> {

        // Check if target, weights, or error terms are empty
        if target.is_empty() || next_layer_weights.is_empty() || next_layer_error_terms.is_empty() {
            return Err(NeuralNetError::EmptyVector {
                message: "calculate_neuron_errors received empty arguments".to_string(),
                line: line!(),
                file: file!().to_string(),
            })
        }

        // Error terms are different for the last layer
        let errors = if self.layer_index == num_layers - 1 {
            hadamard_prod_vec(
                &sub_vec(&self.output, &target)?, 
                &sigmoid_derivative_vec(&self.weighted_sums)?
            )?
        } else {
            hadamard_prod_vec(
                // Performs matrix vector multiplication with transpose of next layer's weight matrix and the next layer's error terms
                &flat_matrix_vector_mult(
                    &next_layer_weights, 
                    &next_layer_error_terms, 
                    self.num_neurons, 
                    next_layer_error_terms.len())?, 
                &sigmoid_derivative_vec(&self.weighted_sums)?
            )?
        };

        // Validate that errors length is equal to num neurons
        if errors.len() != self.num_neurons {
            return Err(NeuralNetError::InvalidReturnLength {
                message: "Calculation of neuron errors returned incorrect number of values".to_string(),
                line: line!(),
                file: file!().to_string(),
            })
        }

        self.neuron_errors = errors.clone();
        Ok(errors)
    }

    // This method takes in the activation vector (output) from the previous layer (layer closer to the input layer) and computes the 
    // outer product between it and the vector of neuron errors from this layer. The result is a matrix in which each term corresponde to
    // the gradient for each weight in this layer
    pub fn calculate_weight_gradients(&mut self, input: &[f32]) -> Result<Vec<f32>, NeuralNetError> {
        // Check if input is empty
        if input.is_empty() {
            return Err(NeuralNetError::EmptyVector {
                message: "calculate_weight_gradients received empty input vector".to_string(),
                line: line!(),
                file: file!().to_string(),
            })
        }

        let weight_gradients = outer_product(input, &self.neuron_errors)?;

        // Validate the length of the returned gradients
        if weight_gradients.len() != self.weights.len() {
            return Err(NeuralNetError::InvalidReturnLength {
                message: "Calculated weight gradients resulted in vector with incorrect number of components".to_string(),
                line: line!(), 
                file: file!().to_string(),
            })
        }

        Ok(weight_gradients)
    }

    // This function receives two vectors consisting of the amount by which we need to adjust each weight and bias in this layer
    pub fn update(&mut self, weight_adjustments: &[f32], bias_adjustments: &[f32]) -> Result<(), NeuralNetError> {
        // Check that the weight adjustments and bias adjustments have the correct length 
        if weight_adjustments.len() != self.num_inputs * self.num_neurons {
            return Err(NeuralNetError::InvalidDimensions {
                message: "update received weight_adjustments vector of incorrect size".to_string(),
                line: line!(),
                file: file!().to_string(),
            })
        }

        if bias_adjustments.len() != self.num_neurons {
            return Err(NeuralNetError::InvalidDimensions {
                message: "update received bias_adjustments vector of incorrect size".to_string(),
                line: line!(),
                file: file!().to_string(),
            })
        }

        for (weight, adjustment) in self.weights.iter_mut().zip(weight_adjustments.iter()) {
            *weight -= adjustment;
        }

        for (bias, adjustment) in self.biases.iter_mut().zip(bias_adjustments.iter()) {
            *bias -= adjustment;
        }

        Ok(())
    }
}


#[cfg(test)]
mod layer_tests {
    use super::*;

    #[test]
    fn test_layer_new() {
        let layer = Layer::new(3, 2, 1);
        
        assert_eq!(layer.num_neurons, 2);
        assert_eq!(layer.num_inputs, 3);
        assert_eq!(layer.layer_index, 1);
        assert_eq!(layer.biases.len(), 2);
        assert_eq!(layer.weights.len(), 3 * 2); // input_size * output_size
    }

    #[test]
    fn test_layer_feed_forward() {
        let mut layer = Layer::new(3, 2, 0);
        let input = vec![0.5, 0.3, -0.2];
    
        // Normal case
        match layer.feed_forward(&input) {
            Ok(output) => assert_eq!(output.len(), 2),
            Err(_) => panic!("feed_forward failed on valid input"),
        }
    
        // Mismatched input length
        let invalid_input = vec![0.5, 0.3];
        assert!(matches!(layer.feed_forward(&invalid_input), Err(NeuralNetError::InvalidDimensions{..})));
    }

    #[test]
    fn test_layer_calculate_neuron_errors() {
        let mut layer = Layer::new(3, 2, 1);
        let target = vec![0.1, 0.2];
        let next_layer_weights = vec![0.2; 4]; // Assuming next layer also has 2 neurons
        let next_layer_error_terms = vec![0.05; 2];
    
        // Normal case
        match layer.calculate_neuron_errors(&target, &next_layer_weights, &next_layer_error_terms, 2) {
            Ok(errors) => assert_eq!(errors.len(), 2),
            Err(_) => panic!("calculate_neuron_errors failed on valid input"),
        }
    
        // Error case for empty input
        assert!(matches!(layer.calculate_neuron_errors(&[], &next_layer_weights, &next_layer_error_terms, 2), Err(NeuralNetError::EmptyVector{..})));
    }

    // TODO: Add more unit tests
    
}
