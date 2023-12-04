use crate::layer::Layer;
use crate::errors::NeuralNetError;

#[derive(Clone)]
pub struct NeuralNet {
    num_layers: usize,
    pub layer_layout: Vec<usize>,
    pub layers: Vec<Layer>,
    pub learning_rate: f32,
}

impl NeuralNet {
    pub fn new(num_layers: usize, layer_layout: Vec<usize>, learning_rate: f32) -> Self {
        // Initialize the vector of layers based on the number of layers and the layer layout
        let mut layers: Vec::<Layer> = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            // Layer is initialized with num_inputs (number of neurons in previous layer), 
            // num_outputs (number of neurons in current layer), and its layer index
            if i == 0 {
                layers.push(Layer::new(layer_layout[i], layer_layout[i], i));
            } else {
                layers.push(Layer::new(layer_layout[i-1], layer_layout[i], i));
            }
        }

        Self { num_layers, layer_layout, layers, learning_rate }
    }

    pub fn feed_forward(&mut self, input: &mut Vec<f32>) -> Result<Vec<f32>, NeuralNetError> {
        // Check if resultant vector is empty
        if input.is_empty() {
            return Err(NeuralNetError::EmptyVector {
                message: "feed_forward received Experience with empty state".to_string(),
                line: line!(),
                file: file!().to_string(),
            });
        }

        // Check that the length of the input vector matches the number of neurons in the input layer
        if input.len() != self.layer_layout[0] {
            return Err(NeuralNetError::InvalidDimensions {
                message: format!("feed_forward received Experience with invalid state length. Expected: {}, Received: {}", self.layer_layout[0], input.len()),
                line: line!(),
                file: file!().to_string(),
            });
        }

        let mut layer_output: Vec<f32> = Vec::new();
        let mut nn_output: Vec<f32> = Vec::with_capacity(self.layer_layout[self.num_layers-1]);

        for i in 0..self.num_layers {
            if i == 0 {
                layer_output = self.layers[i].feed_forward(input)?;
            } else if i == self.num_layers - 1 {
                nn_output = self.layers[i].feed_forward(&layer_output)?;
            } else {
                layer_output = self.layers[i].feed_forward(&layer_output)?;
            }
        }

        // Check that nn_output length matches number of neurons in output layer
        if nn_output.len() != self.layer_layout[self.num_layers - 1] {
            return Err(NeuralNetError::InvalidReturnLength {
                message: "output of feed_forward has invalid length".to_string(),
                line: line!(),
                file: file!().to_string(),
            });
        }

        // Return the output of the neural net
        Ok(nn_output)
    }

    pub fn backwards_propagate(
        &mut self, 
        target: &mut Vec<f32>, 
        total_bias_gradients: &mut Vec<Vec<f32>>, 
        total_weight_gradients: &mut Vec<Vec<f32>>
    ) -> Result<(), NeuralNetError> {

        let mut layer_neuron_errors: Vec<f32> = Vec::new();
        let mut layer_bias_gradients: Vec<f32>;
        let mut layer_weight_gradients: Vec<f32>;

        let temp_vec: Vec<f32> = vec![0.0; 1];

        // Check if target is empty
        if target.is_empty() {
            return Err(NeuralNetError::EmptyVector {
                message: "backwards_propagate received Experience with empty target".to_string(),
                line: line!(),
                file: file!().to_string(),
            })
        }

        for i in (1..self.num_layers).rev() {
            if i == self.num_layers-1 {
                layer_neuron_errors = self.layers[i].calculate_neuron_errors(target, &temp_vec, &temp_vec, self.num_layers)?;
            } else {
                let next_layer_weights = self.layers[i+1].weights.clone();
                layer_neuron_errors = self.layers[i].calculate_neuron_errors(target, &next_layer_weights, &layer_neuron_errors, self.num_layers)?;
            }

            let previous_layer_output = self.layers[i-1].output.clone();
            layer_bias_gradients = layer_neuron_errors.clone();
            layer_weight_gradients = self.layers[i].calculate_weight_gradients(&previous_layer_output)?;

            // Add each component of the layer_bias_gradients and layer_weights_gradients to the total_bias_gradients and total_weight_gradients
            for (a, &b) in total_bias_gradients[i-1].iter_mut().zip(layer_bias_gradients.iter()) {
                *a += b;
            }

            for (a, &b) in total_weight_gradients[i-1].iter_mut().zip(layer_weight_gradients.iter()) {
                *a += b;
            }
        }

        Ok(())
    }
}


#[cfg(test)]
mod neural_net_tests {
    use super::*;

    #[test]
    fn test_neural_net_new() {
        let neural_net = NeuralNet::new(3, vec![2, 3, 1], 0.01);

        assert_eq!(neural_net.num_layers, 3);
        assert_eq!(neural_net.layer_layout, vec![2, 3, 1]);
        assert_eq!(neural_net.learning_rate, 0.01);
        assert_eq!(neural_net.layers.len(), 3);
    }

    // TODO: Add more unit tests
}
