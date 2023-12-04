use bitvec::prelude::*;
use rand::Rng;

use crate::neural_net::NeuralNet;
use crate::agent::Agent;
use crate::experience::Experience;
use crate::errors::NeuralNetError;
use crate::utility::get_batches;
use crate::constants::{EPSILON, GRID_DIM, WINDOW_WIDTH_MAX, WINDOW_HEIGHT_MAX};


pub struct DQN {
    num_layers: usize,
    pub main_network: NeuralNet,
    target_network: NeuralNet,
    discount_factor: f32,
    agent: Agent,
}

impl DQN {
    pub fn new(num_layers: usize, layer_layout: Vec<usize>, learning_rate: f32, discount_factor: f32) -> Self {
        let main_network: NeuralNet = NeuralNet::new(num_layers, layer_layout, learning_rate);
        let target_network: NeuralNet = main_network.clone();
        let agent = Agent::new(EPSILON, GRID_DIM.pow(2));

        DQN { num_layers, main_network, target_network, discount_factor, agent }
    }

    pub fn run_training(&mut self, total_episodes: usize, episodes_per_update: usize, epochs: usize, batch_size: f32) -> Result<(), NeuralNetError> {
        let mut replay_buffer: Vec<Experience>;

        for i in 0..(total_episodes) {
            println!("Episode: {}", i);
            println!("    Interacting ...");
            replay_buffer = self.interact(episodes_per_update);

            let num_batches = (replay_buffer.len() as f32 * batch_size) as usize;

            println!("    Training ...");
            self.train(replay_buffer.clone(), num_batches, epochs)?;
        }

        Ok(())
    }

    pub fn interact(&mut self, episodes: usize) -> Vec<Experience> {
        let mut rng = rand::thread_rng();
        let mut replay_buffer: Vec<Experience> = Vec::new();

        // This loops calls the agent and has it populate its population of grid states
        // Each episode will either result in a new state added to the population (explore) 
        // or the population will be evolved using the agent's GA
        for i in 0..episodes {
            println!("        Episode: {}", i);
            // Decide if the agent should explore or exploit
            let explore = (rng.gen::<f32>() < self.agent.epsilon) || (self.agent.state_space.len() < 5);
            
            // If the agent is exploring, get a new state from the agent
            // Otherwise, generate new states by evolving the state space
            if explore {
                self.agent.explore();
            } else {
                self.agent.exploit();
            };

            // Update agent - With new states having been added to the state space, we need to update the agent
            self.agent.update(WINDOW_WIDTH_MAX as usize, WINDOW_HEIGHT_MAX as usize);
        }

        for (state, &reward) in self.agent.state_space.iter() {
            let action = rng.gen_range(0..GRID_DIM.pow(2));

            // Create a modified state by toggling the bit represented by the number in action
            let mut modified_state = state.clone();
            
            let current_value = if let Some(value) = modified_state.get(action) {
                *value
            } else {
                false
            };

            modified_state.set(action, !current_value);

            let experience = Experience {
                state: modified_state,
                action, 
                reward,
                new_state: state.clone(),
            };

            replay_buffer.push(experience);
        }

        return replay_buffer;
    }

    pub fn train(&mut self, replay_buffer: Vec<Experience>, num_batches: usize, epochs: usize) -> Result<(), NeuralNetError> {
        // Check that training data is not empty
        if replay_buffer.is_empty() {
            return Err(NeuralNetError::EmptyVector {
                message: "train received empty set of training data".to_string(),
                line: line!(), 
                file: file!().to_string(),
            })
        }

        // Check that num_batches is not 0
        if num_batches == 0 {
            return Err(NeuralNetError::InvalidDimensions {
                message: "train received num_batches value of 0".to_string(), 
                line: line!(),
                file: file!().to_string(),
            })
        }

        // Check that num_batches is not greater than length of training data
        if num_batches > replay_buffer.len() {
            return Err(NeuralNetError::InvalidDimensions {
                message: "train received num_batches with value greater than size of training data".to_string(),
                line: line!(),
                file: file!().to_string(),
            })
        }

        // Check that epochs is not zero
        if epochs == 0 {
            return Err(NeuralNetError::InvalidDimensions {
                message: "train received epochs with value 0".to_string(),
                line: line!(),
                file: file!().to_string(),
            })
        }

        let mut batches: Vec<Vec<Experience>> = get_batches(&replay_buffer, num_batches)?;

        // Train the neural net for the specified number of epochs
        let mut batch_number = 0;
        for i in 0..epochs {
            println!("        Epoch: {}", i);
            for batch in batches.iter_mut() {
                println!("            Batch: {}", batch_number);
                self.train_on_batch(batch)?;
                batch_number += 1;
            }
            batch_number = 0;
            self.target_network = self.main_network.clone();
        }

        Ok(())
    }

    fn train_on_batch(&mut self, batch: &mut [Experience]) -> Result<(), NeuralNetError> {
        // Check the length of the batch
        if batch.is_empty() {
            return Err(NeuralNetError::EmptyVector{
                message: "train_on_batch received empty batch".to_string(),
                line: line!(),
                file: file!().to_string(),
            });
        }
    
        // Create vectors to hold the total weight and bias gradients for each layer
        let mut total_biases_gradients: Vec<Vec<f32>> = Vec::with_capacity(self.num_layers - 1);
        let mut total_weights_gradients: Vec<Vec<f32>> = Vec::with_capacity(self.num_layers - 1);

        // Initialize these to zero - Note that we are starting the loop at 1 since the input layer
        // does not have any weights or biases 
        for i in 1..self.num_layers {
            total_biases_gradients.push(vec![0.0; self.main_network.layer_layout[i]]);
            total_weights_gradients.push(vec![0.0; self.main_network.layer_layout[i-1] * self.main_network.layer_layout[i]]);
        }

        let num_experiences = batch.len();

        // Perform feed forward and back propagation and get update the bias and weight gradients
        let mut temp_target_vec: Vec<f32>;
        let mut q_pred_vec: Vec<f32>;
        let mut q_target_vec: Vec<f32>;

        for experience in batch.iter_mut() {
            q_pred_vec = self.main_network.feed_forward(&mut experience.convert_state_to_f32_vec())?;
            temp_target_vec = self.target_network.feed_forward(&mut experience.convert_new_state_to_f32_vec())?;
            q_target_vec = self.create_target(&mut q_pred_vec, &temp_target_vec, &experience.state, experience.action);

            self.main_network.backwards_propagate(&mut q_target_vec, &mut total_biases_gradients, &mut total_weights_gradients)?;
        }

        // Average the gradients and update the weights and biases of each layer
        for i in 1..self.num_layers {
            // Average the gradients
            let average_biases_gradients = total_biases_gradients[i-1].iter()
                .map(|sum| sum / num_experiences as f32)
                .collect::<Vec<f32>>();

            let average_weights_gradients = total_weights_gradients[i-1].iter()
                .map(|sum| sum / num_experiences as f32)
                .collect::<Vec<f32>>();

            // Scale gradients by the learning rate
            let scaled_biases_gradients = average_biases_gradients.iter()
                .map(|&grad| grad * self.main_network.learning_rate)
                .collect::<Vec<f32>>();

            let scaled_weights_gradients = average_weights_gradients.iter()
                .map(|&grad| grad * self.main_network.learning_rate)
                .collect::<Vec<f32>>();

            // Update the weights and biases of layer i using the averaged gradients of the ith layer's weights and biases
            self.main_network.layers[i].update(&scaled_weights_gradients, &scaled_biases_gradients)?;
        }

        Ok(())
    }

    // This method creates a vector based on a clone of the q_pred_vec
    // The max value is retrieved from the temp_target_vec while also noting the index of that value
    // The clone then has the component at that index replaced with the max value
    fn create_target(&mut self, q_pred_vec: &[f32], temp_target_vec: &[f32], state: &BitVec, action: usize) -> Vec<f32> {
        // First, find the index of the maximum value in temp_target_vec
        let max_index = temp_target_vec.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0); // Default to 0 if temp_target_vec is empty


        // Get the max_q_value from the target network
        let max_q = temp_target_vec[max_index];

        // Compute the reward of the state
        let reward = self.agent.run_state(WINDOW_WIDTH_MAX as usize, WINDOW_HEIGHT_MAX as usize, state);

        // Use the Bellman Equation to compute the target Q-value
        let q_target = reward - self.discount_factor * max_q;
    
        // Clone q_pred_vec to create the target vector
        let mut q_target_vec = q_pred_vec.to_vec();
    
        // Replace the component at max_index in q_target_vec with the max value from temp_target_vec
        q_target_vec[action] = q_target;
    
        q_target_vec
    }    
}