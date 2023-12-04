use bitvec::prelude::*;

#[derive(Clone)]
pub struct Experience {
    pub state: BitVec,
    pub action: usize, // Represents index in state bitvec to toggle on/off
    pub reward: f32,
    pub new_state: BitVec,
}

impl Experience {
    pub fn new(state: BitVec, action: usize, reward: f32, new_state: BitVec) -> Self {
        Self { state, action, reward, new_state }
    }

    // This method is needed so that regardless of how Experience is defined, we have a way
    // of converting it into a form which can be consumed by the neural net
    pub fn convert_state_to_f32_vec(&mut self) -> Vec<f32> {
        self.state.iter()
            .map(|bit| if *bit { 1.0 } else { 0.0 })
            .collect()
    }

    pub fn convert_new_state_to_f32_vec(&mut self) -> Vec<f32> {
        self.new_state.iter()
            .map(|bit| if *bit { 1.0 } else { 0.0 })
            .collect()
    }
}