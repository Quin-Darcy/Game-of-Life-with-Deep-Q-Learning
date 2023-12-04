use std::cmp::min;

mod ga;
mod agent;
mod cell;
mod constants;
mod dqn;
mod errors;
mod experience;
mod grid;
mod layer;
mod neural_net;
mod utility;

use crate::grid::Grid;
use crate::dqn::DQN;
use crate::agent::Agent;
use crate::constants::{
    WINDOW_WIDTH_MAX, 
    WINDOW_HEIGHT_MAX, 
    EPSILON, 
    MAX_POPULATION_REPEATS, 
    MAX_POPULATION_AGE, 
    GRID_DIM, 
    EPOCHS, 
    LEARNING_RATE, 
    DISCOUNT_FACTOR, 
    TOTAL_EPISODES, 
    EPISODES_PER_UPDATE, 
    BATCH_SIZE,
};

use nannou::prelude::*;
use bitvec::prelude::*;


struct Model {
    dqn: DQN,
    grid: Grid,

    population_repeats: usize,
    last_cycle_average: f32,
    cycle_average_repeats: usize,

    iterations: usize,

    reset_counter: usize,
}

fn model(app: &App) -> Model {
    let num_actions = GRID_DIM.pow(2);
    let layer_layout = vec![num_actions, num_actions + 100, num_actions - 100, num_actions];

    let learning_rate = LEARNING_RATE;
    let discount_factor = DISCOUNT_FACTOR;
    let total_episodes = TOTAL_EPISODES;
    let episodes_per_update = EPISODES_PER_UPDATE;
    let batch_size = BATCH_SIZE;
    let epochs = EPOCHS;

    let mut dqn = DQN::new(layer_layout.len(), layer_layout, learning_rate, discount_factor);
    dqn.run_training(total_episodes, episodes_per_update, epochs, batch_size).unwrap();

    let grid_state = create_grid_state(&mut dqn);

    let grid = Grid::new(WINDOW_WIDTH_MAX, WINDOW_HEIGHT_MAX, &grid_state);

    app.new_window()
        .size(WINDOW_WIDTH_MAX as u32, WINDOW_HEIGHT_MAX as u32)
        .resizable(true)
        .view(view)
        .event(window_event)
        .build()
        .unwrap();

    Model { dqn, grid, population_repeats: 0, last_cycle_average: -1.0, cycle_average_repeats: 0, iterations: 0, reset_counter: 0 }
}

fn window_event(app: &App, model: &mut Model, event: WindowEvent) {
    // Trigger new grid if window is resized or if mouse is clicked
    match event {
        WindowEvent::Resized(_new_size) => {
            let new_rect = app.window_rect();
            let w = min(new_rect.w() as usize, WINDOW_WIDTH_MAX as usize);
            let h = min(new_rect.h() as usize, WINDOW_HEIGHT_MAX as usize);

            // Reset the grid and initialize it to a new state from the agent
            let grid_state = create_grid_state(&mut model.dqn);

            // Reset the number of iterations
            model.iterations = 0;
            model.reset_counter += 1;

            model.grid = Grid::new(w as f32, h as f32, &grid_state);
        }
        WindowEvent::MousePressed(_button) => {
            let new_rect = app.window_rect();
            let w = min(new_rect.w() as usize, WINDOW_WIDTH_MAX as usize);
            let h = min(new_rect.h() as usize, WINDOW_HEIGHT_MAX as usize);

            // Reset the grid and initialize it to a new state from the agent
            let grid_state = create_grid_state(&mut model.dqn);

            // Reset the number of iterations
            model.iterations = 0;
            model.reset_counter += 1;

            model.grid = Grid::new(w as f32, h as f32, &grid_state);
        }
        _ => {}
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    // Increment the number of iterations
    model.iterations += 1;

    // Check if the population size has repeated cyclically
    if model.grid.cycle_average == model.last_cycle_average {
        model.cycle_average_repeats += 1;
    } else {
        model.last_cycle_average = model.grid.cycle_average;
        model.cycle_average_repeats = 0;
    }

    // Trigger new grid if population is zero or if the population size continues to repeat or if the population age is too high
    if model.grid.population == 0 || model.cycle_average_repeats >= MAX_POPULATION_REPEATS || model.grid.population_age >= MAX_POPULATION_AGE {
        let new_rect = app.window_rect();
        let w = min(new_rect.w() as usize, WINDOW_WIDTH_MAX as usize);
        let h = min(new_rect.h() as usize, WINDOW_HEIGHT_MAX as usize);

        // Reset population repeat counter
        model.population_repeats = 0;

        let grid_state = create_grid_state(&mut model.dqn);

        // Update the reset counter
        model.reset_counter += 1;

        // Reset grid
        model.grid = Grid::new(w as f32, h as f32, &grid_state);
    } else {
        // Update the grid and increase the population age
        model.grid.population_age += 1;
        model.grid.update();
    }
}

fn create_grid_state(dqn: &mut DQN) -> BitVec {
    let num_actions = GRID_DIM.pow(2);
    // Start with random grid state
    let mut temp_agent = Agent::new(EPSILON, num_actions);
    let initial_grid_state = temp_agent.get_new_state();

    // Convert the grid state into a Vec<f32>
    let mut grid_state_vec: Vec<f32> = initial_grid_state.iter()
            .map(|bit| if *bit { 1.0 } else { 0.0 })
            .collect();

    
    for _ in 0..num_actions * 4 {
        // Feed the grid state into the DQN network
        let output = dqn.main_network.feed_forward(&mut grid_state_vec).unwrap();

        // Get the index of the max q-value
        let index = output.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap();
    
        if grid_state_vec[index] == 1.0 {
            grid_state_vec[index] = 0.0;
        } else {
            grid_state_vec[index] = 1.0;
        }
    }

    // Convert the grid state back into a BitVec
    let mut new_grid = bitvec![0; grid_state_vec.len()];
    for i in 0..grid_state_vec.len() {
        if grid_state_vec[i] == 1.0 {
            new_grid.set(i, true);
        } else {
            new_grid.set(i, false);
        }
    }
    new_grid
}

fn view(app: &App, model: &Model, frame: Frame) {
    // Prepare to draw
    let draw = app.draw();

    // Set the background to black
    draw.background().color(BLACK);

    for (_, cell) in model.grid.cells.iter().enumerate() {
        // Determine the cell color based on its state and reset
        let cell_color = if cell.state { WHITE } else { BLACK };

        //let cell_color = if cell.state { WHITE } else { BLACK };

        let stroke_color = if cell.state { BLACK } else { WHITE };
    
        draw.rect()
            .xy(cell.pos)
            .w_h(model.grid.cell_width, model.grid.cell_height)
            .color(cell_color)
            .stroke(stroke_color)
            .stroke_weight(0.5);
    }    

    // Write to the window frame.
    draw.to_frame(app, &frame).unwrap();
}


fn main() {
    nannou::app(model).update(update).run();
}
