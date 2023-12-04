use nannou::prelude::*;
use bitvec::prelude::*;

use crate::cell::Cell;
use crate::constants::{GRID_DIM, MAX_CYCLE_LENGTH};

pub struct Grid {
    pub cells: Vec<Cell>,
    pub columns: usize,
    pub rows: usize,
    pub cell_width: f32,
    pub cell_height: f32,
    pub num_cells: usize,
    pub population: usize,
    pub population_age: usize,
    pub population_mean: f32,
    pub sum_sq_diff: f32,
    pub standard_deviation: f32,

    pub cycle_sum: usize,
    pub cycle_average: f32,

    // Store the initializing bit vector for the grid
    pub grid_state: BitVec,

    // Track the initial and final population of the grid
    pub initial_population: usize,
    pub final_population: usize,
}

impl Grid {
    pub fn new(window_width: f32, window_height: f32, grid_state: &BitVec) -> Self {
        // Calculate grid dimensions
        let cell_width = window_width / (GRID_DIM as f32);
        let cell_height = window_height / (GRID_DIM as f32);
        let cols = f32::floor(window_width / cell_width) as usize;
        let rows = f32::floor(window_height / cell_height) as usize;
        let x_padding = (window_width - (cols as f32 * cell_width)) / 2.0;
        let y_padding = (window_height - (rows as f32 * cell_height)) / 2.0;

        // Initialize grid
        let num_cells = cols * rows;
        let mut cells = Vec::with_capacity(num_cells);
        let mut population = 0;
        let population_age = 0;

        // Set positions for cells based on the dimensions of the grid and the values in the grid_state
        for y in 0..rows {
            for x in 0..cols {
                let x_pos = x_padding + x as f32 * cell_width - window_width / 2.0 + cell_width / 2.0;
                let y_pos = y_padding+ y as f32 * cell_height - window_height / 2.0 + cell_height / 2.0;
                let pos = pt2(x_pos, y_pos);
                let idx = y * cols + x;

                let state = grid_state[idx];

                if state {
                    population += 1;
                }

                cells.push(Cell { pos, state });
            }
        }

        Grid { 
            cells, 
            columns: cols, 
            rows, 
            cell_width, 
            cell_height, 
            num_cells, 
            population, 
            population_age, 
            population_mean: 0.0,
            cycle_average: 0.0,
            cycle_sum: 0,
            sum_sq_diff: 0.0,
            standard_deviation: 0.0,
            grid_state: grid_state.clone(), 
            initial_population: population, 
            final_population: 0 
        }
    }

    // This is solely the logic for the Game of Life 
    pub fn update(&mut self) {
        // This population has lived to see another day!
        self.population_age += 1;

        let mut new_states = vec![false; self.num_cells];

        // Calculate the average population size over the last MAX_CYCLE_LENGTH cycles
        // This tracks if the population is repeating in a cycle
        if self.population_age % MAX_CYCLE_LENGTH == 0 {
            self.cycle_average = self.cycle_sum as f32 / MAX_CYCLE_LENGTH as f32;
            self.cycle_sum = 0;
        } else {
            self.cycle_sum += self.population;
        }

        for y in 0..self.rows {
            for x in 0..self.columns {
                let idx = y * self.columns + x;
                let cell = &self.cells[idx];
                let live_neighbors = self.count_live_neighbors(x, y);

                new_states[idx] = match (cell.state, live_neighbors) {
                    (true, 2) | (true, 3) => true,
                    (true, _) => false,
                    (false, 3) => true,
                    (false, _) => false,
                };
            }
        }

        for (cell, &state) in self.cells.iter_mut().zip(new_states.iter()) {
            if state && !cell.state {
                self.population += 1;
            } else if !state && cell.state {
                self.population -= 1;
            }

            cell.state = state;
        }

        // Calulate standard deviation
        let delta = self.population as f32 - self.population_mean;
        self.population_mean += delta / self.population_age as f32;
        let delta2 = self.population as f32 - self.population_mean;
        self.sum_sq_diff += delta * delta2;

        if self.population_age > 1 {
            self.standard_deviation = (self.sum_sq_diff / (self.population_age - 1) as f32).sqrt();
        } else {
            self.standard_deviation = 0.0;
        }

        self.final_population = self.population;
    }

    fn count_live_neighbors(&self, x: usize, y: usize) -> usize {
        let mut live_neighbors = 0;

        for y_offset in -1..=1 {
            for x_offset in -1..=1 {
                if x_offset == 0 && y_offset == 0 {
                    continue;
                }

                let neighbor_x = x as i32 + x_offset;
                let neighbor_y = y as i32 + y_offset;

                if neighbor_x < 0 || neighbor_x >= self.columns as i32 {
                    continue;
                }

                if neighbor_y < 0 || neighbor_y >= self.rows as i32 {
                    continue;
                }

                let neighbor_idx = neighbor_y as usize * self.columns + neighbor_x as usize;

                if self.cells[neighbor_idx].state {
                    live_neighbors += 1;
                }
            }
        }

        live_neighbors
    }
}