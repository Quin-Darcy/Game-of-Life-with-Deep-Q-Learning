// Constants for the DQN
pub const LEARNING_RATE: f32 = 0.2;
pub const DISCOUNT_FACTOR: f32 = 0.2;
pub const BATCH_SIZE: f32 = 0.25;
pub const EPOCHS: usize = 10;
pub const TOTAL_EPISODES: usize = 2;
pub const EPISODES_PER_UPDATE: usize = 100000;

// Constants for the grid
pub const GRID_DIM: usize = 20;
pub const WINDOW_WIDTH_MAX: f32 = 800.0;
pub const WINDOW_HEIGHT_MAX: f32 = 800.0;

// Constants for the Model
pub const MAX_POPULATION_REPEATS: usize = 24;
pub const MAX_POPULATION_AGE: usize = 2000;

// Constants for the agent
pub const MAX_ALIVE_RATIO: f32 = 0.70;
pub const INITIAL_PROBABILITY: f32 = 0.0;
pub const MAX_STATE_SPACE_SIZE: usize = 820;

// Constants controlling exploration and exploitation
pub const EPSILON: f32 = 0.2;
pub const MAX_EPSILON : f32 = 0.8;
pub const MIN_EPSILON : f32 = 0.05;
pub const INCREASE_FACTOR : f32 = 200.0;
pub const DECREASE_FACTOR : f32 = 100.0;
pub const MAX_CYCLE_LENGTH: usize = 24;

// Constants for the GA
pub const TOURNAMENT_WINNERS_PERCENTAGE: f32 = 0.70;
pub const SELECTION_PRESSURE: f32 = 0.78;
pub const MUTATION_RATE: f32 = 0.20;
pub const CROSSOVER_RATE: f32 = 0.72;
pub const MAX_CROSSOVER_POINTS: f32 = 0.5;
pub const MAX_CROSSOVER_SECTION_SIZE: f32 = 0.5;
pub const MAX_MUTATION_POINTS: f32 = 0.3;

