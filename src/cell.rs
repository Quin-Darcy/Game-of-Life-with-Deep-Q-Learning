use nannou::prelude::*;


#[derive(Debug, Clone)]
pub struct Cell {
    pub pos: Point2,
    pub state: bool,
}