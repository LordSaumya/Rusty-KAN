use std::ops::{Add, Sub, Mul, Div};
use serde::{Serialize, Deserialize};
use crate::data_structures::matrix::Matrix;
use rand::Rng;

/// A vector is a one-dimensional array of numbers.
/// It is represented as a list of elements.
/// The vector struct implements basic operations such as addition, subtraction, multiplication, and division.
/// It also provides methods to calculate the dot product, cross product, and convert to a matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    pub elements: Vec<f64>,
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, other: Vector) -> Vector {
        Vector { elements: self.elements.iter().zip(other.elements.iter()).map(|(a, b)| a + b).collect() }
    }
}

impl Sub for Vector {
    type Output = Vector;

    fn sub(self, other: Vector) -> Vector {
        Vector { elements: self.elements.iter().zip(other.elements.iter()).map(|(a, b)| a - b).collect() }
    }
}

impl Mul<f64> for Vector {
    type Output = Vector;

    fn mul(self, scalar: f64) -> Vector {
        Vector { elements: self.elements.iter().map(|x| x * scalar).collect() }
    }
}

impl Div<f64> for Vector {
    type Output = Vector;

    fn div(self, scalar: f64) -> Vector {
        Vector { elements: self.elements.iter().map(|x| x / scalar).collect() }
    }
}

impl Vector {
    pub fn new(elements: Vec<f64>) -> Vector {
        Vector { elements }
    }

    pub fn zeros(size: usize) -> Vector {
        Vector { elements: vec![0.0; size] }
    }

    pub fn ones(size: usize) -> Vector {
        Vector { elements: vec![1.0; size] }
    }

    pub fn random(size: usize) -> Vector {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        Vector { elements: (0..size).map(|_| rng.gen_range(0.0..1.0)).collect() }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn dot(&self, other: &Vector) -> f64 {
        self.elements.iter().zip(other.elements.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn cross(&self, other: &Vector) -> Vector {
        let mut result = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            result.push(self.elements[(i + 1) % self.len()] * other.elements[(i + 2) % self.len()] - self.elements[(i + 2) % self.len()] * other.elements[(i + 1) % self.len()]);
        }
        Vector::new(result)
    }

    pub fn to_matrix(&self) -> Matrix {
        Matrix::new(vec![self.elements.clone()])
    }
}
