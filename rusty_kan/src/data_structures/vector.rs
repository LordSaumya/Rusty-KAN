use std::{ops::{Add, Div, Index, IndexMut, Mul, Sub}, iter::{Iterator, Map}};
use serde::{Serialize, Deserialize};
use crate::data_structures::matrix::Matrix;
use rand::Rng;

/// A vector is a one-dimensional array of numbers.
/// It is represented as a list of elements.
/// The vector struct implements basic operations such as addition, subtraction, multiplication, and division.
/// It also provides methods to calculate the dot product, cross product, element-wise product, and convert to a matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    pub elements: Vec<f64>,
}

impl Add<Vector> for Vector {
    type Output = Vector;

    fn add(self, other: Vector) -> Vector {
        Vector { elements: self.elements.iter().zip(other.elements.iter()).map(|(a, b)| a + b).collect() }
    }
}

impl Add<&Vector> for &Vector {
    type Output = Vector;

    fn add(self, other: &Vector) -> Vector {
        Vector { elements: self.elements.iter().zip(other.elements.iter()).map(|(a, b)| a + b).collect() }
    }
}

impl Sub<Vector> for Vector {
    type Output = Vector;

    fn sub(self, other: Vector) -> Vector {
        Vector { elements: self.elements.iter().zip(other.elements.iter()).map(|(a, b)| a - b).collect() }
    }
}

impl Sub<&Vector> for &Vector {
    type Output = Vector;

    fn sub(self, other: &Vector) -> Vector {
        Vector { elements: self.elements.iter().zip(other.elements.iter()).map(|(a, b)| a - b).collect() }
    }
}

impl Mul<f64> for Vector {
    type Output = Vector;

    fn mul(self, scalar: f64) -> Vector {
        Vector { elements: self.elements.iter().map(|x| x * scalar).collect() }
    }
}

impl Mul<f64> for &Vector {
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

impl Div<f64> for &Vector {
    type Output = Vector;

    fn div(self, scalar: f64) -> Vector {
        Vector { elements: self.elements.iter().map(|x| x / scalar).collect() }
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.elements[index]
    }
}

impl Index<usize> for &Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.elements[index]
    }
}

impl Index<usize> for &mut Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.elements[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.elements[index]
    }
}

impl IndexMut<usize> for &mut Vector {
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.elements[index]
    }
}

impl Vector {
    /// Create a new vector with the given elements.
    pub fn new(elements: Vec<f64>) -> Vector {
        Vector { elements }
    }

    /// Create a new vector with the given size and all elements set to zero.
    pub fn zeros(size: usize) -> Vector {
        Vector { elements: vec![0.0; size] }
    }

    /// Create a new vector with the given size and all elements set to one.
    pub fn ones(size: usize) -> Vector {
        Vector { elements: vec![1.0; size] }
    }

    /// Create a new vector with the given size and all elements set to random values between 0 and 1.
    pub fn random(size: usize) -> Vector {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        Vector { elements: (0..size).map(|_| rng.gen_range(0.0..1.0)).collect() }
    }

    /// Return the length of the vector.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Return the dot product of two vectors.
    pub fn dot(&self, other: &Vector) -> f64 {
        self.element_wise(other).elements.iter().sum()
    }

    /// Return the cross product of two vectors.
    pub fn cross(&self, other: &Vector) -> Vector {
        let mut result: Vec<f64> = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            result.push(self.elements[(i + 1) % self.len()] * other.elements[(i + 2) % self.len()] - self.elements[(i + 2) % self.len()] * other.elements[(i + 1) % self.len()]);
        }
        Vector::new(result)
    }

    /// Return the element-wise product of two vectors.
    pub fn element_wise(&self, other: &Vector) -> Vector {
        Vector { elements: self.elements.iter().zip(other.elements.iter()).map(|(a, b)| a * b).collect() }
    }

    /// Convert the vector to a matrix.
    pub fn to_matrix(&self) -> Matrix {
        Matrix::new(vec![self.clone()])
    }
}

impl Iterator for Vector {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.elements.pop()
    }
}