use std::ops::{Add, Sub, Mul, Index, IndexMut};
use crate::data_structures::vector::Vector;
use rand::Rng;

/// A matrix is a vector of vectors.
/// It is represented as a two-dimensional array of numbers.
/// The matrix struct implements basic operations such as addition, subtraction, multiplication, and division.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: Vec<Vector>,
}

impl Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix {
        if self.shape() != other.shape() {
            panic!("Matrices must have the same shape for addition.");
        }
        Matrix { rows: self.rows.iter().zip(other.rows.iter()).map(|(a, b)| a + b).collect() }
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        if self.shape() != other.shape() {
            panic!("Matrices must have the same shape for addition.");
        }
        Matrix { rows: self.rows.iter().zip(other.rows.iter()).map(|(a, b)| a + b).collect() }
    }
}

impl Sub<Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, other: Matrix) -> Matrix {
        if self.shape() != other.shape() {
            panic!("Matrices must have the same shape for subtraction.");
        }
        Matrix { rows: self.rows.iter().zip(other.rows.iter()).map(|(a, b)| a - b).collect() }
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Matrix {
        if self.shape() != other.shape() {
            panic!("Matrices must have the same shape for subtraction.");
        }
        Matrix { rows: self.rows.iter().zip(other.rows.iter()).map(|(a, b)| a - b).collect() }
    }
}

impl Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Matrix {
        Matrix { rows: self.rows.iter().map(|x| x * scalar).collect() }
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Matrix {
        Matrix { rows: self.rows.iter().map(|x| x * scalar).collect() }
    }
}

impl Mul<Vector> for Matrix {
    type Output = Vector;

    fn mul(self, other: Vector) -> Vector {
        if self.rows[0].elements.len() != other.elements.len() {
            panic!("The number of columns in the first matrix must be equal to the number of elements in the vector for multiplication.");
        }
        Vector { elements: self.rows.iter().map(|row| row.elements.iter().zip(other.elements.iter()).map(|(a, b)| a * b).sum()).collect() }
    }
}

impl Mul<&Vector> for &Matrix {
    type Output = Vector;

    fn mul(self, other: &Vector) -> Vector {
        if self.rows[0].elements.len() != other.elements.len() {
            panic!("The number of columns in the first matrix must be equal to the number of elements in the vector for multiplication.");
        }
        Vector { elements: self.rows.iter().map(|row| row.elements.iter().zip(other.elements.iter()).map(|(a, b)| a * b).sum()).collect() }
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix {
        if self.rows[0].elements.len() != other.rows.len() {
            panic!("The number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication.");
        }
        let mut result = vec![];
        for i in 0..self.rows.len() {
            let mut row = vec![];
            for j in 0..other.rows[0].elements.len() {
                let mut sum = 0.0;
                for k in 0..self.rows[0].elements.len() {
                    sum += self.rows[i].elements[k] * other.rows[k].elements[j];
                }
                row.push(sum);
            }
            result.push(Vector { elements: row });
        }
        Matrix { rows: result }
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix {
        if self.rows[0].elements.len() != other.rows.len() {
            panic!("The number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication.");
        }
        let mut result = vec![];
        for i in 0..self.rows.len() {
            let mut row = vec![];
            for j in 0..other.rows[0].elements.len() {
                let mut sum = 0.0;
                for k in 0..self.rows[0].elements.len() {
                    sum += self.rows[i].elements[k] * other.rows[k].elements[j];
                }
                row.push(sum);
            }
            result.push(Vector { elements: row });
        }
        Matrix { rows: result }
    }
}

impl Index<usize> for Matrix {
    type Output = Vector;

    fn index(&self, index: usize) -> &Vector {
        &self.rows[index]
    }
}

impl Index<usize> for &Matrix {
    type Output = Vector;

    fn index(&self, index: usize) -> &Vector {
        &self.rows[index]
    }
}

impl Index<usize> for &mut Matrix {
    type Output = Vector;

    fn index(&self, index: usize) -> &Vector {
        &self.rows[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Vector {
        &mut self.rows[index]
    }
}

impl IndexMut<usize> for &mut Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Vector {
        &mut self.rows[index]
    }
}

impl Matrix {
    /// Create a new matrix with the given elements.
    pub fn new(elements: Vec<Vector>) -> Matrix {
        Matrix { rows: elements }
    }

    /// Create a new matrix with the given size and all elements set to zero.
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix { rows: vec![Vector { elements: vec![0.0; cols] }; rows] }
    }

    /// Create a new matrix with the given size and all elements set to random values.
    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::thread_rng();
        Matrix { rows: vec![Vector { elements: (0..cols).map(|_| rng.gen_range(0.0..1.0)).collect() }; rows] }
    }

    /// Create a new matrix with the given size and all elements set to one.
    pub fn ones(rows: usize, cols: usize) -> Matrix {
        Matrix { rows: vec![Vector { elements: vec![1.0; cols] }; rows
        ] }
    }

    /// Create a new identity matrix with the given size.
    pub fn identity(size: usize) -> Matrix {
        let mut elements = vec![];
        for i in 0..size {
            let mut row = vec![];
            for j in 0..size {
                if i == j {
                    row.push(1.0);
                } else {
                    row.push(0.0);
                }
            }
            elements.push(Vector { elements: row });
        }
        Matrix { rows: elements }
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Matrix {
        let mut result: Vec<Vector> = vec![];
        for j in 0..self.rows[0].elements.len() {
            let mut row: Vec<f64> = vec![];
            for i in 0..self.rows.len() {
                row.push(self.rows[i].elements[j]);
            }
            result.push(Vector { elements: row });
        }
        Matrix { rows: result }
    }

    /// Returns the shape of the matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows.len(), self.rows[0].elements.len())
    }

    /// Sets the elements in the given row to the given vector.
    pub fn set_row(&mut self, row: usize, vector: Vector) {
        if vector.elements.len() != self.rows[row].elements.len() {
            panic!("The number of elements in the vector must be equal to the number of columns in the matrix.");
        }
        self.rows[row] = vector;
    }

    /// Sets the elements in the given column to the given vector.
    pub fn set_col(&mut self, col: usize, vector: Vector) {
        if vector.elements.len() != self.rows.len() {
            panic!("The number of elements in the vector must be equal to the number of rows in the matrix.");
        }
        for i in 0..self.rows.len() {
            self.rows[i].elements[col] = vector.elements[i];
        }
    }

    /// Returns a column in the matrix.
    pub fn get_col(&self, col: usize) -> Vector {
        Vector { elements: self.rows.iter().map(|row| row.elements[col]).collect() }
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut result = String::new();
        for row in &self.rows {
            result.push_str(&format!("{}\n", row));
        }
        write!(f, "{}", result)
    }
}