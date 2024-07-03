use std::collections::HashMap;
use crate::data_structures::vector::Vector;

/// A B-spline is a piecewise polynomial function that is used as a parameterised version of a univariate learnable activation function in a KAN.
/// It is represented as a list of control points, a list of knots, and a degree.
/// 
/// The B_Spline struct implements methods to evaluate the function at a given point and calculate the basis function.
/// The basis function is a recursive function that calculates the value of the B-spline at a given point.
/// The eval method calculates the value of the B-spline at a given point by summing the control points multiplied by the basis function. 

#[derive(Debug, Clone)]
pub struct BSpline {
    pub control_points: Vector, // Coefficients to be trained
    pub knots: Vector,
    pub degree: usize,
    pub memo: HashMap<String, f64>,
}

impl BSpline {
    /// Create a new 1D B-spline with a given list of control points and degree.
    /// 
    /// # Arguments
    /// 
    /// * `control_points` - A vector of control points.
    /// 
    /// * `degree` - The degree of the B-spline.
    /// 
    /// # Returns
    /// 
    /// * A B-spline with the given list of control points, specified degree, and uniform knots.
    pub fn new(control_points: Vector, degree: usize) -> BSpline {
        let n: usize = control_points.elements.len();
        let knots: Vector = Vector { elements: (0..n + degree + 1).map(|i| i as f64 / (n + degree) as f64).collect() };
        BSpline { control_points, knots, degree, memo: HashMap::new() }
    }

    /// Evaluate the B-spline at a given parameter value t.
    /// 
    /// # Arguments
    /// 
    /// * `t` - A parameter value between 0 and 1.
    /// 
    /// # Returns
    /// 
    /// * The value of the B-spline at the given parameter value t.
    pub fn eval(&mut self, t: f64) -> f64 {
        if t < 0.0 || t > 1.0 {
            panic!("Parameter value t must be between 0 and 1.");
        }

        let n: usize = self.control_points.len();
        let mut result: f64 = 0.0;
        for i in 0..n {
            result += self.control_points.elements[i] * self.basis(i, self.degree, t);
        }
        result
    }

    /// Calculate the basis function at a given index, degree, and parameter value t.
    /// 
    /// # Arguments
    /// 
    /// * `i` - An index.
    /// 
    /// * `degree` - The degree of the B-spline.
    /// 
    /// * `t` - A parameter value between 0 and 1.
    /// 
    /// # Returns
    /// 
    /// * The value of the basis function at the given index, degree, and parameter value t.
    pub fn basis(&mut self, i: usize, degree: usize, t: f64) -> f64 {
        let hashmap_key: String = i.to_string() + " " + &degree.to_string() + " " + &t.to_string();
        if let Some(&result) = self.memo.get(hashmap_key.as_str()) {
            return result;
        }
        if t < 0.0 || t > 1.0 {
            panic!("Parameter value t must be between 0 and 1.");
        }

        if degree == 0 {
            return if self.knots[i] <= t && t < self.knots[i + 1] { 1.0 } else { 0.0 };
        } else {
            let left: f64 = if self.knots[i + degree] != self.knots[i] {
                (t - self.knots[i]) / (self.knots[i + degree] - self.knots[i]) * self.basis(i, degree - 1, t)
            } else {
                0.0
            };
            let right: f64 = if self.knots[i + degree + 1] != self.knots[i + 1] {
                (self.knots[i + degree + 1] - t) / (self.knots[i + degree + 1] - self.knots[i + 1]) * self.basis(i + 1, degree - 1, t)
            } else {
                0.0
            };
            self.memo.insert(hashmap_key, left + right);
            left + right
        }
    }
}