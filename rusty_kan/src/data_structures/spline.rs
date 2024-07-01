use crate::data_structures::vector::Vector;

/// A B-spline is a piecewise polynomial function that is used as a parameterised version of a univariate learnable activation function in a KAN.
/// It is represented as a list of control points, a list of knots, and a degree.
/// 
/// The B_Spline struct implements methods to evaluate the function at a given point and calculate the basis function.
/// The basis function is a recursive function that calculates the value of the B-spline at a given point.
/// The eval method calculates the value of the B-spline at a given point by summing the control points multiplied by the basis function. 

#[derive(Debug, Clone)]
pub struct BSpline {
    pub control_points: Vec<Vector>, // Coefficients to be trained
    pub knots: Vec<f64>,
    pub degree: usize,
}

impl BSpline {
    /// Create a new B-spline with a given list of control points and degree.
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
    pub fn new(control_points: Vec<Vector>, degree: usize) -> BSpline {
        let knots: Vec<f64> = (0..control_points.len() + degree + 1).map(|x| (x as f64)/(control_points.len() as f64 + degree as f64)).collect(); // Uniform knots
        BSpline { control_points, knots, degree }
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
    pub fn eval(&self, t: f64) -> Vector {
        if t < 0.0 || t > 1.0 {
            panic!("Parameter value t must be between 0 and 1.");
        }

        let n: usize = self.control_points.len();
        let mut result: Vector = Vector { elements: vec![0.0; self.control_points[0].elements.len()] };
        for i in 0..n {
            result = result + &self.control_points[i] * self.basis(i, self.degree, t);
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
    pub fn basis(&self, i: usize, degree: usize, t: f64) -> f64 {
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
            left + right
        }
    }
}