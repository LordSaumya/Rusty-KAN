use crate::data_structures::{vector::Vector, spline::BSpline, matrix::Matrix};
use serde::{Serialize, Deserialize};

/// An edge is a connection between two nodes in a graph.
/// It is represented as an index in the origin layer, an index in the destination layer, and a spline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub start: usize,
    pub end: usize,
    pub spline: BSpline,
    pub gradient: Vector, // To store gradients for control points
}

impl Edge {
    /// Create a new edge with a given start index, end index, and spline.
    /// 
    /// # Arguments
    /// 
    /// * `start` - An index in the origin layer.
    /// 
    /// * `end` - An index in the destination layer.
    /// 
    /// * `spline` - A B-spline that represents the edge.
    /// 
    /// # Returns
    /// 
    /// * An edge with the given start index, end index, and spline.
    /// 
    /// # Example
    /// 
    /// ```
    /// let start = 0;
    /// let end = 1;
    /// let spline = BSpline::new(control_points, knots, degree);
    /// let edge = Edge::new(start, end, spline);
    /// ```
    pub fn new(start: usize, end: usize, spline: BSpline) -> Edge {
        let gradient: Vector = Vector { elements: vec![0.0; spline.control_points.len()] };
        Edge { start, end, spline, gradient }
    }

    /// The forward pass computes the value of the spline at the given parameter value t.
    /// 
    /// # Arguments
    /// 
    /// * `t` - A parameter value between 0 and 1.
    /// 
    /// # Returns
    /// 
    /// * A vector representing the value of the spline at the given parameter value t.
    /// 
    /// # Example
    /// 
    /// ```
    /// let edge = Edge::new(start, end, spline);
    /// let t = 0.5;
    /// let value = edge.forward(t);
    /// ```
    pub fn forward(&self, t: f64) -> Vector {
        self.spline.eval(t)
    }

    /// The forward batch pass computes the value of the spline at the given parameter values.
    /// 
    /// # Arguments
    /// 
    /// * `inputs` - A vector of parameter values between 0 and 1.
    /// 
    /// # Returns
    /// 
    /// * A matrix representing the values of the spline at the given parameter values.
    /// 
    /// # Example
    /// 
    /// ```
    /// let edge = Edge::new(start, end, spline);
    /// let inputs = Vector::new(vec![0.0, 0.5, 1.0]);
    /// let values = edge.forward_batch(inputs);
    /// ```
    pub fn forward_batch(&self, inputs: Vector) -> Matrix {
        let result: Vec<Vector> = inputs.map(|t| self.spline.eval(t)).collect();
        Matrix { rows: result }
    }

    /// The backward pass computes the gradient of the spline with respect to the control points.
    /// 
    /// # Arguments
    /// 
    /// * `t` - A parameter value between 0 and 1.
    /// 
    /// * `upstream_gradient` - A vector representing the gradient of the loss with respect to the value of the spline at the given parameter value t.
    /// 
    /// # Example
    /// 
    /// ```
    /// let edge = Edge::new(start, end, spline);
    /// let t = 0.5;
    /// let upstream_gradient = Vector::new(vec![1.0, 0.0, 0.0]);
    /// edge.backward(t, &upstream_gradient);
    /// ```
    pub fn backward(&mut self, t: f64, upstream_gradient: &Vector) -> Result<(), &'static str> {
        let n: usize = self.spline.control_points.len();
        for i in 0..n {
            self.gradient = &self.gradient + &(upstream_gradient * self.spline.basis(i, self.spline.degree, t));
        }
        Ok(())
    }

    /// The backward batch pass computes the gradient of the spline with respect to the control points for a batch of parameter values.
    /// 
    /// # Arguments
    /// 
    /// * `inputs` - A vector of parameter values between 0 and 1.
    /// 
    /// * `upstream_gradients` - A matrix representing the gradients of the loss with respect to the values of the spline at the given parameter values.
    /// 
    /// # Example
    /// 
    /// ```
    /// let edge = Edge::new(start, end, spline);
    /// let inputs = Vector::new(vec![0.0, 0.5, 1.0]);
    /// let upstream_gradients = Matrix::new(vec![Vector::new(vec![1.0, 0.0, 0.0]), Vector::new(vec![0.0, 1.0, 0.0]), Vector::new(vec![0.0, 0.0, 1.0])]);
    /// edge.backward_batch(inputs, &upstream_gradients);
    /// ```
    pub fn update_weights(&mut self, learning_rate: f64) -> Result<(), &'static str> {
        for i in 0..self.spline.control_points.len() {
            self.spline.control_points[i] = &self.spline.control_points[i] - &(&self.gradient * learning_rate);
        }
        // Reset the gradient after the update
        self.gradient = Vector { elements: vec![0.0; self.spline.control_points.len()] };
        Ok(())
    }
}