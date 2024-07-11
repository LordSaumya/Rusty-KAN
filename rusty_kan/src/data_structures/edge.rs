use crate::data_structures::{vector::Vector, spline::BSpline};

/// An edge is a connection between two nodes in a graph.
/// It is represented as an index in the origin layer, an index in the destination layer, a layer index corresponding to the origin layer, and a spline.
#[derive(Debug, Clone)]
pub struct Edge {
    pub start: usize,
    pub end: usize,
    pub spline: BSpline,
    pub layer: usize,
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
    /// * `layer` - A layer index corresponding to the origin layer.
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
    /// let layer = 0;
    /// let spline = BSpline::new(control_points, knots, degree);
    /// let edge = Edge::new(start, end, spline, layer);
    /// ```
    pub fn new(start: usize, end: usize, spline: BSpline, layer: usize) -> Edge {
        let gradient: Vector = Vector { elements: vec![0.0; spline.control_points.len()] };
        Edge { start, end, spline, gradient, layer }
    }

    /// The forward pass computes the value of the spline at the given parameter value t and adds the value of the basis function (sigmoid linear unit).
    /// 
    /// # Arguments
    /// 
    /// * `t` - A parameter value between 0 and 1.
    /// 
    /// # Returns
    /// 
    /// * A scalar representing the value of the spline at the given parameter value t.
    /// 
    /// # Example
    /// 
    /// ```
    /// let edge = Edge::new(start, end, spline, layer);
    /// let t = 0.5;
    /// let value = edge.forward(t);
    /// ```
    pub fn forward(&mut self, t: f64) -> f64 {
        self.spline.eval(t) + silu(t)
    }

    /// The forward batch pass computes the value of the spline at the given parameter values.
    /// 
    /// # Arguments
    /// 
    /// * `inputs` - A vector of parameter values between 0 and 1.
    /// 
    /// # Returns
    /// 
    /// * A vector representing the values of the spline at the given parameter values.
    /// 
    /// # Example
    /// 
    /// ```
    /// let edge = Edge::new(start, end, spline, layer);
    /// let inputs = Vector::new(vec![0.0, 0.5, 1.0]);
    /// let values = edge.forward_batch(inputs);
    /// ```
    pub fn forward_batch(&mut self, inputs: Vector) -> Vector {
        let mut result: Vec<f64> = inputs.map(|t| self.spline.eval(t) + silu(t)).collect();
        result.reverse();
        Vector::new(result)
    }

    /// The backward pass computes the gradient of the spline with respect to the control points.
    /// 
    /// # Arguments
    /// 
    /// * `t` - A parameter value between 0 and 1.
    /// 
    /// * `upstream_gradient` - A scalar representing the gradient of the loss with respect to the value of the spline at the given parameter value t.
    /// 
    /// # Example
    /// 
    /// ```
    /// let edge = Edge::new(start, end, spline, layer);
    /// let t = 0.5;
    /// let upstream_gradient = 0.25;
    /// edge.backward(t, upstream_gradient);
    /// ```
    pub fn backward(&mut self, t: f64, upstream_gradient: f64) -> Result<(), &'static str> {
        let n: usize = self.spline.control_points.len();
        for i in 0..n {
            self.gradient[i] = self.spline.basis(i, self.spline.degree, t) * upstream_gradient;
        }
        
        Ok(())
    }

    /// Uses the stored gradient of the spline with respect to the control points to update the control points.
    /// 
    /// # Arguments
    /// 
    /// * `learning_rate` - A scalar representing the learning rate.
    /// 
    /// # Returns
    /// 
    /// * A result indicating whether the update was successful.
    /// 
    /// # Example
    /// 
    /// ```
    /// let edge = Edge::new(start, end, spline, layer);
    /// let learning_rate = 0.01;
    /// edge.update_weights(learning_rate);
    /// ```
    pub fn update_weights(&mut self, learning_rate: f64) -> Result<(), &'static str> {
        if learning_rate <= 0.0 {
            panic!("The learning rate must be greater than 0.");
        }
        // control points = control points - learning_rate * gradient
        self.spline.control_points = &self.spline.control_points - &(&self.gradient * learning_rate);
        // Reset gradient
        self.gradient = Vector { elements: vec![0.0; self.spline.control_points.len()] };
        Ok(())
    }
}

/// The Sigmoid Linear Unit (SiLU) activation function.
/// 
/// # Arguments
/// 
/// * `x` - A scalar.
/// 
/// # Returns
/// 
/// * The SiLU of the scalar x.
/// 
/// # Example
/// 
/// ```
/// let x = 0.5;
/// let silu = silu(x);
/// ```
fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

impl std::fmt::Display for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Edge(start: {}, end: {}, layer: {}, spline: {})", self.start, self.end, self.layer, self.spline)
    }
}