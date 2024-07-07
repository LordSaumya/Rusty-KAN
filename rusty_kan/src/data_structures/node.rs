use std::{cell::RefCell, rc::Rc};
use crate::data_structures::{vector::Vector, edge::Edge};

/// A node is an intersection of edges in the network.
/// It is represented as a list of incoming edges, a list of outgoing edges, and a layer index.
/// The node struct provides methods to compute the value of the node and update the gradients of the incoming edges.
#[derive(Debug, Clone)]
pub struct Node {
    pub incoming: Vec<Rc<RefCell<Edge>>>,
    pub outgoing: Vec<Rc<RefCell<Edge>>>,
    pub layer: usize,
}

impl Node {
    /// Create a new node with a given list of incoming edges, a list of outgoing edges, and a layer index.
    /// 
    /// # Arguments
    /// 
    /// * `incoming_edges` - A list of incoming edges.
    /// 
    /// * `outgoing_edges` - A list of outgoing edges.
    /// 
    /// * `layer` - A layer index.
    /// 
    /// # Returns
    /// 
    /// * A node with the given list of incoming edges, a list of outgoing edges, and a layer index.
    /// 
    /// # Example
    /// 
    /// ```
    /// let incoming_edges: Vec<Rc<RefCell<Edge>>> = vec![edge1, edge2].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();
    /// let outgoing_edges: Vec<Rc<RefCell<Edge>>> = vec![edge3, edge4].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();
    /// let layer = 0;
    /// 
    /// let node = Node::new(incoming_edges, outgoing_edges, layer);
    /// 
    /// ```
    pub fn new(incoming_edges: Vec<Rc<RefCell<Edge>>>, outgoing_edges: Vec<Rc<RefCell<Edge>>>, layer: usize) -> Node {
        Node { incoming: incoming_edges, outgoing: outgoing_edges, layer }
    }

    /// Add an incoming edge to the node.
    /// 
    /// # Arguments
    /// 
    /// * `edge` - An incoming edge.
    /// 
    /// # Example
    /// 
    /// ```
    /// let edge = Rc::new(RefCell::new(Edge::new(start, end, spline, layer)));
    /// ```
    pub fn add_incoming(&mut self, edge: Rc<RefCell<Edge>>) {
        self.incoming.push(edge);
    }

    /// Add an outgoing edge to the node.
    /// 
    /// # Arguments
    /// 
    /// * `edge` - An outgoing edge.
    /// 
    /// # Example
    /// 
    /// ```
    /// let edge = Rc::new(RefCell::new(Edge::new(start, end, spline, layer)));
    /// node.add_outgoing(edge);
    /// ```
    pub fn add_outgoing(&mut self, edge: Rc<RefCell<Edge>>) {
        self.outgoing.push(edge);
    }

    /// Compute the value of the node for a given list of values from the incoming edges.
    /// 
    /// # Arguments
    /// 
    /// * `inputs` - A vector of inputs to the incoming edges.
    /// 
    /// # Returns
    /// 
    /// * The sum of the incoming activations.
    /// 
    /// # Example
    /// 
    /// ```
    /// let inputs = Vector { elements: vec![0.0, 1.0, 2.0] };
    /// let value = node.forward(inputs);
    /// ```
    pub fn forward(&mut self, inputs: &Vector) -> f64 {
        if inputs.len() != self.incoming.len() {
            panic!("The number of inputs must match the number of incoming edges.");
        }
        let mut result: f64 = 0.0;
        for (i, edge) in self.incoming.iter().enumerate() {
            result += edge.borrow_mut().forward(inputs[i]);
        }
        result
    }

    /// Compute the gradients of the incoming edges for their inputs.
    /// 
    /// # Arguments
    /// 
    /// * `t` - A parameter value between 0 and 1.
    /// 
    /// * `upstream_gradient` - A scalar representing the gradient of the loss with respect to the value of the node at the given parameter value t.
    /// 
    /// # Returns
    /// 
    /// * A result indicating whether the gradients of the incoming edges were computed successfully.
    /// 
    /// # Example
    /// 
    /// ```
    /// let t = Vector { elements: vec![0.0, 0.5, 1.0] };
    /// let upstream_gradient = 0.25;
    /// node.backward(t, upstream_gradient);
    /// ```
    pub fn backward(&mut self, t: Vector, upstream_gradient: f64) -> Result<(), &'static str> {
        for (i, edge) in self.incoming.iter().enumerate() {
            edge.borrow_mut().backward(t[i], upstream_gradient).unwrap();
        }
        Ok(())
    }

    /// Update the weights of the incoming edges using the given learning rate.
    /// 
    /// # Arguments
    /// 
    /// * `learning_rate` - A learning rate.
    /// 
    /// # Returns
    /// 
    /// * A result indicating whether the weights of the incoming edges were updated successfully.
    /// 
    /// # Example
    /// 
    /// ```
    /// let learning_rate = 0.01;
    /// node.update_weights(learning_rate);
    /// ```
    pub fn update_weights(&mut self, learning_rate: f64) -> Result<(), &'static str> {
        for edge in self.incoming.iter() {
            edge.borrow_mut().update_weights(learning_rate).unwrap_or_else(|err| {
                panic!("{}", err)
            });
        }
        Ok(())
    }
}