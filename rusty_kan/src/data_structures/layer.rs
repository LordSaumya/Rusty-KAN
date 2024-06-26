use crate::data_structures::{node::Node, edge::Edge, vector::Vector, matrix::Matrix};
use std::rc::Rc;
use std::cell::RefCell;

/// A layer is a collection of nodes and its associated edges in a KAN.
/// It is represented as a list of nodes.
/// The layer struct provides methods to compute the value of the layer and update the gradients of the nodes.
#[derive(Debug, Clone)]
pub struct Layer {
    pub nodes: Vec<Rc<RefCell<Node>>>,
}

impl Layer {
    /// Create a new layer with a given list of nodes.
    /// 
    /// # Arguments
    /// 
    /// * `nodes` - A list of nodes.
    /// 
    /// # Returns
    /// 
    /// * A layer with the given list of nodes.
    /// 
    /// # Example
    /// 
    /// ```
    /// let nodes: Vec<Rc<RefCell<Node>>> = vec![node1, node2].iter().map(|node| Rc::new(RefCell::new(node.clone()))).collect();
    /// 
    /// let layer = Layer::new(nodes);
    /// ```
    pub fn new(nodes: Vec<Rc<RefCell<Node>>>) -> Layer {
        Layer { nodes }
    }

    /// Add a node to the layer.
    /// 
    /// # Arguments
    /// 
    /// * `node` - A node.
    /// 
    /// # Example
    /// 
    /// ```
    /// let node = Node::new(incoming_edges, outgoing_edges, layer);
    /// layer.add_node(node);
    /// ```
    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(Rc::new(RefCell::new(node)));
    }

    /// The forward pass computes the value of the layer given the input values.
    /// 
    /// # Arguments
    /// 
    /// * `input` - A vector representing the input values.
    /// 
    /// # Returns
    /// 
    /// * A matrix representing the value of the layer given the input values.
    /// 
    /// # Example
    /// 
    /// ```
    /// let layer = Layer::new(nodes);
    /// let input = Vector { elements: vec![0.0, 1.0, 2.0] };
    /// let value = layer.forward(input);
    /// ```
    pub fn forward(&self, input: Vector) -> Matrix {
        let mut result: Matrix = Matrix::zeros(self.nodes.len(), input.elements.len());
        for (i, node) in self.nodes.iter().enumerate() {
            let node_value: Vector = node.borrow().forward(&input);
            result.set_row(i, node_value);
        }
        result
    }

    /// The backward pass computes the gradients of the edges in the layer given the upstream gradients and the input values.
    /// 
    /// # Arguments
    /// 
    /// * `input` - A vector representing the input values.
    /// 
    /// * `upstream_gradient` - A matrix representing the gradients from the next layer.
    /// 
    /// # Returns
    /// 
    /// * A result indicating whether the gradients were computed successfully.
    /// 
    /// # Example
    /// 
    /// ```
    /// let layer = Layer::new(nodes);
    /// let upstream_gradient = Matrix::zeros(3, 3);
    /// layer.backward(upstream_gradient);
    /// ```
    pub fn backward(&self, input: Vector, upstream_gradient: Matrix) -> Result<(), &str> {
        for (i, node) in self.nodes.iter().enumerate() {
            node.borrow_mut().backward(input[i], &upstream_gradient[i]).unwrap_or_else(|err| {
                println!("{}", err);
            });
        }
        Ok(())
    }

    /// Updates the weights of the incoming edges in the layer.
    /// 
    /// # Arguments
    /// 
    /// * `learning_rate` - A learning rate.
    /// 
    /// # Returns
    /// 
    /// * A result indicating whether the weights were updated successfully.
    /// 
    /// # Example
    /// 
    /// ```
    /// let layer = Layer::new(nodes);
    /// let learning_rate = 0.01;
    /// layer.update_weights(learning_rate);
    /// ```
    pub fn update_weights(&self, learning_rate: f64) -> Result<(), &str> {
        for node in self.nodes.iter() {
            node.borrow_mut().update_weights(learning_rate).unwrap_or_else(|err| {
                println!("{}", err);
            });
        }
        Ok(())
    }
}





