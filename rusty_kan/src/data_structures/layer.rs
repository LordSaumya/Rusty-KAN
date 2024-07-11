use crate::data_structures::{node::Node, vector::Vector, matrix::Matrix};
use std::rc::Rc;
use std::cell::{RefCell, RefMut};

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
    /// * `input` - A matrix where the entry (i, j) is the input to the j-th incoming edge for the i-th node.
    /// 
    /// # Returns
    /// 
    /// * A matrix representing the value of the layer given the input values, where the entry (i, j) is the value of the j-th outgoing edge for the i-th node.
    /// 
    /// # Example
    ///  
    /// ```
    /// let layer = Layer::new(nodes);
    /// let input = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// let value = layer.forward(input);
    /// ```
    pub fn forward(&self, input: Matrix) -> Matrix {
        if input.shape().0 != self.nodes.len() {
            panic!("The number of rows in the input matrix must be equal to the number of nodes in the layer.");
        }
        let mut result: Matrix = Matrix::new(vec![]);
        for i in 0..self.nodes.len() {
            let mut node: RefMut<Node> = self.nodes[i].borrow_mut();
            let sum: f64 = node.forward(&input[i]);
            let result_vector: Vector = Vector::new(vec![sum; node.outgoing.len()]);
            result.push(result_vector);
        }
        result
    }
    
    /// The backward pass computes the gradients of the edges in the incoming layer given the upstream gradients and the input values.
    /// 
    /// # Arguments
    /// 
    /// * `input` - A matrix of input values to the incoming edges of the layer, where the (i, j)-th entry is the input to the j-th incoming edge of the i-th node in the layer.
    /// 
    /// * `upstream_gradient` - A vector representing the gradients from the next layer, where the i-th entry is the gradient of the loss with respect to the outgoing edge of the i-th node.
    /// 
    /// # Returns
    /// 
    /// * A result indicating whether the gradients were computed successfully.
    /// 
    /// # Example
    /// 
    /// ```
    /// let layer = Layer::new(nodes);
    /// let input = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// let upstream_gradient = Vector::new(vec![0.5, 0.25]);
    /// layer.backward(input, upstream_gradient);
    /// ```
    pub fn backward(&self, input: Matrix, upstream_gradient: &Vector) -> Result<(), &str> {
        if input.shape().0 != self.nodes.len() {
            panic!("The number of rows in the input matrix must be equal to the number of nodes in the layer.");
        }
        if upstream_gradient.len() != self.nodes.len() {
            panic!("The number of elements in the upstream gradient vector must be equal to the number of nodes in the layer.");
        }

        for (i, node) in self.nodes.iter().enumerate() {
            let mut node: RefMut<Node> = node.borrow_mut();
            node.backward(input[i].clone(), upstream_gradient[i]).unwrap_or_else(|err| {
                panic!("{}", err)
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
                panic!("{}", err)
            });
        }
        Ok(())
    }
}
