use crate::data_structures::{vector::Vector, matrix::Matrix, node::Node, layer::Layer, edge::Edge};
use std::rc::Rc;
use std::cell::{RefCell, RefMut, Ref};
use std::vec;

/// A KAN is a collection of layers in a network.
/// It is represented as a list of layers.
/// The KAN struct provides methods to compute the value of the KAN and update the activation functions of the edges.
#[derive(Debug, Clone)]
pub struct KAN {
    pub layers: Vec<Rc<RefCell<Layer>>>,
}

impl KAN {
    /// Create a new KAN with a given list of layers.
    /// 
    /// # Arguments
    /// 
    /// * `layers` - A list of layers.
    /// 
    /// # Returns
    /// 
    /// * A KAN with the given list of layers.
    /// 
    /// # Example
    /// 
    /// ```
    /// let layers: Vec<Rc<RefCell<Layer>>> = vec![layer1, layer2].iter().map(|layer| Rc::new(RefCell::new(layer.clone()))).collect();
    /// 
    /// let kan = KAN::new(layers);
    /// ```
    pub fn new(layers: Vec<Rc<RefCell<Layer>>>) -> KAN {
        KAN { layers }
    }

    /// Create a new KAN of standard shape (n inputs, 1 hidden layer with m nodes, 1 output).
    /// The control points of the edges are normally distributed with mean 0 and standard deviation 1.
    /// 
    /// # Arguments
    /// 
    /// * `n` - A scalar representing the number of inputs.
    /// 
    /// * `m` - A scalar representing the number of nodes in the hidden layer.
    /// 
    /// # Returns
    /// 
    /// * A KAN with the given number of inputs and nodes in the hidden layer.
    /// 
    /// # Example
    /// 
    /// ```
    /// let n = 2;
    /// let m = 3;
    /// 
    /// let kan = KAN::standard(n, m);
    /// ```
    pub fn standard(n: usize, m: usize) -> KAN {
        let mut layers: Vec<Rc<RefCell<Layer>>> = Vec::new();
        
        // Input layer nodes (nodes with one incoming edge and m outgoing edges to all nodes in the hidden layer)
        let mut input_nodes: Vec<Rc<RefCell<Node>>> = Vec::with_capacity(n);
        for i in 0..n {
            // Incoming edge
            let incoming_edge: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::standard(i, i, 0)));

            // Outgoing edges
            let mut outgoing_edges: Vec<Rc<RefCell<Edge>>> = Vec::with_capacity(m);
            for j in 0..m {
                let edge: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::standard(i, j, 1)));
                outgoing_edges.push(edge);
            }

            // Node
            let node = Rc::new(RefCell::new(Node::new(vec![incoming_edge], outgoing_edges, 0)));
            input_nodes.push(node);
        }
        

        // Nodes in the hidden layer
        let mut hidden_nodes: Vec<Rc<RefCell<Node>>> = Vec::with_capacity(m);
        for i in 0..m {
            // Incoming edges
            let mut incoming_edges: Vec<Rc<RefCell<Edge>>>= Vec::with_capacity(n);
            for j in 0..n {
                let incoming_edge: Rc<RefCell<Edge>> = input_nodes[j].borrow().outgoing[i].clone();
                incoming_edges.push(incoming_edge);
            }

            // Outgoing edge
            let outgoing_edge = Rc::new(RefCell::new(Edge::standard(i, 0, 2)));

            // Node
            let node = Rc::new(RefCell::new(Node::new(incoming_edges, vec![outgoing_edge], 1)));
            hidden_nodes.push(node);
        }

        let hidden_layer = Rc::new(RefCell::new(Layer::new(hidden_nodes)));

        // Output node
        let mut incoming_edges: Vec<Rc<RefCell<Edge>>> = Vec::with_capacity(m);
        for i in 0..m {
            let hidden_node: Rc<RefCell<Node>> = hidden_layer.borrow().nodes[i].clone();
            let outgoing_edge: Rc<RefCell<Edge>> = hidden_node.borrow().outgoing[0].clone();
            incoming_edges.push(outgoing_edge);
        }
        let output_node = Rc::new(RefCell::new(Node::new(incoming_edges, Vec::new(), 2)));
        
        let output_layer = Rc::new(RefCell::new(Layer::new(vec![output_node])));

        layers.push(hidden_layer);
        layers.push(output_layer);
        
        KAN::new(layers)
    }

    /// Add a layer to the KAN.
    /// 
    /// # Arguments
    /// 
    /// * `layer` - A layer.
    /// 
    /// # Example
    /// 
    /// ```
    /// let layer = Layer::new(nodes);
    /// kan.add_layer(layer);
    /// ```
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(Rc::new(RefCell::new(layer)));
    }

    /// The forward pass computes the value of the KAN given the input values.
    /// 
    /// # Arguments
    /// 
    /// * `input` - A matrix where the entry (i, j) is the input to the j-th incoming edge for the i-th node in the first layer.
    /// 
    /// # Returns
    /// 
    /// * A scalar representing the value of the KAN given the input values.
    /// 
    /// # Example
    /// 
    /// ```
    /// let input = Matrix::new(1, 2, vec![1.0, 2.0]);
    /// 
    /// let output = kan.forward(input);
    /// ```
    pub fn forward(&self, input: Matrix) -> f64 {
        let mut output: Matrix = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            println!("Layer {}", i);
            let layer: Ref<Layer> = layer.borrow();
            println!("Output: {:?}", output);
            output = layer.forward(output);
        }
        output[0][0] // Return the scalar value of the output matrix.
    }

    /// The backward pass computes the gradient of the loss with respect to the input values.
    /// It uses mean squared error as the loss function.
    /// 
    /// # Arguments
    /// 
    /// * `input` - A matrix where the entry (i, j) is the input to the jth incoming edge for the ith node in the first layer.
    /// * `target` - A scalar representing the target value.
    /// 
    /// # Returns
    /// 
    /// * A result indicating whether the backward pass was successful.
    /// 
    /// # Example
    /// 
    /// ```
    /// let input = Matrix::new(vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])]);
    /// let target = 0.5;
    /// 
    /// let result = kan.backward(input, target);
    /// ```
    pub fn backward(&self, input: Matrix, target: f64) -> Result<(), &'static str> {
        // Forward pass and save intermediate values
        let mut layer_outputs: Vec<Matrix> = Vec::new();
        let mut current_output: Matrix = input.clone();
        for layer in self.layers.iter() {
            let layer: Ref<Layer> = layer.borrow();
            current_output = layer.forward(current_output.clone());
            layer_outputs.push(current_output.clone());
        }

        // Calculate initial error gradient (using mean squared error)
        let final_output: f64 = layer_outputs.last().unwrap()[0][0];
        let mut upstream_gradient: Vector = Vector::new(vec![2.0 * (final_output - target)]);

        // Backward pass
        for (i, layer) in self.layers.iter().rev().enumerate() {
            let layer: RefMut<Layer> = layer.borrow_mut();

            // If it is not the first layer, use the output of the previous layer as input
            let layer_input: Matrix = if i > 0 {
                layer_outputs[i - 1].clone()
            } else {
                input.clone()
            };

            layer.backward(layer_input, &upstream_gradient).unwrap();

            // Update the error gradient for the next layer
            upstream_gradient = Vector::new(vec![upstream_gradient.elements.iter().fold(0.0, |acc, &x| acc + x); layer.nodes.len()]);
            
        }

        Ok(())
    }

    /// Update the activation functions of the edges in the KAN.
    /// 
    /// # Arguments
    /// 
    /// * `learning_rate` - A scalar representing the learning rate.
    /// 
    /// # Example
    /// 
    /// ```
    /// let learning_rate = 0.01;
    /// 
    /// kan.update_edges(learning_rate);
    /// ```
    pub fn update_edges(&self, learning_rate: f64) {
        if learning_rate <= 0.0 {
            panic!("Learning rate must be positive.");
        }
        for layer in self.layers.iter() {
            let layer: RefMut<Layer> = layer.borrow_mut();
            layer.update_weights(learning_rate).unwrap();
        }
    }

    /// Calculate the loss of the KAN given the input values and target value.
    /// 
    /// # Arguments
    /// 
    /// * `input` - A vector representing the input values to the first layer.
    /// 
    /// * `target` - A scalar representing the target value.
    /// 
    /// # Returns
    /// 
    /// * A scalar representing the loss of the KAN given the input values and target value, calculated using mean squared error.
    /// 
    /// # Example
    /// 
    /// ```
    /// let input = Vector::new(vec![1.0, 2.0]);
    /// let target = 0.5;
    /// 
    /// let loss = kan.loss_single(input, target);
    /// ```
    pub fn loss_single(&self, input: Vector, target: f64) -> f64 {
        let input_matrix: Matrix = Matrix::new(vec![input]);
        let output: f64 = self.forward(input_matrix);
        (output - target).powi(2)
    }

    /// Calculate the loss of the KAN given a list of input-target pairs.
    /// 
    /// # Arguments
    /// 
    /// * `inputs` - A matrix where the ith row represents the input values to the first layer for the ith input-target pair.
    /// * `targets` - A vector where the ith element represents the target value for the ith input-target pair.
    /// 
    /// # Returns
    /// 
    /// * A scalar representing the loss of the KAN given the input values and target values, calculated using mean squared error.
    /// 
    /// # Example
    /// 
    /// ```
    /// let inputs = Matrix::new(vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])]);
    /// let targets = Vector::new(vec![0.5, 0.75]);
    /// 
    /// let loss = kan.loss(inputs, targets);
    /// ```
    pub fn loss(&self, inputs: Matrix, targets: Vector) -> f64 {
        let mut loss: f64 = 0.0;
        for (i, row) in inputs.rows.iter().enumerate() {
            loss += self.loss_single(row.clone(), targets[i]);
        }
        loss/(inputs.rows.len() as f64)
    }

    /// Train the KAN on one input-target pair.
    /// 
    /// # Arguments
    /// 
    /// * `input` - A vector representing the input values to the first layer.
    /// * `target` - A scalar representing the target value.
    /// * `learning_rate` - A scalar representing the learning rate.
    /// 
    /// # Returns
    /// 
    /// * A result indicating whether the training was successful, with the loss of the KAN given the input values and target value.
    /// 
    /// # Example
    /// 
    /// ```
    /// let input = Vector::new(vec![1.0, 2.0]);
    /// let target = 0.5;
    /// let learning_rate = 0.01;
    /// 
    /// let result = kan.train(input, target, learning_rate).unwrap();
    /// ```
    pub fn train(&self, input: Vector, target: f64, learning_rate: f64) -> Result<f64, &'static str> {
        let input_matrix: Matrix = Matrix::new(vec![input.clone()]);
        self.backward(input_matrix, target)?;
        self.update_edges(learning_rate);
        Ok(self.loss_single(input, target))
    }

    /// Train the KAN on a list of input-target pairs.
    /// 
    /// # Arguments
    /// 
    /// * `inputs` - A matrix where the ith row represents the input values to the first layer for the ith input-target pair.
    /// * `targets` - A vector where the ith element represents the target value for the ith input-target pair.
    /// * `learning_rate` - A scalar representing the learning rate.
    /// 
    /// # Returns
    /// 
    /// * A result indicating whether the training was successful, with the loss of the KAN given the input values and target values.
    /// 
    /// # Example
    /// 
    /// ```
    /// let inputs = Matrix::new(vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])]);
    /// let targets = Vector::new(vec![0.5, 0.75]);
    /// let learning_rate = 0.01;
    /// 
    /// let result = kan.train_batch(inputs, targets, learning_rate).unwrap();
    /// ```
    pub fn train_batch(&self, inputs: Matrix, targets: Vector, learning_rate: f64) -> Result<f64, &'static str> {
        self.backward(inputs.clone(), targets[0])?;
        self.update_edges(learning_rate);
        Ok(self.loss(inputs, targets))
    }
}