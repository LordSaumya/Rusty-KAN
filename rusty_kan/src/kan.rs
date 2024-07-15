use crate::data_structures::layer;
use crate::data_structures::{vector::Vector, matrix::Matrix, node::Node, layer::Layer, edge::Edge};
use std::fmt::UpperExp;
use std::rc::Rc;
use std::cell::{RefCell, RefMut, Ref};

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
        for layer in self.layers.iter() {
            let layer: Ref<Layer> = layer.borrow();
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
        let mut current_output = input.clone();
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
}