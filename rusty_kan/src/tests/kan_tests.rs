macro_rules! assert_is_close {
    ($a:expr, $b:expr, $c:expr) => {{
        let a = $a;
        let b = $b;
        let c = $c;
        assert!(
            (a - b).abs() < c,
            "{} and {} are not within {} precision of each other",
            a, b, c
        );
    }};
}

use crate::data_structures::{vector::Vector, spline::BSpline, edge::Edge, node::Node, layer::Layer, matrix::Matrix};
use crate::kan::KAN;
use std::rc::Rc;
use std::cell::{RefCell, RefMut};

#[test]
fn kan_new_pass() {
    // Layer 1
    let incoming_edge_1: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::standard(0, 0, 0)));
    let outgoing_edge_1: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::standard(0, 0, 1)));
    let node_1: Rc<RefCell<Node>> = Rc::new(RefCell::new(Node::new(vec![incoming_edge_1.clone()], vec![outgoing_edge_1.clone()], 0)));
    let layer_1: Rc<RefCell<Layer>> = Rc::new(RefCell::new(Layer::new(vec![node_1.clone()])));

    // Layer 2
    let incoming_edge_2: Rc<RefCell<Edge>> = outgoing_edge_1.clone();
    let outgoing_edge_2: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::standard(0, 0, 2)));
    let node_2: Rc<RefCell<Node>> = Rc::new(RefCell::new(Node::new(vec![incoming_edge_2.clone()], vec![outgoing_edge_2.clone()], 1)));
    let layer_2: Rc<RefCell<Layer>> = Rc::new(RefCell::new(Layer::new(vec![node_2.clone()])));

    // Layers
    let layers: Vec<Rc<RefCell<Layer>>> = vec![layer_1.clone(), layer_2.clone()];
    let kan: KAN = KAN::new(layers.clone());

    // Check that the layers are correct.
    assert_eq!(kan.layers.len(), layers.len());
}

#[test]
fn kan_standard_pass() {
    let kan: KAN = KAN::standard(1, 1);

    assert_eq!(kan.layers.len(), 2);
    assert_eq!(kan.layers[0].borrow().nodes.len(), 1);
    assert_eq!(kan.layers[1].borrow().nodes.len(), 1);
}

#[test]
fn kan_forward_pass() {
    let kan: KAN = KAN::standard(1, 1);

    let input: Matrix = Vector::new(vec![1.0]).to_matrix();
    let output: f64 = kan.forward(input.clone());
}