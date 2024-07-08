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

use crate::data_structures::{node::Node, vector::Vector, matrix::Matrix, edge::Edge, spline::BSpline, layer::Layer};
use std::rc::Rc;
use std::cell::{RefCell, RefMut};

#[test]
fn layer_new_pass() {
    // Node 1
    let incoming_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0)));
    let incoming_edge_12: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 0, BSpline::new(Vector::new(vec![1.5, 2.5, 3.5]), 2), 0)));

    let outgoing_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let node_1: Node = Node::new(vec![incoming_edge_11, incoming_edge_12], vec![outgoing_edge_11], 0);

    // Node 2
    let incoming_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));
    let incoming_edge_22: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 1, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let outgoing_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));

    let node_2: Node = Node::new(vec![incoming_edge_21, incoming_edge_22], vec![outgoing_edge_21], 0);

    // Layer
    let nodes: Vec<Rc<RefCell<Node>>> = vec![node_1, node_2].iter().map(|node| Rc::new(RefCell::new(node.clone()))).collect();
    let layer: Layer = Layer::new(nodes);

    assert_eq!(layer.nodes.len(), 2);
    assert_eq!(layer.nodes[0].borrow().incoming.len(), 2);
    assert_eq!(layer.nodes[0].borrow().outgoing.len(), 1);
    assert_eq!(layer.nodes[1].borrow().incoming.len(), 2);
    assert_eq!(layer.nodes[1].borrow().outgoing.len(), 1);
}

#[test]
fn layer_add_node_pass() {
    // Node 1
    let incoming_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0)));
    let incoming_edge_12: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 0, BSpline::new(Vector::new(vec![1.5, 2.5, 3.5]), 2), 0)));
    
    let outgoing_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let node_1: Node = Node::new(vec![incoming_edge_11, incoming_edge_12], vec![outgoing_edge_11], 0);

    // Node 2
    let incoming_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));
    let incoming_edge_22: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 1, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let outgoing_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));

    let node_2: Node = Node::new(vec![incoming_edge_21, incoming_edge_22], vec![outgoing_edge_21], 0);

    // Layer
    let nodes: Vec<Rc<RefCell<Node>>> = vec![node_1].iter().map(|node| Rc::new(RefCell::new(node.clone()))).collect();
    let mut layer: Layer = Layer::new(nodes);

    assert_eq!(layer.nodes.len(), 1);
    layer.add_node(node_2);
    assert_eq!(layer.nodes.len(), 2);
}

#[test]
fn layer_forward_pass() {
    // Node 1
    let incoming_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0)));
    let incoming_edge_12: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 0, BSpline::new(Vector::new(vec![1.5, 2.5, 3.5]), 2), 0)));
    
    let outgoing_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let node_1: Node = Node::new(vec![incoming_edge_11, incoming_edge_12], vec![outgoing_edge_11], 0);

    // Node 2
    let incoming_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));
    let incoming_edge_22: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 1, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let outgoing_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));

    let node_2: Node = Node::new(vec![incoming_edge_21, incoming_edge_22], vec![outgoing_edge_21], 0);

    // Layer
    let nodes: Vec<Rc<RefCell<Node>>> = vec![node_1, node_2].iter().map(|node| Rc::new(RefCell::new(node.clone()))).collect();
    let layer: Layer = Layer::new(nodes);

    let input: Matrix = Matrix::new(vec![Vector::from(vec![0.1, 0.2]), Vector::from(vec![0.3, 0.4])]);
    
    let value: Matrix = layer.forward(input.clone());

    assert_eq!(value.rows.len(), 2);
    assert_eq!(value.rows[0].elements.len(), 1);
    assert_eq!(value.rows[1].elements.len(), 1);
    assert_is_close!(value[0][0], layer.nodes[0].borrow_mut().forward(&input[0]), 1e-6);
    assert_is_close!(value[1][0], layer.nodes[1].borrow_mut().forward(&input[1]), 1e-6);
}

#[test]
#[should_panic]
fn layer_forward_less_rows_fail() {
    // Node 1
    let incoming_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0)));
    let incoming_edge_12: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 0, BSpline::new(Vector::new(vec![1.5, 2.5, 3.5]), 2), 0)));
    
    let outgoing_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let node_1: Node = Node::new(vec![incoming_edge_11, incoming_edge_12], vec![outgoing_edge_11], 0);

    // Node 2
    let incoming_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));
    let incoming_edge_22: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 1, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let outgoing_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));

    let node_2: Node = Node::new(vec![incoming_edge_21, incoming_edge_22], vec![outgoing_edge_21], 0);

    // Layer
    let nodes: Vec<Rc<RefCell<Node>>> = vec![node_1, node_2].iter().map(|node| Rc::new(RefCell::new(node.clone()))).collect();
    let layer: Layer = Layer::new(nodes);

    // Input dimensions should be 2 x 2
    let input: Matrix = Matrix::new(vec![Vector::from(vec![0.1, 0.2])]);

    layer.forward(input);
}

#[test]
#[should_panic]
fn layer_forward_more_rows_fail() {
    // Node 1
    let incoming_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0)));
    let incoming_edge_12: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 0, BSpline::new(Vector::new(vec![1.5, 2.5, 3.5]), 2), 0)));
    
    let outgoing_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let node_1: Node = Node::new(vec![incoming_edge_11, incoming_edge_12], vec![outgoing_edge_11], 0);

    // Node 2
    let incoming_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));
    let incoming_edge_22: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 1, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let outgoing_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));

    let node_2: Node = Node::new(vec![incoming_edge_21, incoming_edge_22], vec![outgoing_edge_21], 0);

    // Layer
    let nodes: Vec<Rc<RefCell<Node>>> = vec![node_1, node_2].iter().map(|node| Rc::new(RefCell::new(node.clone()))).collect();
    let layer: Layer = Layer::new(nodes);

    // Input dimensions should be 2 x 2
    let input: Matrix = Matrix::new(vec![Vector::from(vec![0.1, 0.2]), Vector::from(vec![0.3, 0.4]), Vector::from(vec![0.5, 0.6])]);

    layer.forward(input);
}

#[test]
#[should_panic]
fn layer_forward_less_cols_fail() {
    // Node 1
    let incoming_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0)));
    let incoming_edge_12: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 0, BSpline::new(Vector::new(vec![1.5, 2.5, 3.5]), 2), 0)));
    
    let outgoing_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let node_1: Node = Node::new(vec![incoming_edge_11, incoming_edge_12], vec![outgoing_edge_11], 0);

    // Node 2
    let incoming_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));
    let incoming_edge_22: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 1, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let outgoing_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));

    let node_2: Node = Node::new(vec![incoming_edge_21, incoming_edge_22], vec![outgoing_edge_21], 0);

    // Layer
    let nodes: Vec<Rc<RefCell<Node>>> = vec![node_1, node_2].iter().map(|node| Rc::new(RefCell::new(node.clone()))).collect();
    let layer: Layer = Layer::new(nodes);

    // Input dimensions should be 2 x 2
    let input: Matrix = Matrix::new(vec![Vector::from(vec![0.1]), Vector::from(vec![0.3])]);

    layer.forward(input);
}

#[test]
#[should_panic]
fn layer_forward_more_cols_fail() {
    // Node 1
    let incoming_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0)));
    let incoming_edge_12: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 0, BSpline::new(Vector::new(vec![1.5, 2.5, 3.5]), 2), 0)));
    
    let outgoing_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let node_1: Node = Node::new(vec![incoming_edge_11, incoming_edge_12], vec![outgoing_edge_11], 0);

    // Node 2
    let incoming_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));
    let incoming_edge_22: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 1, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let outgoing_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));

    let node_2: Node = Node::new(vec![incoming_edge_21, incoming_edge_22], vec![outgoing_edge_21], 0);

    // Layer
    let nodes: Vec<Rc<RefCell<Node>>> = vec![node_1, node_2].iter().map(|node| Rc::new(RefCell::new(node.clone()))).collect();
    let layer: Layer = Layer::new(nodes);

    // Input dimensions should be 2 x 2
    let input: Matrix = Matrix::new(vec![Vector::from(vec![0.1, 0.2, 0.3]), Vector::from(vec![0.3, 0.4, 0.5])]);

    layer.forward(input);
}

#[test]
fn layer_backward_pass() {
    // Node 1
    let incoming_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0)));
    let incoming_edge_12: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 0, BSpline::new(Vector::new(vec![1.5, 2.5, 3.5]), 2), 0)));
    
    let outgoing_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let node_1: Node = Node::new(vec![incoming_edge_11, incoming_edge_12], vec![outgoing_edge_11], 0);

    // Node 2
    let incoming_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));
    let incoming_edge_22: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 1, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let outgoing_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));

    let node_2: Node = Node::new(vec![incoming_edge_21, incoming_edge_22], vec![outgoing_edge_21], 0);

    // Layer
    let nodes: Vec<Rc<RefCell<Node>>> = vec![node_1.clone(), node_2.clone()].iter().map(|node| Rc::new(RefCell::new(node.clone()))).collect();
    let layer: Layer = Layer::new(nodes);

    let inputs: Matrix = Matrix::new(vec![Vector::from(vec![0.1, 0.2]), Vector::from(vec![0.3, 0.4])]);
    let upstream_gradient: Vector = Vector::from(vec![0.4, 0.8]);

    layer.backward(inputs.clone(), upstream_gradient.clone()).unwrap();

    // Node 1:
    // Incoming edge 1
    let incoming_edge_1: RefMut<Edge> = node_1.incoming[0].borrow_mut();
    assert_is_close!(incoming_edge_1.gradient[0], incoming_edge_1.clone().spline.basis(0, incoming_edge_1.spline.degree, inputs[0][0]) * upstream_gradient[0], 1e-3);
    assert_is_close!(incoming_edge_1.gradient[1], incoming_edge_1.clone().spline.basis(1, incoming_edge_1.spline.degree, inputs[0][0]) * upstream_gradient[0], 1e-3);
    assert_is_close!(incoming_edge_1.gradient[2], incoming_edge_1.clone().spline.basis(2, incoming_edge_1.spline.degree, inputs[0][0]) * upstream_gradient[0], 1e-3);

    // Incoming edge 2
    let incoming_edge_2: RefMut<Edge> = node_1.incoming[1].borrow_mut();
    assert_is_close!(incoming_edge_2.gradient[0], incoming_edge_2.clone().spline.basis(0, incoming_edge_2.spline.degree, inputs[0][1]) * upstream_gradient[0], 1e-3);
    assert_is_close!(incoming_edge_2.gradient[1], incoming_edge_2.clone().spline.basis(1, incoming_edge_2.spline.degree, inputs[0][1]) * upstream_gradient[0], 1e-3);
    assert_is_close!(incoming_edge_2.gradient[2], incoming_edge_2.clone().spline.basis(2, incoming_edge_2.spline.degree, inputs[0][1]) * upstream_gradient[0], 1e-3);

    // Node 2:
    // Incoming edge 1
    let incoming_edge_1: RefMut<Edge> = node_2.incoming[0].borrow_mut();
    assert_is_close!(incoming_edge_1.gradient[0], incoming_edge_1.clone().spline.basis(0, incoming_edge_1.spline.degree, inputs[1][0]) * upstream_gradient[1], 1e-3);
    assert_is_close!(incoming_edge_1.gradient[1], incoming_edge_1.clone().spline.basis(1, incoming_edge_1.spline.degree, inputs[1][0]) * upstream_gradient[1], 1e-3);
    assert_is_close!(incoming_edge_1.gradient[2], incoming_edge_1.clone().spline.basis(2, incoming_edge_1.spline.degree, inputs[1][0]) * upstream_gradient[1], 1e-3);

    // Incoming edge 2
    let incoming_edge_2: RefMut<Edge> = node_2.incoming[1].borrow_mut();
    assert_is_close!(incoming_edge_2.gradient[0], incoming_edge_2.clone().spline.basis(0, incoming_edge_2.spline.degree, inputs[1][1]) * upstream_gradient[1], 1e-3);
    assert_is_close!(incoming_edge_2.gradient[1], incoming_edge_2.clone().spline.basis(1, incoming_edge_2.spline.degree, inputs[1][1]) * upstream_gradient[1], 1e-3);
    assert_is_close!(incoming_edge_2.gradient[2], incoming_edge_2.clone().spline.basis(2, incoming_edge_2.spline.degree, inputs[1][1]) * upstream_gradient[1], 1e-3);
}

#[test]
#[should_panic]
fn layer_backward_wrong_input_dims_fail() {
    // Node 1
    let incoming_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0)));
    let incoming_edge_12: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 0, BSpline::new(Vector::new(vec![1.5, 2.5, 3.5]), 2), 0)));
    
    let outgoing_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let node_1: Node = Node::new(vec![incoming_edge_11, incoming_edge_12], vec![outgoing_edge_11], 0);

    // Node 2
    let incoming_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));
    let incoming_edge_22: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 1, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));
    
    let outgoing_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));

    let node_2: Node = Node::new(vec![incoming_edge_21, incoming_edge_22], vec![outgoing_edge_21], 0);

    // Layer
    let nodes: Vec<Rc<RefCell<Node>>> = vec![node_1.clone(), node_2.clone()].iter().map(|node| Rc::new(RefCell::new(node.clone()))).collect();
    let layer: Layer = Layer::new(nodes);

    let inputs: Matrix = Matrix::new(vec![Vector::from(vec![0.1, 0.2])]);
    let upstream_gradient: Vector = Vector::from(vec![0.4, 0.8]);

    layer.backward(inputs.clone(), upstream_gradient.clone()).unwrap();
}

#[test]
#[should_panic]
fn layer_backward_wrong_gradient_dims_fail() {
    // Node 1
    let incoming_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0)));
    let incoming_edge_12: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 0, BSpline::new(Vector::new(vec![1.5, 2.5, 3.5]), 2), 0)));
    
    let outgoing_edge_11: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 0, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));

    let node_1: Node = Node::new(vec![incoming_edge_11, incoming_edge_12], vec![outgoing_edge_11], 0);

    // Node 2
    let incoming_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));
    let incoming_edge_22: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(1, 1, BSpline::new(Vector::new(vec![0.5, 1.5, 2.5]), 2), 0)));
    
    let outgoing_edge_21: Rc<RefCell<Edge>> = Rc::new(RefCell::new(Edge::new(0, 1, BSpline::new(Vector::new(vec![0.0, 1.0, 2.0]), 2), 0)));

    let node_2: Node = Node::new(vec![incoming_edge_21, incoming_edge_22], vec![outgoing_edge_21], 0);

    // Layer
    let nodes: Vec<Rc<RefCell<Node>>> = vec![node_1.clone(), node_2.clone()].iter().map(|node| Rc::new(RefCell::new(node.clone()))).collect();
    let layer: Layer = Layer::new(nodes);

    let inputs: Matrix = Matrix::new(vec![Vector::from(vec![0.1, 0.2]), Vector::from(vec![0.3, 0.4])]);
    let upstream_gradient: Vector = Vector::from(vec![0.4]);

    layer.backward(inputs.clone(), upstream_gradient.clone()).unwrap();
}