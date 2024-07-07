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

use crate::data_structures::{node::Node, vector::Vector, edge::Edge, spline::BSpline};
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn node_new_pass() {
    let incoming_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0);
    let incoming_edge_2: Edge = Edge::new(1, 0, BSpline::new(Vector::new(vec![4.0, 5.0, 6.0]), 2), 0);

    let outgoing_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![7.0, 8.0, 9.0]), 2), 0);

    let incoming_edges: Vec<Rc<RefCell<Edge>>> = vec![incoming_edge_1, incoming_edge_2].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();
    let outgoing_edges: Vec<Rc<RefCell<Edge>>> = vec![outgoing_edge_1].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();

    let node: Node = Node::new(incoming_edges, outgoing_edges, 0);

    assert_eq!(node.incoming.len(), 2);
    assert_eq!(node.outgoing.len(), 1);
    assert_eq!(node.layer, 0);
}

#[test]
fn node_add_incoming_pass() {
    let incoming_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0);
    let incoming_edge_2: Edge = Edge::new(1, 0, BSpline::new(Vector::new(vec![4.0, 5.0, 6.0]), 2), 0);

    let outgoing_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![7.0, 8.0, 9.0]), 2), 0);

    let incoming_edges: Vec<Rc<RefCell<Edge>>> = vec![incoming_edge_1, incoming_edge_2].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();
    let outgoing_edges: Vec<Rc<RefCell<Edge>>> = vec![outgoing_edge_1].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();

    let mut node: Node = Node::new(incoming_edges, outgoing_edges, 0);

    let incoming_edge_3: Edge = Edge::new(2, 0, BSpline::new(Vector::new(vec![10.0, 11.0, 12.0]), 2), 0);
    let incoming_edge_3: Rc<RefCell<Edge>> = Rc::new(RefCell::new(incoming_edge_3));

    node.add_incoming(incoming_edge_3);

    assert_eq!(node.incoming.len(), 3);
}

#[test]
fn node_add_outgoing_pass() {
    let incoming_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0);
    let incoming_edge_2: Edge = Edge::new(1, 0, BSpline::new(Vector::new(vec![4.0, 5.0, 6.0]), 2), 0);

    let outgoing_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![7.0, 8.0, 9.0]), 2), 0);

    let incoming_edges: Vec<Rc<RefCell<Edge>>> = vec![incoming_edge_1, incoming_edge_2].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();
    let outgoing_edges: Vec<Rc<RefCell<Edge>>> = vec![outgoing_edge_1].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();

    let mut node: Node = Node::new(incoming_edges, outgoing_edges, 0);

    let outgoing_edge_2: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![13.0, 14.0, 15.0]), 2), 0);
    let outgoing_edge_2: Rc<RefCell<Edge>> = Rc::new(RefCell::new(outgoing_edge_2));

    node.add_outgoing(outgoing_edge_2);

    assert_eq!(node.outgoing.len(), 2);
}

#[test]
fn node_forward_pass() {
    let mut incoming_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0);
    let mut incoming_edge_2: Edge = Edge::new(1, 0, BSpline::new(Vector::new(vec![4.0, 5.0, 6.0]), 2), 0);

    let outgoing_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![7.0, 8.0, 9.0]), 2), 0);

    let incoming_edges: Vec<Rc<RefCell<Edge>>> = vec![incoming_edge_1.clone(), incoming_edge_2.clone()].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();
    let outgoing_edges: Vec<Rc<RefCell<Edge>>> = vec![outgoing_edge_1].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();

    let mut node: Node = Node::new(incoming_edges, outgoing_edges, 0);

    let input: Vector = Vector::new(vec![0.0, 1.0]);
    let value: f64 = node.forward(&input);

    // Expected value
    let expected_value: f64 = incoming_edge_1.forward(0.0) + incoming_edge_2.forward(1.0);

    assert_is_close!(value, expected_value, 1e-3);
}

#[test]
#[should_panic]
fn node_forward_fail() {
    let incoming_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0);
    let incoming_edge_2: Edge = Edge::new(1, 0, BSpline::new(Vector::new(vec![4.0, 5.0, 6.0]), 2), 0);

    let outgoing_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![7.0, 8.0, 9.0]), 2), 0);

    let incoming_edges: Vec<Rc<RefCell<Edge>>> = vec![incoming_edge_1.clone(), incoming_edge_2.clone()].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();
    let outgoing_edges: Vec<Rc<RefCell<Edge>>> = vec![outgoing_edge_1].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();

    let mut node: Node = Node::new(incoming_edges, outgoing_edges, 0);

    let input: Vector = Vector::new(vec![0.0, 1.0, 2.0]);
    let _ = node.forward(&input);
}

#[test]
fn node_backward_pass() {
    let mut incoming_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0);
    let mut incoming_edge_2: Edge = Edge::new(1, 0, BSpline::new(Vector::new(vec![4.0, 5.0, 6.0]), 2), 0);

    let outgoing_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![7.0, 8.0, 9.0]), 2), 0);

    let incoming_edges: Vec<Rc<RefCell<Edge>>> = vec![incoming_edge_1.clone(), incoming_edge_2.clone()].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();
    let outgoing_edges: Vec<Rc<RefCell<Edge>>> = vec![outgoing_edge_1].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();

    let mut node: Node = Node::new(incoming_edges, outgoing_edges, 0);

    let upstream_gradient: f64 = 0.5;
    let inputs: Vector = Vector::new(vec![0.5, 1.0]);

    node.backward(inputs.clone(), upstream_gradient).unwrap();
    assert_is_close!(node.incoming[0].borrow().gradient[0], incoming_edge_1.spline.basis(0, incoming_edge_1.spline.degree, inputs[0]) * upstream_gradient, 1e-3);
    assert_is_close!(node.incoming[0].borrow().gradient[1], incoming_edge_1.spline.basis(1, incoming_edge_1.spline.degree, inputs[0]) * upstream_gradient, 1e-3);
    assert_is_close!(node.incoming[0].borrow().gradient[2], incoming_edge_1.spline.basis(2, incoming_edge_1.spline.degree, inputs[0]) * upstream_gradient, 1e-3);

    assert_is_close!(node.incoming[1].borrow().gradient[0], incoming_edge_2.spline.basis(0, incoming_edge_2.spline.degree, inputs[1]) * upstream_gradient, 1e-3);
    assert_is_close!(node.incoming[1].borrow().gradient[1], incoming_edge_2.spline.basis(1, incoming_edge_2.spline.degree, inputs[1]) * upstream_gradient, 1e-3);
    assert_is_close!(node.incoming[1].borrow().gradient[2], incoming_edge_2.spline.basis(2, incoming_edge_2.spline.degree, inputs[1]) * upstream_gradient, 1e-3);
}

#[test]
#[should_panic]
fn node_backward_fail() {
    let incoming_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0);
    let incoming_edge_2: Edge = Edge::new(1, 0, BSpline::new(Vector::new(vec![4.0, 5.0, 6.0]), 2), 0);

    let outgoing_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![7.0, 8.0, 9.0]), 2), 0);

    let incoming_edges: Vec<Rc<RefCell<Edge>>> = vec![incoming_edge_1.clone(), incoming_edge_2.clone()].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();
    let outgoing_edges: Vec<Rc<RefCell<Edge>>> = vec![outgoing_edge_1].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();

    let mut node: Node = Node::new(incoming_edges, outgoing_edges, 0);

    let upstream_gradient: f64 = 0.5;
    let inputs: Vector = Vector::new(vec![0.0]);

    node.backward(inputs, upstream_gradient).unwrap();
}

#[test]
fn node_weight_update_pass() {
    let mut incoming_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0);
    let mut incoming_edge_2: Edge = Edge::new(1, 0, BSpline::new(Vector::new(vec![4.0, 5.0, 6.0]), 2), 0);
    
    let outgoing_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![7.0, 8.0, 9.0]), 2), 0);

    let incoming_edges: Vec<Rc<RefCell<Edge>>> = vec![incoming_edge_1.clone(), incoming_edge_2.clone()].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();
    let outgoing_edges: Vec<Rc<RefCell<Edge>>> = vec![outgoing_edge_1].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();

    let mut node: Node = Node::new(incoming_edges, outgoing_edges, 0);
    let inputs: Vector = Vector::new(vec![0.0, 1.0]);
    let upstream_gradient: f64 = 0.25;
    node.backward(inputs.clone(), upstream_gradient).unwrap();
    let learning_rate: f64 = 0.1;
    node.update_weights(learning_rate).unwrap();

    // Expected values
    let expected_gradient_1: Vector = Vector::from(vec![incoming_edge_1.spline.basis(0, incoming_edge_1.spline.degree, inputs[0]), incoming_edge_1.spline.basis(1, incoming_edge_1.spline.degree, inputs[0]), incoming_edge_1.spline.basis(2, incoming_edge_1.spline.degree, inputs[0])]) * upstream_gradient;
    let expected_gradient_2: Vector = Vector::from(vec![incoming_edge_2.spline.basis(0, incoming_edge_2.spline.degree, inputs[1]), incoming_edge_2.spline.basis(1, incoming_edge_2.spline.degree, inputs[1]), incoming_edge_2.spline.basis(2, incoming_edge_2.spline.degree, inputs[1])]) * upstream_gradient;

    let result_control_points_1: Vector = incoming_edge_1.spline.control_points.clone() - expected_gradient_1 * learning_rate;
    let result_control_points_2: Vector = incoming_edge_2.spline.control_points.clone() - expected_gradient_2 * learning_rate;

    assert_eq!(node.incoming[0].borrow().spline.control_points, result_control_points_1);
    assert_eq!(node.incoming[1].borrow().spline.control_points, result_control_points_2);
}

#[test]
fn node_train_pass() {
    let mut incoming_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![1.0, 2.0, 3.0]), 2), 0);
    let mut incoming_edge_2: Edge = Edge::new(1, 0, BSpline::new(Vector::new(vec![4.0, 5.0, 6.0]), 2), 0);

    let outgoing_edge_1: Edge = Edge::new(0, 0, BSpline::new(Vector::new(vec![7.0, 8.0, 9.0]), 2), 0);

    let incoming_edges: Vec<Rc<RefCell<Edge>>> = vec![incoming_edge_1.clone(), incoming_edge_2.clone()].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();
    let outgoing_edges: Vec<Rc<RefCell<Edge>>> = vec![outgoing_edge_1].iter().map(|edge| Rc::new(RefCell::new(edge.clone()))).collect();

    let mut node: Node = Node::new(incoming_edges, outgoing_edges, 0);

    let inputs: Vector = Vector::new(vec![0.0, 1.0]);
    let forward_result: f64 = node.forward(&inputs);
    let actual_value: f64 = 0.5;

    let mse_gradient: f64 = forward_result - actual_value;

    // Expected forward result
    let expected_forward_result: f64 = incoming_edge_1.forward(0.0) + incoming_edge_2.forward(1.0);
    assert_is_close!(forward_result, expected_forward_result, 1e-3);

    node.backward(inputs.clone(), mse_gradient).unwrap();

    // Expected gradients
    let expected_gradient_1: Vector = Vector::from(vec![incoming_edge_1.spline.basis(0, incoming_edge_1.spline.degree, inputs[0]), incoming_edge_1.spline.basis(1, incoming_edge_1.spline.degree, inputs[0]), incoming_edge_1.spline.basis(2, incoming_edge_1.spline.degree, inputs[0])]) * mse_gradient;
    let expected_gradient_2: Vector = Vector::from(vec![incoming_edge_2.spline.basis(0, incoming_edge_2.spline.degree, inputs[1]), incoming_edge_2.spline.basis(1, incoming_edge_2.spline.degree, inputs[1]), incoming_edge_2.spline.basis(2, incoming_edge_2.spline.degree, inputs[1])]) * mse_gradient;
    
    assert_is_close!(node.incoming[0].borrow().gradient[0], expected_gradient_1[0], 1e-3);
    assert_is_close!(node.incoming[0].borrow().gradient[1], expected_gradient_1[1], 1e-3);
    assert_is_close!(node.incoming[0].borrow().gradient[2], expected_gradient_1[2], 1e-3);
    assert_is_close!(node.incoming[1].borrow().gradient[0], expected_gradient_2[0], 1e-3);
    assert_is_close!(node.incoming[1].borrow().gradient[1], expected_gradient_2[1], 1e-3);
    assert_is_close!(node.incoming[1].borrow().gradient[2], expected_gradient_2[2], 1e-3);

    let learning_rate: f64 = 0.1;
    node.update_weights(learning_rate).unwrap();
}