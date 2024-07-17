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

use std::collections::HashMap;
use crate::data_structures::{vector::Vector, spline::BSpline, edge::Edge};

#[test]
fn edge_new_pass() {
    let control_points: Vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let degree: usize = 2;
    let spline: BSpline = BSpline::new(control_points.clone(), degree);
    let edge: Edge = Edge::new(0, 1, spline, 0);

    assert_eq!(edge.start, 0);
    assert_eq!(edge.end, 1);
    assert_eq!(edge.spline.control_points, control_points);
    assert_eq!(edge.spline.knots.len(), edge.spline.control_points.len() + edge.spline.degree + 1);
    assert_eq!(edge.spline.degree, 2);

    assert_eq!(edge.gradient.elements.len(), edge.spline.control_points.len());
    assert_eq!(edge.layer, 0);
}

#[test]
fn edge_standard_pass() {
    let start: usize = 0;
    let end: usize = 1;
    let layer: usize = 0;

    let edge: Edge = Edge::standard(start, end, layer);

    assert_eq!(edge.start, start);
    assert_eq!(edge.end, end);
    assert_eq!(edge.layer, layer);

    assert_eq!(edge.spline.control_points.len(), 5);
    assert_eq!(edge.spline.knots.len(), edge.spline.control_points.len() + edge.spline.degree + 1);
    assert_eq!(edge.spline.degree, 2);
}

#[test]
fn edge_forward_pass() {
    let control_points: Vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let knots: Vector = Vector::new(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    let degree: usize = 2;
    let spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone(), memo: HashMap::new() };
    let mut edge: Edge = Edge::new(0, 1, spline.clone(), 0);

    println!("{:?}", spline);

    // Basis function
    let silu = |x: f64| x/(1.0 + (-x).exp());

    // Input values
    let t1: f64 = 0.3;
    let t2: f64 = 0.5;
    let t3: f64 = 0.7;

    // Actual results
    let result1: f64 = edge.forward(t1);
    let result2: f64 = edge.forward(t2);
    let result3: f64 = edge.forward(t3);

    // Expected results
    let expected_result1: f64 = 1.0 + silu(t1);
    let expected_result2: f64 = 2.0 + silu(t2);
    let expected_result3: f64 = 2.5 + silu(t3);

    // Check that results are correct.
    assert_is_close!(result1, expected_result1, 1e-3);
    assert_is_close!(result2, expected_result2, 1e-3);
    assert_is_close!(result3, expected_result3, 1e-3);
}

#[test]
fn edge_forward_batch_pass() {
    let control_points: Vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let knots: Vector = Vector::new(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    let degree: usize = 2;
    let spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone(), memo: HashMap::new() };
    let mut edge: Edge = Edge::new(0, 1, spline.clone(), 0);
    
    let inputs: Vector = Vector::new(vec![0.3, 0.5, 0.7]);
    let result: Vector = edge.forward_batch(inputs.clone());

    let silu = |x: f64| x/(1.0 + (-x).exp());

    // Expected results
    let expected_result1: f64 = 1.0 + silu(0.3);
    let expected_result2: f64 = 2.0 + silu(0.5);
    let expected_result3: f64 = 2.5 + silu(0.7);
    let expected_result: Vector = Vector::new(vec![expected_result1, expected_result2, expected_result3]);

    // Check that results lengths are correct.
    assert_eq!(result.len(), expected_result.len());

    print!("Results: {}\n\n", result);
    print!("Expected Results: {}", expected_result);

    // Check that results are correct.
    for i in 0..result.len() {
        assert_is_close!(result[i], expected_result[i], 1e-3);
    }
}

#[test]
fn edge_backward_pass() {
    let control_points: Vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let knots: Vector = Vector::new(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    let degree: usize = 2;
    let mut spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone(), memo: HashMap::new() };
    let mut edge: Edge = Edge::new(0, 1, spline.clone(), 0);

    let t: f64 = 0.1;
    let upstream_gradient: f64 = 1.0;

    // Checks if returned result type is ok
    let _ = edge.backward(t, upstream_gradient).unwrap();

    // Checks if gradient is ok
    let result_gradient: Vector = edge.gradient.clone();
    print!("Result Gradient: {}\n", result_gradient);
    let expected_gradient: Vec<f64> = vec![1.0 * spline.basis(0, spline.degree, t), 1.0 * spline.basis(1, spline.degree, t), 1.0 * spline.basis(2, spline.degree, t)];
    for i in 0..result_gradient.elements.len() {
        assert_is_close!(result_gradient.elements[i], expected_gradient[i], 1e-3);
    }
}

#[test]
fn edge_weight_update_pass() {
    let control_points: Vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let knots: Vector = Vector::new(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    let degree: usize = 2;
    let mut spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone(), memo: HashMap::new() };
    let mut edge: Edge = Edge::new(0, 1, spline.clone(), 0);

    let learning_rate: f64 = 0.1;

    edge.backward(0.1, 1.0).unwrap();
    edge.update_weights(learning_rate).unwrap();

    let result_control_points: Vector = edge.spline.control_points.clone();
    print!("Result Control Points: {}\n", result_control_points);
    let expected_control_points: Vec<f64> = vec![1.0 - learning_rate * 1.0 * spline.basis(0, spline.degree, 0.1), 2.0 - learning_rate * 1.0 * spline.basis(1, spline.degree, 0.1), 3.0 - learning_rate * 1.0 * spline.basis(2, spline.degree, 0.1)];
    for i in 0..result_control_points.elements.len() {
        assert_is_close!(result_control_points.elements[i], expected_control_points[i], 1e-3);
    }
}

#[test]
#[should_panic]
fn edge_weight_update_fail() {
    let control_points: Vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let knots: Vector = Vector::new(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    let degree: usize = 2;
    let spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone(), memo: HashMap::new() };
    let mut edge: Edge = Edge::new(0, 1, spline.clone(), 0);

    edge.update_weights(-0.1).unwrap();
}