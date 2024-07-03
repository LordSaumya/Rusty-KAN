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

use std::vec;
use std::collections::HashMap;

use crate::data_structures::{vector::Vector, spline::BSpline};

#[test]
fn spline_new_pass() {
    let control_points: Vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let degree: usize = 2;
    let spline: BSpline = BSpline::new(control_points.clone(), degree);

    println!("{:?}", spline);

    assert_eq!(spline.control_points, control_points);
    assert_eq!(spline.knots.len(), spline.control_points.len() + spline.degree + 1);
    assert_eq!(spline.degree, 2);
}

#[test]
fn spline_eval_pass() {
    let control_points: Vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let knots: Vector = Vector::new(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    let degree: usize = 2;
    let mut spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone(), memo: HashMap::new()};

    println!("{:?}", spline);

    // Input values
    let t1: f64 = 0.3;
    let t2: f64 = 0.5;
    let t3: f64 = 0.7;

    // Actual results
    let result1: f64 = spline.eval(t1);
    let result2: f64 = spline.eval(t2);
    let result3: f64 = spline.eval(t3);

    // Expected results
    let expected_result1: f64 = 1.0;
    let expected_result2: f64 = 2.0;
    let expected_result3: f64 = 2.5;

    // Check that results are correct.
    assert_is_close!(result1, expected_result1, 1e-3);
    assert_is_close!(result2, expected_result2, 1e-3);
    assert_is_close!(result3, expected_result3, 1e-3);
}

#[test]
#[should_panic]
fn spline_eval_fail() {
    let control_points: Vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let knots: Vector = Vector::new(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    let degree: usize = 2;
    let mut spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone(), memo: HashMap::new()};

    // t < 0.0 -> should fail
    let _ = spline.eval(-0.5);
}

#[test]
fn spline_basis_pass() {
    let control_points: Vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let knots: Vector = Vector::new(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    let degree: usize = 2;
    let mut spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone(), memo: HashMap::new()};

    println!("{:?}", spline);

    // N 0,0 (0.0) = 1.0
    let result1: f64 = spline.basis(0, 0, 0.0);
    assert_is_close!(result1, 1.0, 1e-3);

    // N 1, 0 (0.0) = 0.0
    let result2: f64 = spline.basis(1, 0, 0.0);
    assert_is_close!(result2, 0.0, 1e-3);

    // N 0, 2 (0.45) = 0.28125
    let result3: f64 = spline.basis(0, 2, 0.45);
    assert_is_close!(result3, 0.28125, 1e-3);

    // N 1, 2 (0.45) = 0.6875
    let result4: f64 = spline.basis(1, 2, 0.45);
    assert_is_close!(result4, 0.6875, 1e-3);

    // N 2, 2 (0.45) = 0.03125
    let result5: f64 = spline.basis(2, 2, 0.45);
    assert_is_close!(result5, 0.03125, 1e-3);
}

#[test]
#[should_panic]
fn spline_basis_fail() {
    let control_points: Vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let knots: Vector = Vector::new(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    let degree: usize = 2;
    let mut spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone(), memo: HashMap::new()};

    // i > degree -> should fail
    let _ = spline.basis(3, 2, 0.5);
}