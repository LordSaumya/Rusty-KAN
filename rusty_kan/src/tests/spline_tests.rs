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

use crate::data_structures::{vector::Vector, spline::BSpline};

#[test]
fn spline_new_pass() {
    let control_points: Vec<Vector> = vec![Vector { elements: vec![1.0, 2.0] }, Vector { elements: vec![3.0, 4.0] }, Vector { elements: vec![5.0, 6.0] }];
    let degree: usize = 2;
    let spline: BSpline = BSpline::new(control_points, degree);

    println!("{:?}", spline);

    assert_eq!(spline.control_points, vec![Vector { elements: vec![1.0, 2.0] }, Vector { elements: vec![3.0, 4.0] }, Vector { elements: vec![5.0, 6.0] }]);
    assert_eq!(spline.knots.len(), spline.control_points.len() + spline.degree + 1);
    assert_eq!(spline.degree, 2);
}

#[test]
fn spline_eval_pass() {
    let control_points: Vec<Vector> = vec![Vector { elements: vec![1.0, 2.0] }, Vector { elements: vec![3.0, 4.0] }, Vector { elements: vec![5.0, 6.0] }];
    let knots: Vec<f64> = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let degree: usize = 2;
    let spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone() };

    println!("{:?}", spline);

    // Input values
    let t1: f64 = 0.3;
    let t2: f64 = 0.5;
    let t3: f64 = 0.7;

    // Actual results
    let result1: Vector = spline.eval(t1);
    let result2: Vector = spline.eval(t2);
    let result3: Vector = spline.eval(t3);

    // Expected results
    let expected_result1: Vector = Vector { elements: vec![1.125, 2.0] };
    let expected_result2: Vector = Vector { elements: vec![3.0, 4.0] };
    let expected_result3: Vector = Vector { elements: vec![4.125, 5.0] };

    // Check that results lengths are correct.
    assert_eq!(result1.elements.len(), control_points[0].elements.len());
    assert_eq!(result2.elements.len(), control_points[0].elements.len());
    assert_eq!(result3.elements.len(), control_points[0].elements.len());

    // Check that results are correct.
    for i in 0..result1.elements.len() {
        assert_is_close!(result1.elements[i], expected_result1.elements[i], 1e-3);
        assert_is_close!(result2.elements[i], expected_result2.elements[i], 1e-3);
        assert_is_close!(result3.elements[i], expected_result3.elements[i], 1e-3);
    }
}

#[test]
#[should_panic]
fn spline_eval_fail() {
    let control_points: Vec<Vector> = vec![Vector { elements: vec![1.0, 2.0] }, Vector { elements: vec![3.0, 4.0] }, Vector { elements: vec![5.0, 6.0] }];
    let degree: usize = 2;
    let knots: Vec<f64> = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone() };

    // t < 0.0 -> should fail
    let _ = spline.eval(-0.5);
}

#[test]
fn spline_basis_pass() {
    let control_points: Vec<Vector> = vec![Vector { elements: vec![1.0, 2.0] }, Vector { elements: vec![3.0, 4.0] }, Vector { elements: vec![5.0, 6.0] }];
    let knots: Vec<f64> = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let degree: usize = 2;
    let spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone() };

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
    let control_points: Vec<Vector> = vec![Vector { elements: vec![1.0, 2.0] }, Vector { elements: vec![3.0, 4.0] }, Vector { elements: vec![5.0, 6.0] }];
    let degree: usize = 2;
    let knots: Vec<f64> = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let spline: BSpline = BSpline { control_points: control_points.clone(), knots: knots.clone(), degree: degree.clone() };

    // i > degree -> should fail
    let _ = spline.basis(3, 2, 0.5);
}