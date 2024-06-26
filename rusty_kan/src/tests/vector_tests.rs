use crate::data_structures::{vector::Vector, matrix::Matrix};

#[test]
fn vector_add_pass() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let b = Vector { elements: vec![4.0, 5.0, 6.0] };

    let c = a + b;

    assert_eq!(c.elements, vec![5.0, 7.0, 9.0]);
}

#[test]
#[should_panic]
fn vector_add_fail() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let b = Vector { elements: vec![4.0, 5.0] };
    let _ = a + b;
}

#[test]
fn vector_sub_pass() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let b = Vector { elements: vec![4.0, 5.0, 6.0] };

    let c = a - b;

    assert_eq!(c.elements, vec![-3.0, -3.0, -3.0]);
}

#[test]
#[should_panic]
fn vector_sub_fail() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let b = Vector { elements: vec![4.0, 5.0] };
    let _ = a - b;
}

#[test]
fn vector_mul_scalar_pass() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let scalar = 2.0;

    let b = a * scalar;
    assert_eq!(b.elements, vec![2.0, 4.0, 6.0]);
}

#[test]
fn vector_div_scalar_pass() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let scalar = 2.0;

    let b = a / scalar;
    assert_eq!(b.elements, vec![0.5, 1.0, 1.5]);
}

#[test]
#[should_panic]
fn vector_div_scalar_fail() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let scalar = 0.0;

    let _ = a / scalar;
}

#[test]
fn vector_index_pass() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };

    let b = a[1];
    assert_eq!(b, 2.0);
}

#[test]
#[should_panic]
fn vector_index_fail() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };

    let _ = a[3];
}

#[test]
fn vector_dot_pass() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let b = Vector { elements: vec![4.0, 5.0, 6.0] };

    let c = a.dot(&b);

    assert_eq!(c, 32.0);
}

#[test]
#[should_panic]
fn vector_dot_fail() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let b = Vector { elements: vec![4.0, 5.0] };

    a.dot(&b);
}

#[test]
fn vector_element_wise_pass() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let b = Vector { elements: vec![4.0, 5.0, 6.0] };

    let c = a.element_wise(&b);

    assert_eq!(c.elements, vec![4.0, 10.0, 18.0]);
}

#[test]
#[should_panic]
fn vector_element_wise_fail() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let b = Vector { elements: vec![4.0, 5.0] };

    a.element_wise(&b);
}

#[test]
fn vector_len_pass() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };

    let b = a.len();

    assert_eq!(b, 3);
}

#[test]
fn vector_zeros_pass() {
    let a = Vector::zeros(3);

    assert_eq!(a.elements, vec![0.0, 0.0, 0.0]);
}

#[test]
fn vector_ones_pass() {
    let a = Vector::ones(3);

    assert_eq!(a.elements, vec![1.0, 1.0, 1.0]);
}

#[test]
fn vector_random_pass() {
    let a = Vector::random(3);

    assert_eq!(a.elements.len(), 3);
}

#[test]
fn vector_to_matrix_pass() {
    let a = Vector { elements: vec![1.0, 2.0, 3.0] };
    let b = a.to_matrix();
    let c = Matrix::new(vec![a.clone()]);
    assert_eq!(b, c);
}