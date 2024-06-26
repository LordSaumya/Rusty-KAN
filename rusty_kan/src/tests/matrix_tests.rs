use crate::data_structures::{vector::Vector, matrix::Matrix};

#[test]
fn matrix_add_pass() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0, 3.0] }, Vector { elements: vec![4.0, 5.0, 6.0] }] };
    let b = Matrix { rows: vec![Vector { elements: vec![7.0, 8.0, 9.0] }, Vector { elements: vec![10.0, 11.0, 12.0] }] };

    let c = a + b;

    assert_eq!(c.rows, vec![Vector { elements: vec![8.0, 10.0, 12.0] }, Vector { elements: vec![14.0, 16.0, 18.0] }]);
}

#[test]
#[should_panic]
fn matrix_add_fail() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0, 3.0] }, Vector { elements: vec![4.0, 5.0, 6.0] }] };
    let b = Matrix { rows: vec![Vector { elements: vec![7.0, 8.0, 9.0] }] };
    let _ = a + b;
}

#[test]
fn matrix_sub_pass() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0, 3.0] }, Vector { elements: vec![4.0, 5.0, 6.0] }] };
    let b = Matrix { rows: vec![Vector { elements: vec![7.0, 8.0, 9.0] }, Vector { elements: vec![10.0, 11.0, 12.0] }] };

    let c = a - b;

    assert_eq!(c.rows, vec![Vector { elements: vec![-6.0, -6.0, -6.0] }, Vector { elements: vec![-6.0, -6.0, -6.0] }]);
}

#[test]
#[should_panic]
fn matrix_sub_fail() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0, 3.0] }, Vector { elements: vec![4.0, 5.0, 6.0] }] };
    let b = Matrix { rows: vec![Vector { elements: vec![7.0, 8.0, 9.0] }] };
    let _ = a - b;
}

#[test]
fn matrix_mul_scalar_pass() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0, 3.0] }, Vector { elements: vec![4.0, 5.0, 6.0] }] };
    let scalar = 2.0;

    let b = a * scalar;

    assert_eq!(b.rows, vec![Vector { elements: vec![2.0, 4.0, 6.0] }, Vector { elements: vec![8.0, 10.0, 12.0] }]);
}

#[test]
fn matrix_mul_vector_pass() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0, 3.0] }, Vector { elements: vec![4.0, 5.0, 6.0] }] };
    let b = Vector { elements: vec![7.0, 8.0, 9.0] };
    
    let c = a * b;

    assert_eq!(c.elements, vec![50.0, 122.0]);
}

#[test]
fn matrix_mul_matrix_pass() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0] }, Vector { elements: vec![3.0, 4.0] }] };
    let b = Matrix { rows: vec![Vector { elements: vec![5.0, 6.0] }, Vector { elements: vec![7.0, 8.0] }] };

    let c = a * b;

    assert_eq!(c.rows, vec![Vector { elements: vec![19.0, 22.0] }, Vector { elements: vec![43.0, 50.0] }]);
}

#[test]
#[should_panic]
fn matrix_mul_matrix_fail() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0, 3.0] }, Vector { elements: vec![4.0, 5.0, 6.0] }] };
    let b = Matrix { rows: vec![Vector { elements: vec![4.0, 5.0] }, Vector { elements: vec![7.0, 8.0] }] };
    let _ = a * b;
}

#[test]
fn matrix_index_pass() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0, 3.0] }, Vector { elements: vec![4.0, 5.0, 6.0] }] };

    assert_eq!(a[0][0], 1.0);
    assert_eq!(a[0][1], 2.0);
    assert_eq!(a[0][2], 3.0);
    assert_eq!(a[1][0], 4.0);
    assert_eq!(a[1][1], 5.0);
    assert_eq!(a[1][2], 6.0);
}

#[test]
#[should_panic]
fn matrix_index_fail() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0, 3.0] }, Vector { elements: vec![4.0, 5.0, 6.0] }] };
    let _ = a[0][3];
}

#[test]
fn matrix_zeros_pass() {
    let a = Matrix::zeros(2, 3);

    assert_eq!(a.rows, vec![Vector { elements: vec![0.0, 0.0, 0.0] }, Vector { elements: vec![0.0, 0.0, 0.0] }]);
}

#[test]
fn matrix_ones_pass() {
    let a = Matrix::ones(2, 3);

    assert_eq!(a.rows, vec![Vector { elements: vec![1.0, 1.0, 1.0] }, Vector { elements: vec![1.0, 1.0, 1.0] }]);
}

#[test]
fn matrix_random_pass() {
    let a = Matrix::random(2, 3);

    assert_eq!(a.rows.len(), 2);
    assert_eq!(a.rows[0].elements.len(), 3);
    assert_eq!(a.rows[1].elements.len(), 3);
}

#[test]
fn matrix_identity_pass() {
    let a = Matrix::identity(3);
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                assert_eq!(a[i][j], 1.0);
            } else {
                assert_eq!(a[i][j], 0.0);
            }
        }
    }
}

#[test]
fn matrix_shape_pass() {
    let a = Matrix::zeros(2, 3);

    assert_eq!(a.shape(), (2, 3));
}

#[test]
fn matrix_get_col_pass() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0, 3.0] }, Vector { elements: vec![4.0, 5.0, 6.0] }] };

    let b = a.get_col(1);

    assert_eq!(b.elements, vec![2.0, 5.0]);
}

#[test]
#[should_panic]
fn matrix_get_col_fail() {
    let a = Matrix::zeros(2, 3);
    let _ = a.get_col(3);
}

#[test]
fn matrix_set_row_pass() {
    let mut a = Matrix::zeros(2, 3);
    let b = Vector { elements: vec![1.0, 2.0, 3.0] };

    a.set_row(1, b);

    assert_eq!(a.rows, vec![Vector { elements: vec![0.0, 0.0, 0.0] }, Vector { elements: vec![1.0, 2.0, 3.0] }]);
}

#[test]
#[should_panic]
fn matrix_set_row_fail() {
    let mut a = Matrix::zeros(2, 3);
    let b = Vector { elements: vec![1.0, 2.0] };
    a.set_row(1, b);
}

#[test]
fn matrix_set_col_pass() {
    let mut a = Matrix::zeros(2, 3);
    let b = Vector { elements: vec![1.0, 2.0] };

    a.set_col(1, b);

    assert_eq!(a.rows, vec![Vector { elements: vec![0.0, 1.0, 0.0] }, Vector { elements: vec![0.0, 2.0, 0.0] }]);
}

#[test]
#[should_panic]
fn matrix_set_col_fail() {
    let mut a = Matrix::zeros(2, 3);
    let b = Vector { elements: vec![1.0, 2.0, 3.0] };
    a.set_col(1, b);
}

#[test]
fn matrix_transpose_pass() {
    let a = Matrix { rows: vec![Vector { elements: vec![1.0, 2.0, 3.0] }, Vector { elements: vec![4.0, 5.0, 6.0] }] };

    let b = a.transpose();

    assert_eq!(b.rows, vec![Vector { elements: vec![1.0, 4.0] }, Vector { elements: vec![2.0, 5.0] }, Vector { elements: vec![3.0, 6.0] }]);
}