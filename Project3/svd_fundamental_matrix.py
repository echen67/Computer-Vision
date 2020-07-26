import numpy as np


def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.
    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B
    -   normalize: Boolean, set to "True" if you normalize the coordinates
                   before calculating F
    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    mean_a = np.mean(points_a, axis=0)
    mean_b = np.mean(points_b, axis=0)
    std_a = np.std(points_a, axis=0)
    std_b = np.std(points_b, axis=0)
    T_a = np.asarray([[1.0/std_a[0], 0, -mean_a[0]/std_a[0]],
                      [0, 1.0/std_a[1], -mean_a[1]/std_a[1]],
                      [0, 0, 1]])
    T_b = np.asarray([[1.0/std_b[0], 0, -mean_b[0]/std_b[0]],
                      [0, 1.0/std_b[1], -mean_b[1]/std_b[1]],
                      [0, 0, 1]])
    points_a = np.hstack((points_a, np.ones((len(points_a), 1)))).T
    points_b = np.hstack((points_b, np.ones((len(points_b), 1)))).T
    points_a = np.dot(T_a, points_a)[:2].T
    points_b = np.dot(T_b, points_b)[:2].T

    A = []
    for pa, pb in zip(points_a, points_b):
        ua, va = pa
        ub, vb = pb
        A.append([ua*ub, va*ub, ub, ua*vb, va*vb, vb, ua, va, 1])
    A = np.vstack(A)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1, :].reshape((3, 3))

    # enforce the singularity constraint
    U, D, Vt = np.linalg.svd(F)
    D[-1] = 0
    F = np.dot(np.dot(U, np.diag(D)), Vt)

    F = np.dot(np.dot(T_b.T, F), T_a)

    return F