"""Fundamental matrix utilities."""

import math
import numpy as np

def point_line_distance(line, point):
    """Calculate line-point distance according to the formula
    from the project webpage. 

    d(l, x) = (au + bv + c) / sqrt(a^2 + b^2)

    Arguments:
        line {3-vector} -- Line coordinates a, b, c
        point {3-vector} -- homogeneous point u, v, w

        Note that we don't really use w because w = 1 for the
        homogeneous coordinates

    Returns:
        float -- signed distance between line and point
    """
    a, b, c = line
    u, v, w = point
    error = 0

    ##############################
    # TODO: Student code goes here

    top = a*u + b*v + c
    bottom = np.sqrt(a*a + b*b)
    error = top / bottom

    ##############################

    return error

def signed_point_line_errors(x_0s, F, x_1s):
    """Calculate all signed line-to-point distances. Takes as input
    the list of x_0 and x_1 points, as well as the current estimate
    for F, and calculates errors for every pair of points and
    returns it as a list of floats.

    You'll want to call point_line_distance() to get the error
    between line and point.

    Keep in mind that this is a symmetric line-to-point error,
    meaning we calculate the line-to-point distance between Fx_1 and
    x_0, as well as F'x_0 and x_1, where F' is F transposed. 

    Use np.dot() to calculate Fx_1 and F'x_0
    
    Arguments:
        x_0s {list} -- points in image 1
        F {3x3 array} -- Fundamental matrix
        x_1s {list} -- points in image 2

    Returns:
        [float] -- list of d(Fx_1,x_0) and d(F'x_0,x_1) for each
        pair of points, because SciPy takes care of squaring and 
        summing
    """
    assert F.shape == (3, 3)
    assert len(x_0s) == len(x_1s)
    errors = []

    ##############################
    # TODO: Student code goes here

    for x in range(len(x_0s)):
        Fx1 = np.dot(F, x_1s[x])
        d1 = point_line_distance(Fx1, x_0s[x])
        errors.append(d1)
        Fx0 = np.dot(np.transpose(F), x_0s[x])
        d2 = point_line_distance(Fx0, x_1s[x])
        errors.append(d2)

    ##############################

    return errors

def skew(x, y, z):
    """Skew symmetric matrix."""
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], np.float)


def create_F(K, R, t):
    """Create F from calibration and pose R,t between two views.
    Used in unit tests

    Arguments:
        K {3x3 matrix} -- Calibration matrix
        R {3x3 matrix} -- wRc, rotation from second camera to first (world)
        t {3-vector} -- wtc, position of camera in first (world)
    """
    x, y, z = t
    T = skew(x, y, z)
    Kinv = np.linalg.inv(K)
    F = np.dot(Kinv.T, T).dot(R).dot(Kinv)
    return F / np.linalg.norm(F)