import numpy as np

def spot_rate(term: int, face_value: np.array, price: np.array, coupon_rate: np.array):
    F = face_value
    P = price
    c = coupon_rate
    C = c*F
    t = term

    A = np.ones((t, t))
    A = C*A
    A = A.T
    A = np.tril(A)
    A += np.eye(t)*F
    A_inv = np.linalg.inv(A)
    S = np.log(A_inv@P)
    T = np.arange(1, t+1)
    S = S/T
    S = -S
    return S