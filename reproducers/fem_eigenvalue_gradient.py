"""Candidate: differentiating the separate iterative FEM eigensolver's reconstruction."""

import numpy as np

import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr


@wp.kernel
def evaluate(a: wp.array[wp.mat33d], loss: wp.array[wp.float64]):
    d, p = symmetric_eigenvalues_qr(a[0], wp.float64(0.0))
    b = wp.transpose(p) * wp.diag(d) * p
    loss[0] = b[0, 1] + b[1, 0]


wp.set_device("cpu")
value = np.diag([1.0, 2.0, 3.0])
direction = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
a = wp.array(value[None], dtype=wp.mat33d, requires_grad=True)
loss = wp.zeros(1, dtype=wp.float64, requires_grad=True)
with wp.Tape() as tape:
    wp.launch(evaluate, dim=1, inputs=[a, loss])
tape.backward(loss=loss)
print("Reconstruction AD:", np.sum(a.grad.numpy()[0] * direction), "expected 2")
for h in (1e-3, 1e-4, 1e-5):
    values = []
    for sign in (1, -1):
        perturbed = wp.array((value + sign * h * direction)[None], dtype=wp.mat33d)
        wp.launch(evaluate, dim=1, inputs=[perturbed, loss])
        values.append(float(loss.numpy()[0]))
    print("h =", h, "Warp forward difference:", (values[0] - values[1]) / (2 * h))
