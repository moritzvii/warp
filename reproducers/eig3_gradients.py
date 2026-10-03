"""Eigenvalue sums and smooth reconstruction at repeated or nearby eigenvalues."""

import numpy as np

import warp as wp


@wp.kernel
def evaluate(a: wp.array[wp.mat33d], loss: wp.array[wp.float64], reconstruct: int):
    q, d = wp.eig3(a[0])
    if reconstruct == 0:
        loss[0] = d[0] + d[1] + d[2]
    else:
        b = q * wp.diag(d) * wp.transpose(q)
        loss[0] = b[0, 1] + b[1, 0]


wp.set_device("cpu")
direction = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
for diagonal in ([1, 2, 3], [2, 2, 3], [2, 2 + 1e-7, 3]):
    value = np.diag(diagonal).astype(float)
    for reconstruct in (0, 1):
        a = wp.array(value[None], dtype=wp.mat33d, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float64, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(evaluate, dim=1, inputs=[a, loss, reconstruct])
        tape.backward(loss=loss)
        gradient = a.grad.numpy()[0]
        if reconstruct == 0:
            print(diagonal, "sum: gradient =", gradient, "expected identity")
        else:
            h = 1e-5
            perturbed = wp.array(np.array([value + h * direction]), dtype=wp.mat33d)
            wp.launch(evaluate, dim=1, inputs=[perturbed, loss, reconstruct])
            plus = float(loss.numpy()[0])
            perturbed = wp.array(np.array([value - h * direction]), dtype=wp.mat33d)
            wp.launch(evaluate, dim=1, inputs=[perturbed, loss, reconstruct])
            fd = (plus - float(loss.numpy()[0])) / (2 * h)
            print(diagonal, "reconstruction: AD =", np.sum(gradient * direction), "FD =", fd, "expected 2")
