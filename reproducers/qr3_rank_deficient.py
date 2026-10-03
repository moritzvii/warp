"""The smooth identity ||R||^2 = ||A||^2 gets a zero gradient at rank deficiency."""

import numpy as np

import warp as wp


@wp.kernel
def evaluate(a: wp.array[wp.mat33d], loss: wp.array[wp.float64]):
    _q, r = wp.qr3(a[0])
    loss[0] = wp.ddot(r, r)


wp.set_device("cpu")
for diagonal in ([1, 2, 3], [0, 0, 3]):
    value = np.diag(diagonal).astype(float)
    a = wp.array(value[None], dtype=wp.mat33d, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float64, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(evaluate, dim=1, inputs=[a, loss])
    tape.backward(loss=loss)
    print(diagonal, "loss:", loss.numpy()[0], "gradient:", a.grad.numpy()[0], "expected:", 2 * value)
    values = []
    h = 1e-5
    for sign in (1, -1):
        perturbed = value.copy()
        perturbed[2, 2] += sign * h
        wp.launch(evaluate, dim=1, inputs=[wp.array(perturbed[None], dtype=wp.mat33d), loss])
        values.append(float(loss.numpy()[0]))
    print("E22 forward difference:", (values[0] - values[1]) / (2 * h), "expected 6")
