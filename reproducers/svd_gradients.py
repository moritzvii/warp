"""Smooth reconstruction, stretch and polar functions at A = 2I."""

import numpy as np

import warp as wp


def make_kernel(n):
    matrix = wp.types.matrix((n, n), dtype=wp.float64)

    @wp.kernel(module="unique")
    def evaluate(a: wp.array[matrix], loss: wp.array[wp.float64], mode: int):
        if wp.static(n == 2):
            u, s, v = wp.svd2(a[0])
        else:
            u, s, v = wp.svd3(a[0])
        b = u * wp.diag(s) * wp.transpose(v)
        if mode == 1:
            b = v * wp.diag(s) * wp.transpose(v)
        elif mode == 2:
            b = u * wp.transpose(v)
        loss[0] = b[0, 1]

    return evaluate, matrix


wp.set_device("cpu")
for n in (2, 3):
    kernel, matrix = make_kernel(n)
    for mode, name in enumerate(("reconstruction", "stretch", "polar")):
        direction = np.zeros((n, n))
        direction[0, 1] = 1
        direction[1, 0] = -1 if mode == 2 else 1
        a = wp.array((2 * np.eye(n))[None], dtype=matrix, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float64, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(kernel, dim=1, inputs=[a, loss, mode])
        tape.backward(loss=loss)
        ad = np.sum(a.grad.numpy()[0] * direction)
        differences = []
        for h in (1e-3, 1e-4, 1e-5):
            values = []
            for sign in (1, -1):
                perturbed = wp.array((2 * np.eye(n) + sign * h * direction)[None], dtype=matrix)
                wp.launch(kernel, dim=1, inputs=[perturbed, loss, mode])
                values.append(float(loss.numpy()[0]))
            differences.append((values[0] - values[1]) / (2 * h))
        print(f"svd{n} {name}: AD = {ad}, Warp FD = {differences}, expected = {0.5 if mode == 2 else 1.0}")
