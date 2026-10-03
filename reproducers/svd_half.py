"""Half-precision SVD gives NaN gradients for the smooth squared norm."""

import numpy as np

import warp as wp


def make_kernel(n, scalar):
    matrix = wp.types.matrix((n, n), dtype=scalar)

    @wp.kernel(module="unique")
    def evaluate(a: wp.array[matrix], loss: wp.array[scalar]):
        if wp.static(n == 2):
            _u, s, _v = wp.svd2(a[0])
        else:
            _u, s, _v = wp.svd3(a[0])
        loss[0] = wp.dot(s, s)

    return evaluate, matrix


wp.set_device("cpu")
for n in (2, 3):
    for scalar in (wp.float32, wp.float16):
        kernel, matrix = make_kernel(n, scalar)
        a = wp.array((2 * np.eye(n))[None], dtype=matrix, requires_grad=True)
        loss = wp.zeros(1, dtype=scalar, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(kernel, dim=1, inputs=[a, loss])
        tape.backward(loss=loss)
        print(f"svd{n} {scalar.__name__}: loss =", loss.numpy()[0], "expected:", 4 * n)
        print("Gradient:", a.grad.numpy()[0], "expected:", 4 * np.eye(n))
