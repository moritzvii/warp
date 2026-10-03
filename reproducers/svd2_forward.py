"""Invalid isotropic factors and NaN singular values for finite nonsingular inputs."""

import numpy as np

import warp as wp


def make_kernel(scalar):
    matrix = wp.types.matrix((2, 2), dtype=scalar)
    vector = wp.types.vector(2, dtype=scalar)

    @wp.kernel(module="unique")
    def evaluate(
        a: wp.array[matrix], reconstructed: wp.array[matrix], spectrum: wp.array[vector], loss: wp.array[scalar]
    ):
        u, s, v = wp.svd2(a[0])
        reconstructed[0] = u * wp.diag(s) * wp.transpose(v)
        spectrum[0] = s
        loss[0] = wp.dot(s, s)

    return evaluate, matrix, vector


wp.set_device("cpu")
for scalar, bad in (
    (wp.float32, [[0.879429280757904, 0.4445582628250122], [0.15186011791229248, 0.07688016444444656]]),
    (wp.float64, [[0.8034521044309734, 0.5815443494212681], [0.10333033149471714, 0.074791230333683]]),
):
    kernel, matrix, vector = make_kernel(scalar)
    for name, value in (
        ("distinct control", np.diag([1, 2])),
        ("negative isotropic", -2 * np.eye(2)),
        ("cancellation", np.array(bad)),
    ):
        a = wp.array(value[None], dtype=matrix, requires_grad=True)
        reconstructed = wp.zeros(1, dtype=matrix)
        spectrum = wp.zeros(1, dtype=vector)
        loss = wp.zeros(1, dtype=scalar, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(kernel, dim=1, inputs=[a, reconstructed, spectrum, loss])
        tape.backward(loss=loss)
        print(scalar.__name__, name)
        print("Reconstruction:", reconstructed.numpy()[0], "expected:", value)
        print(
            "Singular values:",
            spectrum.numpy()[0],
            "NumPy:",
            np.linalg.svd(a.numpy()[0].astype(float), compute_uv=False),
        )
        print("Norm-squared gradient:", a.grad.numpy()[0], "expected:", 2 * value)
