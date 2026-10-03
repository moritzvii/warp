"""Zero seeds do not protect a parameter shared with a repeated-eigenvalue sample."""

import numpy as np

import warp as wp


@wp.kernel
def evaluate(p: wp.array[float], bases: wp.array[wp.mat33], losses: wp.array[float]):
    i = wp.tid()
    _q, d = wp.eig3(p[0] * bases[i])
    losses[i] = d[0] + d[1] + d[2]


wp.set_device("cpu")
for seeds in ([0, 1], [0, 0]):
    p = wp.array([1.0], dtype=float, requires_grad=True)
    bases = wp.array(np.array([np.diag([2, 2, 3]), np.diag([1, 2, 3])]), dtype=wp.mat33)
    losses = wp.zeros(2, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(evaluate, dim=2, inputs=[p, bases, losses])
    tape.backward(grads={losses: wp.array(seeds, dtype=float)})
    gradient = float(p.grad.numpy()[0])
    print("Seeds:", seeds, "gradient:", gradient, "expected:", 7 * seeds[0] + 6 * seeds[1])
    print("One SGD step:", 1 - 0.01 * gradient)
