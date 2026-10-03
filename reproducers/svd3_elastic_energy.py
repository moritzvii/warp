"""A stretch-based Hooke energy can have a gradient with the wrong sign."""

import numpy as np

import warp as wp


@wp.kernel
def evaluate(a: wp.array[wp.mat33], loss: wp.array[float]):
    _u, s, v = wp.svd3(a[0])
    strain = v * wp.diag(s) * wp.transpose(v) - wp.identity(n=3, dtype=float)
    loss[0] = wp.ddot(strain, strain) + 0.5 * wp.trace(strain) * wp.trace(strain)


def reference(a):
    _u, s, vt = np.linalg.svd(a)
    strain = vt.T @ np.diag(s) @ vt - np.eye(3)
    return np.sum(strain * strain) + 0.5 * np.trace(strain) ** 2


wp.set_device("cpu")
value = np.array(
    [
        [-0.20882338285446167, -1.3351235389709473, -1.4743940830230713],
        [0.9388079643249512, -1.3731375932693481, 1.110465168952942],
        [-1.7535769939422607, -0.5761409401893616, 0.7700839638710022],
    ],
    dtype=np.float32,
)
direction = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
a = wp.array(value[None], dtype=wp.mat33, requires_grad=True)
loss = wp.zeros(1, requires_grad=True)
with wp.Tape() as tape:
    wp.launch(evaluate, dim=1, inputs=[a, loss])
tape.backward(loss=loss)
print("Energy:", loss.numpy()[0], "NumPy:", reference(value.astype(float)))
print("Shear directional AD:", np.sum(a.grad.numpy()[0] * direction))
for h in (1e-2, 1e-3):
    values = []
    for sign in (1, -1):
        perturbed = wp.array((value + sign * h * direction)[None], dtype=wp.mat33)
        wp.launch(evaluate, dim=1, inputs=[perturbed, loss])
        values.append(float(loss.numpy()[0]))
    oracle = (reference(value.astype(float) + h * direction) - reference(value.astype(float) - h * direction)) / (2 * h)
    print("h =", h, "Warp FD:", (values[0] - values[1]) / (2 * h), "NumPy FD:", oracle)
