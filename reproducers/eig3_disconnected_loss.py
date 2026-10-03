"""An unrelated eig3 diagnostic on the same tape contaminates trace gradients."""

import numpy as np

import warp as wp


@wp.kernel
def diagnostic(a: wp.array[wp.mat33], output: wp.array[float]):
    _q, d = wp.eig3(a[0])
    output[0] = d[0] + d[1] + d[2]


@wp.kernel
def trace_loss(a: wp.array[wp.mat33], loss: wp.array[float]):
    loss[0] = wp.trace(a[0])


wp.set_device("cpu")
for include_diagnostic in (False, True):
    a = wp.array(np.diag([2, 2, 3])[None], dtype=wp.mat33, requires_grad=True)
    output = wp.zeros(1, requires_grad=True)
    loss = wp.zeros(1, requires_grad=True)
    with wp.Tape() as tape:
        if include_diagnostic:
            wp.launch(diagnostic, dim=1, inputs=[a, output])
        wp.launch(trace_loss, dim=1, inputs=[a, loss])
    tape.backward(loss=loss)
    print("Diagnostic:", include_diagnostic, "loss:", loss.numpy()[0])
    print("Gradient:", a.grad.numpy()[0], "expected identity")
