"""NaN gradients at zero in smooth compositions and unused root expressions."""

import warp as wp


@wp.kernel
def evaluate(x: wp.array[float], loss: wp.array[float], mode: int):
    value = x[0]
    if mode == 0:
        root = wp.sqrt(value * value)
        loss[0] = root * root
    elif mode == 1:
        root = wp.cbrt(value)
        loss[0] = root * root * root
    elif mode == 2:
        loss[0] = value + 0.0 * wp.sqrt(wp.abs(value))
    elif mode == 3:
        _unused = wp.sqrt(wp.abs(value))
        loss[0] = value
    else:
        length = wp.length(wp.vec3(value, 0.0, 0.0))
        loss[0] = length * length


wp.set_device("cpu")
for mode, name in enumerate(("sqrt(x*x)^2", "cbrt(x)^3", "zero-weight sqrt", "unused sqrt", "length squared control")):
    for value in (1.0, 0.0):
        x = wp.array([value], dtype=float, requires_grad=True)
        loss = wp.zeros(1, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(evaluate, dim=1, inputs=[x, loss, mode])
        tape.backward(loss=loss)
        expected = 2 * value if mode in (0, 4) else 1
        print(name, "x =", value, "loss =", loss.numpy()[0], "gradient =", x.grad.numpy()[0], "expected =", expected)
