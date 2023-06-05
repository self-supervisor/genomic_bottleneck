# from chatGPT
from typing import List

import flax
import jax
from tensorflow.python.ops.numpy_ops import np_config

from inner_loop_utils import (
    compute_metrics,
    create_p_model,
    create_train_state,
    get_datasets,
    inner_loop_train_step,
)
from outer_loop_utils import (
    create_g_nets,
    g_net_train_step,
    generate_input_arr_for_g0,
    generate_input_arr_for_g0_bias,
    generate_input_arr_for_g1,
)

np_config.enable_numpy_behavior()

LR = 1e-4
MOMENTUM = 0.9
NUM_EPOCHS = 1
NUM_CLASSES = 10
BATCH_SIZE = 32
SEED = 0


def inner_loop(rng, w0, b0, w1) -> jax.Array:
    train_ds, test_ds = get_datasets(NUM_EPOCHS, BATCH_SIZE)

    num_steps_per_epoch = train_ds.cardinality().numpy() // NUM_EPOCHS
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    model = create_p_model(NUM_CLASSES)
    state = create_train_state(model, rng, LR, MOMENTUM)

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        state = inner_loop_train_step(state, batch)
        state = compute_metrics(state=state, batch=batch)

        if (step + 1) % num_steps_per_epoch == 0:
            for metric, value in state.metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)
            state = state.replace(metrics=state.metrics.empty())

            test_state = state
            for test_batch in test_ds.as_numpy_iterator():
                test_state = compute_metrics(state=test_state, batch=test_batch)

            for metric, value in test_state.metrics.compute().items():
                metrics_history[f"test_{metric}"].append(value)

            print(
                f"train epoch: {(step+1) // num_steps_per_epoch}, "
                f"loss: {metrics_history['train_loss'][-1]}, "
                f"accuracy: {metrics_history['train_accuracy'][-1] * 100}"
            )
            print(
                f"test epoch: {(step+1) // num_steps_per_epoch}, "
                f"loss: {metrics_history['test_loss'][-1]}, "
                f"accuracy: {metrics_history['test_accuracy'][-1] * 100}"
            )


def outer_loop(
    g_net_0_train_state,
    g_net_0_bias_train_state,
    g_net_1_train_state,
    p_net_train_state,
) -> List[jax.Array]:
    g_net_0, g_net_0_bias, g_net_1 = create_g_nets()
    g0_input, g0_bias_input, g1_input = (
        generate_input_arr_for_g0(),
        generate_input_arr_for_g0_bias(),
        generate_input_arr_for_g1(),
    )
    w0 = p_net_train_state.params["layers_0"]["kernel"]
    w0_labels = w0.reshape(-1)
    w0_bias = p_net_train_state.params["layers_0"]["bias"]
    w0_bias_labels = w0_bias.reshape(-1)
    w1 = p_net_train_state.params["layers_2"]["kernel"]
    w1_labels = w1.reshape(-1)
    # for epoch in range(NUM_EPOCHS):
    #     for batch_i in range(len(g0_input) // BATCH_SIZE):
    #         g_net_train_step(
    #             g_net_0_train_state,
    #             g0_input[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE],
    #             w0_labels[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE],
    #         )
    #     for batch_i in range(len(g0_bias_input) // BATCH_SIZE):
    #         g_net_train_step(
    #             g_net_0_bias_train_state,
    #             g0_bias_input[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE],
    #             w0_bias_labels[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE],
    #         )
    #     for batch_i in range(len(g1_input) // BATCH_SIZE):
    #         g_net_train_step(
    #             g_net_1_train_state,
    #             g1_input[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE],
    #             w1_labels[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE],
    #         )

    p_net_train_state.params.unfreeze()
    p_net_train_state.params["layers_0"]["kernel"] = g_net_0.apply(
        g_net_0_train_state.params, g0_input
    )
    p_net_train_state.params["layers_0"]["bias"] = g_net_0_bias_train_state.apply_fn(
        g_net_0_bias_train_state.params, g0_bias_input
    )
    p_net_train_state.params["layers_2"]["kernel"] = g_net_1_train_state.apply_fn(
        g_net_1_train_state.params, g1_input
    )
    p_net_train_state.params = flax.core.frozen_dict.freeze(p_net_train_state.params)
    return p_net_train_state


if __name__ == "__main__":
    init_rng = jax.random.PRNGKey(SEED)
    g_net_0, g_net_0_bias, g_net_1 = create_g_nets()

    rng, g_net_0_rng = jax.random.split(init_rng)
    input_shape = (1, 20)
    g_net_0_train_state = create_train_state(
        g_net_0, g_net_0_rng, input_shape, LR, MOMENTUM
    )

    rng, g_net_0_bias_rng = jax.random.split(rng)
    input_shape = (1, 10)
    g_net_0_bias_train_state = create_train_state(
        g_net_0_bias, g_net_0_bias_rng, input_shape, LR, MOMENTUM
    )

    rng, g_net_1_rng = jax.random.split(rng)
    input_shape = (1, 20)
    g_net_1_train_state = create_train_state(
        g_net_1, g_net_1_rng, input_shape, LR, MOMENTUM
    )

    rng, p_net_rng = jax.random.split(rng)
    p_net = create_p_model(NUM_CLASSES)
    input_shape = (1, 28 * 28)
    p_net_train_state = create_train_state(p_net, p_net_rng, input_shape, LR, MOMENTUM)

    for _ in range(100):
        p_net_train_state = outer_loop(
            g_net_0_train_state,
            g_net_0_bias_train_state,
            g_net_1_train_state,
            p_net_train_state,
        )
        inner_loop(p_net_train_state)
