# from chatGPT
from typing import List

import wandb
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow.python.ops.numpy_ops import np_config

from inner_loop_utils import (
    PModel,
    TrainState,
    compute_metrics,
    create_p_net_train_state,
    create_train_state,
    get_datasets,
    inner_loop_train_step,
    Metrics,
)
from outer_loop_utils import (
    create_g_nets,
    g_net_train_step,
    get_g_net_inputs,
    # generate_input_arr_for_g0,
    # generate_input_arr_for_g0_bias,
    # generate_input_arr_for_g1,
)

np_config.enable_numpy_behavior()

LR = 1e-3
NUM_EPOCHS = 10
NUM_CLASSES = 10
BATCH_SIZE = 100
SEED = 0


def test(rng, g0_train_state, g0_bias_train_state, g1_train_state, epoch):
    train_ds, test_ds = get_datasets(1, 1000)
    model = PModel(g0_train_state, g0_bias_train_state, g1_train_state, epoch)
    params = model.init(rng, jnp.ones([1, 28 * 28]))
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(LR),
        rng=rng,
        num_classes=NUM_CLASSES,
    )
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    test_state = state
    for test_batch in test_ds.as_numpy_iterator():
        test_state = compute_metrics(state=test_state, batch=test_batch)

    for metric, value in test_state.metrics.compute().items():
        metrics_history[f"test_{metric}"].append(value)

    print(
        f"loss: {metrics_history['test_loss'][-1]}, "
        f"accuracy: {metrics_history['test_accuracy'][-1] * 100}"
    )
    wandb.log(
        {
            "initial_p_net_test_loss": np.array(metrics_history["test_loss"][-1]),
            "initial_p_net_test_accuracy": np.array(
                metrics_history["test_accuracy"][-1] * 100
            ),
        }
    )


def inner_loop(
    rng, g0_train_state, g0_bias_train_state, g1_train_state, epoch
) -> jax.Array:
    train_ds, test_ds = get_datasets(NUM_EPOCHS, BATCH_SIZE)

    num_steps_per_epoch = train_ds.cardinality().numpy() // NUM_EPOCHS
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    model = PModel(g0_train_state, g0_bias_train_state, g1_train_state, epoch)
    tx = optax.adam(LR)
    params = model.init(rng, jnp.ones([1, 28 * 28]))

    state = TrainState.create(
        apply_fn=model.apply, params=params["params"], tx=tx, metrics=Metrics.empty()
    )
    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        state = inner_loop_train_step(state, batch)
        state = compute_metrics(state=state, batch=batch)
        if (step + 1) % num_steps_per_epoch == 0:
            print("step", step)
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
            wandb.log(
                {
                    "p_net_train_loss": np.array(metrics_history["train_loss"][-1]),
                    "p_net_train_accuracy": np.array(
                        metrics_history["train_accuracy"][-1] * 100
                    ),
                    "p_net_test_loss": np.array(metrics_history["test_loss"][-1]),
                    "p_net_test_accuracy": np.array(
                        metrics_history["test_accuracy"][-1] * 100
                    ),
                }
            )
    return state


def outer_loop(
    rng,
    g_net_0_train_state,
    g_net_0_bias_train_state,
    g_net_1_train_state,
    p_net_train_state,
) -> List[jax.Array]:
    g_net_0, g_net_0_bias, g_net_1 = create_g_nets()
    g0_input, g0_bias_input, g1_input = get_g_net_inputs()
    w0 = p_net_train_state.params["w0"]
    w0_labels = w0.copy().reshape(-1)
    w0_bias = p_net_train_state.params["b0"]
    w0_bias_labels = w0_bias.copy().reshape(-1)
    w1 = p_net_train_state.params["w1"]
    w1_labels = w1.copy().reshape(-1)

    # w0_labels = w0_labels[:32]
    # g0_input = g0_input[:32]
    # w0_bias_labels = w0_bias_labels[:32]
    # g0_bias_input = g0_bias_input[:32]
    # w1_labels = w1_labels[:32]
    # g1_input = g1_input[:32]

    for epoch in range(NUM_EPOCHS // 10):
        rng, shuffle_rng = jax.random.split(rng)

        permutation = jax.random.permutation(shuffle_rng, len(w0_labels))
        w0_labels = jnp.take(w0_labels, permutation, axis=0)
        g0_input = jnp.take(g0_input, permutation, axis=0)

        permutation = jax.random.permutation(shuffle_rng, len(w0_bias_labels))
        w0_bias_labels = jnp.take(w0_bias_labels, permutation, axis=0)
        g0_bias_input = jnp.take(g0_bias_input, permutation, axis=0)

        permutation = jax.random.permutation(shuffle_rng, len(w1_labels))
        w1_labels = jnp.take(w1_labels, permutation, axis=0)
        g1_input = jnp.take(g1_input, permutation, axis=0)
        for batch_i in range(len(g0_input) // (BATCH_SIZE * 10) // 5):
            g_net_0_train_state = g_net_train_step(
                g_net_0_train_state,
                g0_input[
                    batch_i * (BATCH_SIZE * 10) : (batch_i + 1) * (BATCH_SIZE * 10)
                ],
                w0_labels[
                    batch_i * (BATCH_SIZE * 10) : (batch_i + 1) * (BATCH_SIZE * 10)
                ],
            )
        for batch_i in range(len(g0_bias_input) // BATCH_SIZE):
            g_net_0_bias_train_state = g_net_train_step(
                g_net_0_bias_train_state,
                g0_bias_input[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE],
                w0_bias_labels[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE],
            )

        for batch_i in range(len(g1_input) // BATCH_SIZE):
            g_net_1_train_state = g_net_train_step(
                g_net_1_train_state,
                g1_input[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE],
                w1_labels[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE],
            )
        random_batch = np.random.randint(0, len(g0_input) // (BATCH_SIZE))
        pred_0 = g_net_0_train_state.apply_fn(
            {"params": g_net_0_train_state.params},
            g0_input[random_batch * BATCH_SIZE : (random_batch + 1) * BATCH_SIZE],
        )
        loss_0 = np.array(
            jnp.mean(
                (
                    pred_0
                    - w0_labels[
                        random_batch * BATCH_SIZE : (random_batch + 1) * BATCH_SIZE
                    ]
                )
                ** 2
            )
        )
        random_batch = np.random.randint(0, len(g0_bias_input) // (BATCH_SIZE))
        pred_0_bias = g_net_0_bias_train_state.apply_fn(
            {"params": g_net_0_bias_train_state.params},
            g0_bias_input[random_batch * BATCH_SIZE : (random_batch + 1) * BATCH_SIZE],
        )
        loss_0_bias = np.array(
            jnp.mean(
                (
                    pred_0_bias
                    - w0_bias_labels[
                        random_batch * BATCH_SIZE : (random_batch + 1) * BATCH_SIZE
                    ]
                )
                ** 2
            )
        )
        random_batch = np.random.randint(0, len(g1_input) // (BATCH_SIZE))
        pred_1 = g_net_1_train_state.apply_fn(
            {"params": g_net_1_train_state.params},
            g1_input[random_batch * BATCH_SIZE : (random_batch + 1) * BATCH_SIZE * 100],
        )
        loss_1 = np.array(
            jnp.mean(
                (
                    pred_1
                    - w1_labels[
                        random_batch * BATCH_SIZE : (random_batch + 1) * BATCH_SIZE
                    ]
                )
                ** 2
            )
        )
        wandb.log(
            {
                "g_net_0_loss": np.array(loss_0),
                "g_net_0_bias_loss": np.array(loss_0_bias),
                "g_net_1_loss": np.array(loss_1),
            }
        )

    return rng, g_net_0_train_state, g_net_0_bias_train_state, g_net_1_train_state


if __name__ == "__main__":
    wandb.init(project="genomic_bottleneck")
    init_rng = jax.random.PRNGKey(SEED)
    g_net_0, g_net_0_bias, g_net_1 = create_g_nets()

    rng, g_net_0_rng = jax.random.split(init_rng)
    input_shape = (1, 20)
    g_net_0_train_state = create_train_state(g_net_0, g_net_0_rng, input_shape, LR)

    rng, g_net_0_bias_rng = jax.random.split(rng)
    input_shape = (1, 10)
    g_net_0_bias_train_state = create_train_state(
        g_net_0_bias, g_net_0_bias_rng, input_shape, LR
    )

    rng, g_net_1_rng = jax.random.split(rng)
    input_shape = (1, 20)
    g_net_1_train_state = create_train_state(g_net_1, g_net_1_rng, input_shape, LR)

    for epoch in range(100):
        rng, inner_rng = jax.random.split(rng)
        test(
            rng,
            g_net_0_train_state,
            g_net_0_bias_train_state,
            g_net_1_train_state,
            epoch,
        )
        p_net_train_state = inner_loop(
            rng,
            g_net_0_train_state,
            g_net_0_bias_train_state,
            g_net_1_train_state,
            epoch,
        )
        (
            rng,
            g_net_0_train_state,
            g_net_0_bias_train_state,
            g_net_1_train_state,
        ) = outer_loop(
            rng,
            g_net_0_train_state,
            g_net_0_bias_train_state,
            g_net_1_train_state,
            p_net_train_state,
        )
