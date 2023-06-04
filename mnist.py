# from chatGPT
from typing import List

import jax
from tensorflow.python.ops.numpy_ops import np_config

from inner_loop_utils import (
    compute_metrics,
    create_p_model,
    create_train_state,
    get_datasets,
    inner_loop_train_step,
)

np_config.enable_numpy_behavior()

LR = 1e-3
MOMENTUM = 0.9
NUM_EPOCHS = 10
NUM_CLASSES = 10
BATCH_SIZE = 32


def inner_loop() -> jax.Array:
    train_ds, test_ds = get_datasets(NUM_EPOCHS, BATCH_SIZE)

    num_steps_per_epoch = train_ds.cardinality().numpy() // NUM_EPOCHS
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    model = create_p_model(NUM_CLASSES)
    init_rng = jax.random.PRNGKey(0)
    state = create_train_state(model, init_rng, LR, MOMENTUM)

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
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


def outer_loop() -> List[jax.Array]:
    pass


if __name__ == "__main__":
    inner_loop()
