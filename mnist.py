# from chatGPT
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from clu import metrics
from flax import struct
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

LR = 1e-3
MOMENTUM = 0.9
NUM_EPOCHS = 10
NUM_CLASSES = 10
BATCH_SIZE = 32


def create_model(num_classes):
    model = nn.Sequential(
        [
            nn.Dense(128),
            nn.relu,
            nn.Dense(128),
            nn.relu,
            nn.Dense(num_classes),
            nn.log_softmax,
        ]
    )
    return model


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([1, 28 * 28]))["params"]
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"].reshape(-1, 784))
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({"params": state.params}, batch["image"].reshape(-1, 784))
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch["label"], loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


def get_datasets(num_epochs, batch_size):
    """Load MNIST train and test datasets into memory."""
    train_ds = tfds.load("mnist", split="train")
    test_ds = tfds.load("mnist", split="test")

    train_ds = train_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255.0,
            "label": sample["label"],
        }
    )
    test_ds = test_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255.0,
            "label": sample["label"],
        }
    )

    train_ds = train_ds.repeat(num_epochs).shuffle(1024)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    test_ds = test_ds.shuffle(1024)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    return train_ds, test_ds


def main():
    num_epochs = 10
    batch_size = 32

    train_ds, test_ds = get_datasets(num_epochs, batch_size)

    num_steps_per_epoch = train_ds.cardinality().numpy() // num_epochs
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    model = create_model(NUM_CLASSES)
    init_rng = jax.random.PRNGKey(0)
    state = create_train_state(model, init_rng, LR, MOMENTUM)

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        state = train_step(state, batch)
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


if __name__ == "__main__":
    main()
