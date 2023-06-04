import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import metrics
from flax import linen as nn
from flax import struct
from flax.training import train_state
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


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


@jax.jit
def inner_loop_train_step(state, batch):
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


def create_p_model(num_classes):
    model = nn.Sequential(
        [nn.Dense(800), nn.relu, nn.Dense(num_classes), nn.log_softmax]
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
