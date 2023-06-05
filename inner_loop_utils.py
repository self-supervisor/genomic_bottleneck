import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import metrics
from flax import linen as nn
from flax import struct
from flax.training import train_state
from outer_loop_utils import GInitialiser, get_g_net_inputs
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


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, input_shape, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones(input_shape))["params"]
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


def create_p_net_train_state(module, rng, learning_rate, momentum):
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(apply_fn=module.apply, params=module.param, tx=tx)


class PModel(nn.Module):
    g0_train_state: TrainState
    g0_bias_train_state: TrainState
    g1_train_state: TrainState

    def setup(self):
        g0_input, g0_bias_input, g1_input = get_g_net_inputs()
        self.weight_0 = (
            self.param(
                "w0",
                GInitialiser(
                    g_network_train_state=self.g0_train_state,
                    g_network_inputs=g0_input,
                ),
            )
            .reshape(-1)
            .reshape(784, 800)
        )
        self.bias_0 = self.param(
            "b0",
            GInitialiser(
                g_network_train_state=self.g0_bias_train_state,
                g_network_inputs=g0_bias_input,
            ),
        ).reshape(-1)
        self.weight_1 = (
            self.param(
                "w1",
                GInitialiser(
                    g_network_train_state=self.g1_train_state,
                    g_network_inputs=g1_input,
                ),
            )
            .reshape(-1)
            .reshape(800, 10)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.dot(x, self.weight_0) + self.bias_0
        x = nn.relu(x)
        x = jnp.dot(x, self.weight_1)
        return x
