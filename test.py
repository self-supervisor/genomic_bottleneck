import jax
import pytest

print("devices", jax.devices())
import flax.linen as nn
import jax.numpy as jnp

from mnist import create_model, load_mnist_dataset


@pytest.fixture
def mnist_dataset():
    return load_mnist_dataset()


@pytest.fixture
def model():
    num_classes = 10
    model = create_model(num_classes)
    return model


def test_load_mnist_dataset(mnist_dataset):
    x_train, y_train, x_test, y_test = mnist_dataset

    assert len(x_train) == len(y_train) == 60000
    assert len(x_test) == len(y_test) == 10000

    # Check the shapes of the input images and labels
    assert x_train[0].shape == (784,)
    assert y_train[0].shape == (10,)
    assert x_test[0].shape == (784,)
    assert y_test[0].shape == (10,)

    # Check the range of pixel values
    assert x_train[0].min() >= 0
    assert x_train[0].max() <= 1
    assert x_test[0].min() >= 0
    assert x_test[0].max() <= 1

    # Check the one-hot encoding of labels
    assert y_train[0].argmax() == y_train[0].nonzero()[0][0]
    assert y_test[0].argmax() == y_test[0].nonzero()[0][0]

    # check data is jnp array
    assert isinstance(x_train[0], jnp.ndarray)
    assert isinstance(y_train[0], jnp.ndarray)
    assert isinstance(x_test[0], jnp.ndarray)
    assert isinstance(y_test[0], jnp.ndarray)


def test_create_model(model, mnist_dataset):
    # Check the type of the model
    assert isinstance(model, nn.Sequential)

    model.init(jax.random.PRNGKey(0), jnp.zeros([1, 28 * 28]))

    unit_test_batch = mnist_dataset[0][:16]
    # check input shape matches
    params = model.init(jax.random.PRNGKey(0), unit_test_batch)["params"]
    pred = model.apply({"params": params}, unit_test_batch)

    # check output shape
    assert pred.shape == (16, 10)
