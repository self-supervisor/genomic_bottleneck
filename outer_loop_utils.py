from typing import List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state


@jax.jit
def g_net_train_step(state, inputs, outputs):
    def loss_fn(params):
        preds = state.apply_fn({"params": params}, inputs)
        loss = ((preds - outputs) ** 2).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


def create_g_models() -> nn.Sequential:
    model = nn.Sequential([nn.Dense(32), nn.relu, nn.Dense(32), nn.relu, nn.Dense(1)])
    return model


def gray_code(num_bits: int) -> List:
    """
    Generate 2D Gray code sequence of given number of bits.
    """
    sequence = []
    for i in range(2 ** num_bits):
        sequence.append((i ^ (i >> 1)) % (2 ** num_bits))
    return sequence


def binary_representation(num: int, num_bits: int) -> str:
    """
    Convert decimal number to binary representation with leading zeros.
    """
    return bin(num)[2:].zfill(num_bits)


def generate_coordinates_array(array_size: int) -> np.ndarray:
    """
    Generate flattened array of shape (array_size, 10) representing X, Y coordinate pairs using 2D Gray code.
    """
    num_bits = 5
    gray_sequence = gray_code(num_bits)

    coordinates_array = np.zeros((array_size, 10), dtype=int)

    for i in range(array_size):
        x_gray = gray_sequence[i % (2 ** num_bits)]
        y_gray = gray_sequence[i // (2 ** num_bits)]

        x_binary = binary_representation(x_gray, num_bits)
        y_binary = binary_representation(y_gray, num_bits)

        coordinates = x_binary + y_binary
        coordinates_array[i] = [int(bit) for bit in coordinates]

    return coordinates_array


# def gray_code(num_bits: int) -> List:
#     """
#     Generate Gray code sequence of given number of bits.
#     """
#     sequence = []
#     for i in range(2 ** num_bits):
#         sequence.append(i ^ (i >> 1))
#     return sequence


# def binary_representation(num: int, num_bits: int) -> str:
#     """
#     Convert decimal number to binary representation with leading zeros.
#     """
#     return bin(num)[2:].zfill(num_bits)


# def generate_coordinates_array(array_size: int) -> np.ndarray:
#     """
#     Generate flattened array of shape (array_size, 10) representing X, Y coordinate pairs.
#     """
#     num_bits = 5
#     gray_sequence = gray_code(num_bits)

#     coordinates_array = np.zeros((array_size, 10), dtype=int)

#     for i in range(array_size):
#         x_gray = gray_sequence[i % 28]
#         y_gray = gray_sequence[i // 28]

#         x_binary = binary_representation(x_gray, num_bits)
#         y_binary = binary_representation(y_gray, num_bits)

#         coordinates = x_binary + y_binary
#         coordinates_array[i] = [int(bit) for bit in coordinates]

#     return coordinates_array


def combine_strings(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Combine strings from two arrays to create all possible combinations.
    """
    # Create meshgrid of indices
    idx1, idx2 = np.meshgrid(
        np.arange(array1.shape[0]), np.arange(array2.shape[0]), indexing="ij"
    )

    # Concatenate the strings from both arrays
    combined_strings = np.concatenate(
        (array1[idx1.flatten()], array2[idx2.flatten()]), axis=1
    )

    return combined_strings


def create_one_hot_vectors(size: int) -> np.ndarray:
    integers = np.arange(size)
    one_hot_vectors = np.eye(size, dtype=int)[integers]
    return one_hot_vectors


def generate_input_arr_for_g0() -> np.ndarray:
    input_neuron_tags = generate_coordinates_array(784)
    hidden_layer_neuron_tags = generate_coordinates_array(800)
    binary_connection_representations = combine_strings(
        input_neuron_tags, hidden_layer_neuron_tags
    )
    return binary_connection_representations


def generate_input_arr_for_g0_bias() -> np.ndarray:
    binary_connection_representations = generate_coordinates_array(800)
    return binary_connection_representations


def generate_input_arr_for_g1() -> np.ndarray:
    input_neuron_tags = generate_coordinates_array(800)
    output_neuron_tags = create_one_hot_vectors(size=10)
    binary_connection_representations = combine_strings(
        input_neuron_tags, output_neuron_tags
    )
    return binary_connection_representations


def generate_xy_inputs(max_x, max_y) -> np.ndarray:
    input_arr = []
    for x in range(max_x):
        for y in range(max_y):
            input_arr.append([x, y])
    input_arr = np.array(input_arr) / max([max_x, max_y])

    return input_arr


def get_g_net_inputs() -> Tuple[jax.Array]:
    input_arr_for_g0 = generate_xy_inputs(784, 800)
    input_arr_bias_for_g0 = generate_xy_inputs(1, 800)
    input_arr_for_g1 = generate_xy_inputs(800, 10)
    return (
        jnp.array(input_arr_for_g0),
        jnp.array(input_arr_bias_for_g0),
        jnp.array(input_arr_for_g1),
    )


def create_g_nets() -> Tuple[nn.Sequential]:
    return create_g_models(), create_g_models(), create_g_models()


@struct.dataclass
class GInitialiser:
    g_network_train_state: train_state.TrainState
    g_network_inputs: jax.Array

    def __call__(self, rng, dtype=jnp.float32):
        return self.g_network_train_state.apply_fn(
            {"params": self.g_network_train_state.params}, self.g_network_inputs
        )
