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
    model = nn.Sequential([nn.Dense(30), nn.relu, nn.Dense(1)])
    return model


def generate_gray_code(n):
    if n <= 0:
        return []

    gray_code = ["0", "1"]

    i = 2
    while i <= n:
        reflected_gray_code = gray_code[::-1]
        gray_code = ["0" + code for code in gray_code] + [
            "1" + code for code in reflected_gray_code
        ]
        i += 1

    max_length = len(gray_code[-1])
    gray_code = [code.zfill(max_length) for code in gray_code]

    return gray_code


def create_one_hot_vectors(size: int) -> np.ndarray:
    integers = np.arange(size)
    one_hot_vectors = np.eye(size, dtype=int)[integers]
    return one_hot_vectors


def generate_input_tags() -> np.ndarray:
    code = generate_gray_code(5)
    input_arr = []
    for x in range(28):
        for y in range(28):
            input_arr.append([list(str(code[x]) + str(code[y]))])
    input_arr = np.array(input_arr)
    return input_arr


def generate_binary_tags(bits: int = 10) -> List[int]:
    codes = []
    for i in range(2 ** bits):
        code = bin(i)[2:].zfill(bits)
        codes.append(code)
    return codes


def generate_hidden_layer_tags() -> np.ndarray:
    code = generate_binary_tags(10)
    input_arr = []
    for x in range(800):
        input_arr.append(list(code[x]))
    input_arr = np.array(input_arr)
    return input_arr


def generate_output_layer_tags() -> np.ndarray:
    input_arr = [i for i in range(10)]
    input_arr = np.eye(10, dtype=int)[input_arr]
    return input_arr


def weight_matrix_tag_combinations(n_tags, n_plus_one_tags) -> np.ndarray:
    combinations = []
    for i in range(len(n_tags)):
        for j in range(len(n_plus_one_tags)):
            combinations.append(np.concatenate((n_tags[i], n_plus_one_tags[j])))
    return np.array(combinations).astype(np.float32)


def get_g_net_inputs() -> Tuple[jax.Array]:
    input_tags = generate_input_tags().squeeze(1)
    hidden_layer_tags = generate_hidden_layer_tags()
    output_layer_tags = generate_output_layer_tags()
    input_to_g0 = weight_matrix_tag_combinations(input_tags, hidden_layer_tags)
    input_to_g1 = weight_matrix_tag_combinations(hidden_layer_tags, output_layer_tags)
    input_to_g0_bias = hidden_layer_tags.copy().astype(np.float32)
    # input_arr_for_g0 = generate_xy_inputs(784, 800)
    # input_arr_bias_for_g0 = generate_xy_inputs(1, 800)
    # input_arr_for_g1 = generate_xy_inputs(800, 10)
    # input_arr_for_g0 = generate_input_arr_for_g0()
    # input_arr_bias_for_g0 = generate_input_arr_for_g0_bias()
    # input_arr_for_g1 = generate_input_arr_for_g1()
    return (
        jnp.array(input_to_g0),
        jnp.array(input_to_g0_bias),
        jnp.array(input_to_g1),
    )


def create_g_nets() -> Tuple[nn.Sequential]:
    return create_g_models(), create_g_models(), create_g_models()


@struct.dataclass
class GInitialiser:
    g_network_train_state: train_state.TrainState
    g_network_inputs: jax.Array

    def __call__(self, rng, dtype=jnp.float32):
        initialise_params = self.g_network_train_state.apply_fn(
            {"params": self.g_network_train_state.params}, self.g_network_inputs
        )
        # rng, noise_rng = jax.random.split(rng)
        # noise = (
        #     jax.random.normal(noise_rng, initialise_params.shape, dtype=dtype) * 0.01
        # )
        # params = jax.lax.stop_gradient(initialise_params + noise)

        return initialise_params
