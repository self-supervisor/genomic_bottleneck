import numpy as np


def gray_code(num_bits: int) -> list:
    """
    Generate Gray code sequence of given number of bits.
    """
    sequence = []
    for i in range(2**num_bits):
        sequence.append(i ^ (i >> 1))
    return sequence


def binary_representation(num: int, num_bits: int) -> str:
    """
    Convert decimal number to binary representation with leading zeros.
    """
    return bin(num)[2:].zfill(num_bits)


def generate_coordinates_array(array_size: int) -> np.ndarray:
    """
    Generate flattened array of shape (array_size, 10) representing X, Y coordinate pairs.
    """
    num_bits = 5
    gray_sequence = gray_code(num_bits)

    coordinates_array = np.zeros((array_size, 10), dtype=int)

    for i in range(array_size):
        x_gray = gray_sequence[i % 28]
        y_gray = gray_sequence[i // 28]

        x_binary = binary_representation(x_gray, num_bits)
        y_binary = binary_representation(y_gray, num_bits)

        coordinates = x_binary + y_binary
        coordinates_array[i] = [int(bit) for bit in coordinates]

    return coordinates_array


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


def generate_input_arr_for_g0() -> np.ndarray:
    input_neuron_tags = generate_coordinates_array(784)
    hidden_layer_neuron_tags = generate_coordinates_array(800)
    binary_connection_representations = combine_strings(
        input_neuron_tags, hidden_layer_neuron_tags
    )
    return binary_connection_representations
