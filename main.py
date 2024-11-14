from typing import Tuple
import pyopencl as cl
import numpy as np
import pickle
import math
import zlib
from concurrent.futures import ThreadPoolExecutor
from numpy import ndarray
import random

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


def load_core(core_element):
    with open(f"./core/{core_element}.txt", "r") as f:
        return cl.Program(ctx, f.read()).build()


def create_array(size, ones=False, zeros=False, random=False, dtype=np.float32):
    if ones:
        return np.ones(size, dtype=dtype)

    if zeros:
        return np.zeros(size, dtype=dtype)

    if random:
        return np.random.rand(size).astype(dtype=dtype)


def relu_weight_init(size, fan_in, dtype=np.float32, random=True):
    if random:
        # He initialization for ReLU activation: np.sqrt(2 / fan_in)
        return np.random.randn(*size).astype(dtype=dtype) * np.sqrt(2.0 / fan_in)
    else:
        return np.zeros(size, dtype=dtype)


def serialize_array(array: np.ndarray) -> str:
    array_bytes = array.tobytes()
    return zlib.compress(array_bytes).hex()


def deserialize_array(data: str) -> np.ndarray:
    byte_array = zlib.decompress(bytearray.fromhex(data))
    return np.frombuffer(byte_array, dtype=np.float32)


def to_buffer(array: np.ndarray) -> cl.Buffer:
    return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=array.astype(np.float32))


def from_buffer(buffer: cl.Buffer, buffer_size: int) -> np.ndarray:
    arr = np.empty(buffer_size, dtype=np.float32)
    cl.enqueue_copy(queue, arr, buffer)
    return arr


class NetworkInput:
    def __init__(self, array_2d: np.ndarray):
        self.array = array_2d.ravel()

        self.buffer = to_buffer(self.array)
        self.width = array_2d.shape[0]
        self.height = array_2d.shape[1] if len(array_2d.shape) != 1 else 1
        self.depth = array_2d.shape[2] if len(array_2d.shape) > 2 else 1

    def to_array(self):
        result = np.empty(self.width * self.height * self.depth, dtype=np.float32)
        cl.enqueue_copy(queue, result, self.buffer)
        return result


class NetworkLayer:
    def __init__(self, buffer, width, height, depth=1):
        self.buffer = buffer
        self.width = width
        self.height = height
        self.depth = depth

    def to_array(self):
        result = np.empty(self.width * self.height * self.depth, dtype=np.float32)
        cl.enqueue_copy(queue, result, self.buffer)
        return result


class Activation:
    ReLU = 1
    Sigmoid = 2


class FeatureMap:
    def __init__(self, input_size: Tuple[int, int], filter_size: Tuple[int, int], colour_channels: int = 1):
        self.input_size = input_size
        self.colour_channels = colour_channels
        self.filter_size = filter_size

        self.weights = to_buffer(create_array(self.__weight_count(), random=True))
        self.biases = to_buffer(create_array(self.__bias_count(), random=True))

        # ADAM
        self.m_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_weights = np.zeros_like(self.weights)
        self.v_biases = np.zeros_like(self.biases)
        self.t = 0

        self.program = load_core("feature_map")

    def __cell_count(self) -> int:
        """ Returns the number of cells in the feature map """
        return math.ceil(self.input_size[0] / self.filter_size[0]) * math.ceil(
            self.input_size[1] / self.filter_size[1])

    def __cell_grid_size(self) -> Tuple[int, int]:
        """ Returns the grid size of the feature map (width, height) """
        return math.ceil(self.input_size[0] / (self.filter_size[0] * self.colour_channels)), math.ceil(
            self.input_size[1] / self.filter_size[1])

    def __weight_count(self) -> int:
        """ Returns the total number of weights in the feature map """
        filter_input_count = self.filter_size[0] * self.filter_size[1] * self.colour_channels
        return filter_input_count

    def __bias_count(self) -> int:
        """ Returns the total number of biases in the feature map"""
        return self.__cell_count()

    def __output_count(self) -> int:
        """ Returns the total number of outputs to the feature map"""
        return self.__cell_count()

    def train_single(self, input_data, target_output: np.ndarray, learning_rate: float):
        input_data = NetworkLayer(to_buffer(input_data), input_data.shape[0], input_data.shape[1], input_data.shape[2])
        output = self.forward_pass(input_data)

        error = output.to_array() - target_output.ravel()

        weight_nudges, bias_nudges, _ = self.backward_pass(input_data, output, NetworkLayer(to_buffer(error), output.width, output.height, output.depth), learning_rate)

        self.apply_gradients(weight_nudges, bias_nudges)
        return sum([abs(err) for err in error]) / (error.shape[0])

    def forward_pass(self, input_data: NetworkLayer) -> NetworkLayer:
        """ Performs the forward pass through the feature map """
        output_buffer = cl.Buffer(ctx, mf.READ_WRITE, size=self.__output_count() * np.dtype(np.float32).itemsize)

        self.program.filter(queue, self.__cell_grid_size(), None,
                            input_data.buffer, self.weights, self.biases, output_buffer,
                            np.int32(self.input_size[0] * self.colour_channels), np.int32(self.input_size[1]),
                            np.int32(self.filter_size[0] * self.colour_channels),
                            np.int32(self.filter_size[1]), np.int32(self.__cell_grid_size()[0]))

        return NetworkLayer(output_buffer, self.__cell_grid_size()[0], self.__cell_grid_size()[1])

    def backward_pass(self, input_values: NetworkLayer, output_values: NetworkLayer, output_error: NetworkLayer,
                      learning_rate: float) -> Tuple[ndarray, ndarray, NetworkInput]:
        backprop_program = load_core("trainer/feature_map")

        input_errors = NetworkInput(create_array(self.input_size[0] * self.input_size[1] * self.colour_channels, zeros=True))

        weights_nudges = to_buffer(create_array(self.__weight_count(), zeros=True))
        weights_nudges_unreduced = to_buffer(create_array(self.input_size, zeros=True))
        biases_nudges = to_buffer(create_array(self.__bias_count(), zeros=True))

        backprop_program.filter(queue, self.__cell_grid_size(), None,
                                input_errors.buffer, self.weights, self.biases, output_error.buffer,
                                input_values.buffer, output_values.buffer, weights_nudges_unreduced, biases_nudges,
                                np.int32(self.input_size[0] * self.colour_channels), np.int32(self.input_size[1]),
                                np.int32(self.filter_size[0] * self.colour_channels), np.int32(self.filter_size[1]),
                                np.int32(self.__cell_grid_size()[0]),
                                np.float32(learning_rate))

        backprop_program.sum_errors(queue, self.__cell_grid_size(), None,
                                    weights_nudges_unreduced, weights_nudges,
                                    np.int32(self.filter_size[0]), np.int32(self.filter_size[1]),
                                    np.int32(self.input_size[0]), np.int32(self.input_size[1]),
                                    np.int32(self.__cell_grid_size()[0]), np.int32(self.__cell_grid_size()[1])
                                    )

        return (from_buffer(weights_nudges, self.__weight_count()),
                from_buffer(biases_nudges, self.__bias_count()),
                input_errors)

    def apply_gradients(self, weight_nudges: np.ndarray, bias_nudges: np.ndarray, beta1: float, beta2: float, epsilon: float) -> None:
        """ Apply our gradients using ADAM """
        # Increment the time step
        self.t += 1

        # Update biased first moment (m)
        self.m_weights = beta1 * self.m_weights + (1 - beta1) * weight_nudges
        self.m_biases = beta1 * self.m_biases + (1 - beta1) * bias_nudges

        # Update biased second moment (v)
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (weight_nudges ** 2)
        self.v_biases = beta2 * self.v_biases + (1 - beta2) * (bias_nudges ** 2)

        # Compute bias-corrected first moment
        m_weights_hat = self.m_weights / (1 - beta1 ** self.t)
        m_biases_hat = self.m_biases / (1 - beta1 ** self.t)

        # Compute bias-corrected second moment
        v_weights_hat = self.v_weights / (1 - beta2 ** self.t)
        v_biases_hat = self.v_biases / (1 - beta2 ** self.t)

        # Get weights from GPU Buffers
        weights = from_buffer(self.weights, self.__weight_count())
        biases = from_buffer(self.biases, self.__bias_count())

        # Update weights and biases
        weights += m_weights_hat / (np.sqrt(v_weights_hat) + epsilon)
        biases += m_biases_hat / (np.sqrt(v_biases_hat) + epsilon)

        # Push weights to GPU Buffers
        self.weights = to_buffer(weights)
        self.biases = to_buffer(biases)

        # Hope it works...

    @staticmethod
    def deserialize(data: str):
        """ Deserializes a feature map """
        input_size, colour_channels, filer_size, weights, biases = data.split(".")
        loaded_layer = FeatureMap(eval(input_size), eval(filer_size), eval(colour_channels))
        loaded_layer.weights = to_buffer(deserialize_array(weights))
        loaded_layer.biases = to_buffer(deserialize_array(biases))

        return loaded_layer

    def serialize(self) -> str:
        """ Serializes the feature map """
        return (f"{self.input_size}."
                f"{self.colour_channels}."
                f"{self.filter_size}."
                f"{serialize_array(from_buffer(self.weights, self.__weight_count()))}."
                f"{serialize_array(from_buffer(self.biases, self.__bias_count()))}")


class FullyPopulated:
    def __init__(self, input_size: int, output_size: int, activation: int):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.weights = to_buffer(create_array(self.__weight_count(), random=True))
        self.biases = to_buffer(create_array(self.__bias_count(), random=True))

        # ADAM
        self.m_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_weights = np.zeros_like(self.weights)
        self.v_biases = np.zeros_like(self.biases)
        self.t = 0

        self.program = load_core("fully_connected")

    def __weight_count(self) -> int:
        """ Returns the total number of weights in the network layer """
        return self.input_size * self.output_size

    def __bias_count(self) -> int:
        """ Returns the total number of biases in the network layer"""
        return int(self.output_size)

    def forward_pass(self, input_data: NetworkLayer) -> NetworkLayer:
        """ Performs the forward pass of the network layer """
        output_buffer = cl.Buffer(ctx, mf.READ_WRITE, size=self.output_size * np.dtype(np.float32).itemsize)

        self.program.forward_pass(queue, (self.output_size,), None,
                                  input_data.buffer, self.weights, self.biases, output_buffer,
                                  np.int32(input_data.width), np.int32(self.activation))

        return NetworkLayer(output_buffer, self.output_size, 1)

    def backward_pass(self, input_values: NetworkLayer, output_values: NetworkLayer, output_error: NetworkLayer,
                      learning_rate: float):
        backprop_program = load_core("trainer/fully_populated")

        input_errors = NetworkInput(create_array(self.input_size, zeros=True))
        pre_summed_errors = to_buffer(create_array(self.__weight_count(), zeros=True))
        weights_nudges = to_buffer(create_array(self.__weight_count(), zeros=True))
        biases_nudges = to_buffer(create_array(self.__bias_count(), zeros=True))

        backprop_program.backward(queue, (self.input_size, self.output_size), None,
                                  pre_summed_errors, self.weights, self.biases, output_error.buffer,
                                  input_values.buffer, output_values.buffer, weights_nudges, biases_nudges,
                                  np.int32(self.input_size), np.int32(self.output_size),
                                  np.int32(self.activation), np.float32(learning_rate))

        backprop_program.sum_input_errors(queue, (self.input_size,), None,
                                          pre_summed_errors, input_errors.buffer,
                                          np.int32(self.input_size), np.int32(self.output_size)
                                          )

        return (from_buffer(weights_nudges, self.__weight_count()),
                from_buffer(biases_nudges, self.__bias_count()),
                input_errors)

    def apply_gradients(self, weight_nudges: np.ndarray, bias_nudges: np.ndarray, beta1: float, beta2: float, epsilon: float) -> None:
        """ Apply our gradients using ADAM """
        # Increment the time step
        self.t += 1

        # Update biased first moment (m)
        self.m_weights = beta1 * self.m_weights + (1 - beta1) * weight_nudges
        self.m_biases = beta1 * self.m_biases + (1 - beta1) * bias_nudges

        # Update biased second moment (v)
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (weight_nudges ** 2)
        self.v_biases = beta2 * self.v_biases + (1 - beta2) * (bias_nudges ** 2)

        # Compute bias-corrected first moment
        m_weights_hat = self.m_weights / (1 - beta1 ** self.t)
        m_biases_hat = self.m_biases / (1 - beta1 ** self.t)

        # Compute bias-corrected second moment
        v_weights_hat = self.v_weights / (1 - beta2 ** self.t)
        v_biases_hat = self.v_biases / (1 - beta2 ** self.t)

        # Get weights from GPU Buffers
        weights = from_buffer(self.weights, self.__weight_count())
        biases = from_buffer(self.biases, self.__bias_count())

        # Update weights and biases
        weights += m_weights_hat / (np.sqrt(v_weights_hat) + epsilon)
        biases += m_biases_hat / (np.sqrt(v_biases_hat) + epsilon)

        # Push weights to GPU Buffers
        self.weights = to_buffer(weights)
        self.biases = to_buffer(biases)

        # Hope it works...

    @staticmethod
    def deserialize(data: str):
        """ Deserializes a layer """
        input_size, output_size, activation, weights, biases = data.split(".")
        loaded_layer = FullyPopulated(int(input_size), int(output_size), int(activation))
        loaded_layer.weights = to_buffer(deserialize_array(weights))
        loaded_layer.biases = to_buffer(deserialize_array(biases))

        return loaded_layer

    def serialize(self) -> str:
        """ Serializes the layer """
        return (f"{self.input_size}."
                f"{self.output_size}."
                f"{self.activation}."
                f"{serialize_array(from_buffer(self.weights, self.__weight_count()))}."
                f"{serialize_array(from_buffer(self.biases, self.__bias_count()))}")


class FilterLayer:
    def __init__(self, mul: float, add: float):
        self.mul = mul
        self.add = add

        self.program = load_core("math_filter")

    def forward_pass(self, input_layer: NetworkLayer) -> NetworkInput:
        """ Performs the forward pass of the network layer """

        output_layer = NetworkInput(create_array((input_layer.width, input_layer.height, input_layer.depth), zeros=True))

        self.program.forward_pass(queue, (input_layer.width * input_layer.height * input_layer.depth,), None,
                                  input_layer.buffer, output_layer.buffer,
                                  np.float32(self.mul), np.float32(self.add))

        return output_layer

    def backward_pass(self, input_values: NetworkLayer, output_values: NetworkLayer, output_error: NetworkLayer,
                      learning_rate: float):
        backprop_program = load_core("trainer/math_filter")

        backprop_program.backward_pass(queue, (output_error.width * output_error.height,), None,
                                       output_error.buffer, np.float32(self.mul), np.float32(self.add))

        return None, None, output_error  # todo fix me

    def apply_gradients(self, weight_nudges: np.ndarray, bias_nudges: np.ndarray, beta1, beta2, epsilon) -> None:
        return

    @staticmethod
    def deserialize(data: str):
        """ Deserializes a layer """
        mul, add = data.split("..")
        loaded_layer = FilterLayer(float(mul), float(add))

        return loaded_layer

    def serialize(self) -> str:
        """ Serializes the layer """
        return (f"{self.mul}.."
                f"{self.add}")


class Network:
    def __init__(self, net_layout: list, device):
        self.layout = net_layout
        self.device = device

    def forward_pass(self, inputs: np.ndarray, save_layer_data: bool = False):
        """ Forward pass through the Network """

        layer_data: NetworkInput = NetworkInput(inputs.astype(np.float32))
        node_values = []

        if save_layer_data:
            node_values.append(layer_data)

        for layer in self.layout:
            layer_data = layer.forward_pass(layer_data)

            if save_layer_data:
                node_values.append(layer_data)

        if save_layer_data:
            return layer_data.to_array(), node_values

        return layer_data.to_array()

    def async_forward_pass(self, inputs: np.ndarray, save_layer_data: bool = False):
        with ThreadPoolExecutor() as executor:
            return executor.submit(self.forward_pass, inputs, save_layer_data)

    @staticmethod
    def __clamp(data, value):
        return np.clip(data, -value, value)

    def _backpropagation(self, inputs: np.ndarray, target: np.ndarray, learning_rate: float):
        """ Backpropagation through the Network, Returns average abs error """
        results, node_values = self.forward_pass(inputs, save_layer_data=True)
        error = NetworkInput(self.__clamp(target - results, 1).astype(np.float32))

        #print(results, error.to_array())

        next_error = error

        w_nudge = []
        b_nudge = []

        for i in range(len(self.layout)):
            layer = self.layout[len(self.layout) - i - 1]

            weight_nudges, bias_nudges, next_error = layer.backward_pass(node_values[i - 1], node_values[i],
                                                                         next_error, learning_rate)

            w_nudge.append(weight_nudges)
            b_nudge.append(bias_nudges)

        return w_nudge, b_nudge, sum([abs(err) for err in error.to_array()]) / (error.width * error.height)

    def backpropagation(self, training_data: list, learning_rate: float, beta1=0.9, beta2=0.999, epsilon=1e-7) -> float:
        """ Performs multiple backpropagation on all training data, then applies the nudges, returns abs average error """
        total_error = 0
        weight_nudges = []
        bias_nudges = []

        for (inputs, target) in training_data:
            w, b, err = self._backpropagation(inputs, target, learning_rate)

            weight_nudges.append(w)
            bias_nudges.append(b)
            total_error += err

        weight_nudges = self.sum_nudges(weight_nudges, len(training_data))
        bias_nudges = self.sum_nudges(bias_nudges, len(training_data))

        for i in range(len(self.layout)):
            j = len(self.layout) - i - 1
            self.layout[i].apply_gradients(
                np.array(weight_nudges[j], dtype=np.float32),
                np.array(bias_nudges[j], dtype=np.float32),
                beta1, beta2, epsilon)

        return total_error / len(training_data)

    @staticmethod
    def sum_nudges(nudges, count):
        total = nudges[0]

        for i in range(1, len(nudges)):
            for j, x in enumerate(total):
                if nudges[i][j] is not None:
                    total[j] += nudges[i][j] / count

        return total

    def save(self, path: str) -> None:
        """ Saves the network to a given path """

        with open(path, "w") as f:
            f.write("\n".join([layer.serialize() + ":" + layer.__class__.__name__ for layer in self.layout]))

    @staticmethod
    def load(path: str, device=None):
        """ Loads the network from a given path """
        with open(path, "r") as f:
            return Network(
                [globals()[layer.split(":")[1]].deserialize(layer.split(":")[0]) for layer in f.read().split("\n")], device)
