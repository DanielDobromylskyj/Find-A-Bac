import pyopencl as cl
import math


def multiply_list(x):
    y = 1
    for z in x:
        y *= z
    return y


def approximate_network_memory_usage(net):  # what a mess of a function
    all_totals = []
    for layer in net.layout:
        total_float32s = 0

        if type(layer.weights) in [tuple, list]:
            if hasattr(layer.weights[0], "size"):
                total_float32s += layer.weights[0].size
                total_float32s += layer.biases[0].size
            else:
                total_float32s += multiply_list(
                    layer.weights[0].get_shape()
                )

                total_float32s += multiply_list(
                    layer.biases[0].get_shape()
                )
        else:
            if hasattr(layer.weights, "size"):
                total_float32s += layer.weights.size
                total_float32s += layer.biases.size
            else:
                total_float32s += multiply_list(
                    layer.weights.get_shape()
                )

                total_float32s += multiply_list(
                    layer.biases.get_shape()
                )

        if type(layer.input_size) in [tuple, list]:
            total_float32s += multiply_list(layer.input_size) * 2
        else:
            total_float32s += layer.input_size * 2

        all_totals.append(total_float32s)

    return max(all_totals) * 4


def approximate_network_core_usage(net):
    all_totals = []
    for layer in net.layout:
        if hasattr(layer, "filter_size"):
            all_totals.append(
                math.ceil(layer.input_size[0] / layer.filter_size[0]) * math.ceil(
                    layer.input_size[1] / layer.filter_size[1])
            )

        else:
            all_totals.append(
                layer.input_size * layer.output_size
            )

    return max(all_totals)


def get_gpus():
    gpus = []

    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        for device in devices:
            gpus.append(GPU(device))

    return gpus


class GPU:
    def __init__(self, device):
        self.device = device
        self.network = None

        self.network_core_usage = None
        self.network_mem_usage = None

    def set_network(self, net):
        self.network = net

        self.network_core_usage = approximate_network_core_usage(net)
        self.network_mem_usage = approximate_network_memory_usage(net)

    def get_name(self):
        return self.device.name

    def get_max_work_size(self):
        return self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

    def get_total_memory(self):
        return self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE)

    def get_max_concurrent_networks(self):
        mem_total = self.get_total_memory() // self.network_mem_usage
        core_total = max(self.get_max_work_size() // self.network_core_usage, 1)

        return min(mem_total, core_total)


