import os
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib

OPENSLIDE_PATH = r'\openslide-win64\bin'
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(os.getcwd() + OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import main

with open("trainingPoints.txt", "r") as f:
    unprocessedTrainingData = eval(f.read())

trainingData = []
for segment in unprocessedTrainingData:
    slide = openslide.open_slide(segment["tiffPath"])
    for point in segment["points"]:
        img = slide.read_region(point[0], 0, (100, 100))
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)

        trainingData.append([
            arr / 255,
            [point[1][0], 0 if point[1][0] == 1 else 1]
        ])


def display_network(net, layer_data, error_data, fig=None, axes=None):
    if fig is None or axes is None:
        # Create figure and axes only if they are not passed in (i.e., on first call)
        fig, axes = plt.subplots(2, len(layer_data) + 1, figsize=(10, 5))

    # Clear all axes for the new plot
    for ax in axes:
        for row in ax:
            row.clear()

    for i, layer in enumerate(layer_data):
        if layer.depth == 1:
            data = layer.to_array().reshape((layer.height, layer.width, layer.depth))

            axes[0][i].imshow(data, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
            axes[0][i].set_title(f'Layer {i}')
        else:
            data = layer.to_array().reshape((layer.height, layer.width, layer.depth))

            axes[0][i].imshow(data, interpolation='nearest')
            axes[0][i].set_title(f'Layer {i}')

    for i, net_layer in enumerate(net.layout):
        i += 1
        if isinstance(net_layer, main.FilterLayer):
            continue

        if isinstance(net_layer, main.FeatureMap):
            size = net_layer.filter_size[0] * net_layer.filter_size[1] * net_layer.colour_channels
            data = main.from_buffer(net_layer.weights, size).reshape(
                (net_layer.filter_size[1], net_layer.filter_size[0] * net_layer.colour_channels))

            axes[1][i].imshow(data, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
            axes[1][i].set_title(f'Layer {i}')

        if isinstance(net_layer, main.FullyPopulated):
            size = net_layer.input_size * net_layer.output_size
            data = main.from_buffer(net_layer.weights, size).reshape(
                (net_layer.input_size, net_layer.output_size))

            axes[1][i].imshow(data, cmap='viridis', interpolation='nearest')
            axes[1][i].set_title(f'Layer {i}')


    # Plot error data in the last axis
    axes[0][len(layer_data)].plot(error_data)
    axes[0][len(layer_data)].set_title('Error Data')

    # Adjust layout and refresh the plot
    plt.tight_layout()
    plt.draw()  # Use plt.draw() to update the figure
    plt.pause(0.1)  # Pause briefly to allow updates

    return fig, axes  # Return fig and axes for reuse in future calls


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    plt.ion()

    print("making network")
    layout = (
        main.FullyPopulated(30000, 500, main.Activation.ReLU),
        main.FullyPopulated(500, 2, main.Activation.Sigmoid)
    )

    # todo - Fix a problem where we get exploding values when we have a fully populated going into a second pop net

    net = main.Network(layout)
    #net = main.Network.load("ZF2.pyn")

    error_data = []
    fig, axs = None, None
    print("go")
    for i in range(500):
        outputs = net.backpropagation(trainingData[:5], 0.01)
        error_data.append(outputs)
        print(i, outputs)

        data = net.forward_pass(random.choice(trainingData)[0], True)
        fig, axs = display_network(net, data[1], error_data, fig, axs)

        if i % 20 == 0:
            print("Saving...")
            net.save("ZF2.pyn")

    net.save("ZF2.pyn")
