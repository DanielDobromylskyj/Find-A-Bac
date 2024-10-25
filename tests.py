import main
import os
import time
import numpy as np

OPENSLIDE_PATH = r'\openslide-win64\bin'
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(os.getcwd() + OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def test_performance(net):
    slide = openslide.open_slide("trainingData/16772.tiff")

    region_read_time = 0
    network_time = 0

    test_points = 200
    for i in range(test_points):
        start = time.time()
        region = slide.read_region((1500, 1500), 0, (100, 100)).convert("RGB")
        region_array = np.asarray(region, dtype=np.float32) / 255
        region_read_time += time.time() - start

        start = time.time()
        net.forward_pass(region_array)
        network_time += time.time() - start

    region_read_time = round(region_read_time / test_points * 1000)
    network_time = round(network_time / test_points * 1000)

    total_time = network_time + region_read_time

    print("> PERFORMANCE TEST <")
    print(f"Status (20ms): {'PASSED' if total_time <= 20 else 'FAILED'}")
    print(f"Status (50ms): {'PASSED' if total_time <= 50 else 'FAILED'}")
    print(f"Total: {total_time}ms, Load: {region_read_time}ms, Network: {network_time}ms")
    print("\n")


def test_accuracy(net):
    with open("trainingPoints.txt", "r") as f:
        unprocessedTrainingData = eval(f.read())

    tests = 0
    wrong = 0
    false_positive = 0
    false_negative = 0
    inconclusive = 0


    for segment in unprocessedTrainingData:
        slide = openslide.open_slide(segment["tiffPath"])
        for point in segment["points"]:
            img = slide.read_region(point[0], 0, (100, 100))
            img = img.convert("RGB")
            image_data_normalised = np.asarray(img, dtype=np.float32) / 255

            target = [point[1][0], 0 if point[1][0] == 1 else 1]

            output = net.forward_pass(image_data_normalised)
            tests += 1

            if output[0] > 0.5 and output[1] > 0.5:
                inconclusive += 1
                wrong += 1
            elif output[0] < 0.5 and output[1] < 0.5:
                inconclusive += 1
                wrong += 1

            elif target[0] == 1:
                if output[0] > 0.5 and output[1] < 0.5:
                    pass  # correct
                else:
                    false_negative += 1
                    wrong += 1

            elif target[1] == 1:
                if output[0] < 0.5 and output[1] > 0.5:
                    pass  # correct
                else:
                    false_positive += 1
                    wrong += 1

    print("> ACCURACY TEST <")
    print(f"Tested {tests} cases")
    print(f"Accuracy: {round(100 * (1 - (wrong / tests)))}%")
    print(f"False Negative Rate: {round(100 * ((false_negative / tests)))}%")
    print(f"False Positive Rate: {round(100 * ((false_positive / tests)))}%")
    print(f"Inconclusive Rate: {round(100 * ((inconclusive / tests)))}%")
    print("\n")


if __name__ == "__main__":
    test_net = main.Network.load("ZF.pyn")
    test_performance(test_net)
    test_accuracy(test_net)
