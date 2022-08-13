import numpy as np
from PIL import Image


class Transformations:
    @staticmethod
    def grad(a):
        initial_shape = a.shape
        if len(initial_shape) > 2:
            a = a.flatten()[0:a.size:initial_shape[2]]
            a.shape = (initial_shape[0], initial_shape[1])
        gradient_kernel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        gradient_kernel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        def pad_with_zeros(a, kernel_shape):
            array_shape = a.shape
            new_array = np.insert(a, 0, np.zeros((kernel_shape[1] // 2, array_shape[0],), dtype=int), axis=1)
            new_array = np.insert(new_array, 0, np.zeros((kernel_shape[0] // 2, new_array.shape[1]), dtype=int), axis=0)
            new_array = np.insert(new_array, new_array.shape[1],
                                  np.zeros((kernel_shape[1] // 2, new_array.shape[0],), dtype=int), axis=1)
            new_array = np.insert(new_array, new_array.shape[0],
                                  np.zeros((kernel_shape[0] // 2, new_array.shape[1]), dtype=int), axis=0)
            return new_array

        def remove_padding(a, array_shape, kernel_shape):
            return a[kernel_shape[0] // 2:array_shape[0] + kernel_shape[0] // 2,
                   kernel_shape[0] // 2:array_shape[1] + kernel_shape[0] // 2]

        if gradient_kernel_x.shape != gradient_kernel_y.shape:
            raise Exception('Gradient kernels must be of the same size')
        array_shape = (a.shape[0], a.shape[1])
        out = np.zeros(array_shape)
        new_array = pad_with_zeros(a, gradient_kernel_x.shape)
        kernel = np.rot90(np.rot90(gradient_kernel_x))
        kernel_shape = kernel.shape
        for i in range(kernel_shape[0] // 2, array_shape[0] + kernel_shape[0] // 2):
            for j in range(kernel_shape[0] // 2, array_shape[1] + kernel_shape[0] // 2):
                adjacent_elements = new_array[i - (kernel_shape[0] // 2):i + (kernel_shape[0] // 2) + 1, j - (kernel_shape[1] // 2):j + (kernel_shape[1] // 2) + 1]
                grad_x = np.sum(np.multiply(kernel, adjacent_elements))
                grad_y = np.sum(np.multiply(gradient_kernel_y, adjacent_elements))
                out[i - kernel_shape[0] // 2, j - kernel_shape[1] // 2] = np.sqrt(np.square(grad_x) + np.square(grad_y))
        out.shape = (initial_shape[0], initial_shape[1])
        return out

    @staticmethod
    def normalize(image):
        if len(image.shape) > 2:
            image = image[:, :, :3].mean(axis=2)
        initial_shape = image.shape

        image = image.flatten()
        uq, count = np.unique(image, return_counts=True)
        count = np.asarray(list(map(lambda x: x / image.size, count)))

        count_cumsum = np.cumsum(count)
        equalized_frequencies = np.asarray(list(map(lambda x: 255 * x, count_cumsum))).astype(int)

        plot_data = dict(zip(uq, equalized_frequencies))

        image = np.asarray(list(map(lambda x: plot_data[x], image)))
        image.shape = (initial_shape[0], initial_shape[1])
        return image

    @staticmethod
    def gaussian_blur(image):
        initial_shape = image.shape
        if len(initial_shape) > 2:
            image = image.flatten()[0:image.size:initial_shape[2]]
            image.shape = (initial_shape[0], initial_shape[1])

        def gaussian_kernel(shape):
            sigma = 0.001
            kernel = np.zeros(shape)
            s = 0.0
            k = 1.0
            for i in range(-shape[0] // 2, (shape[0] // 2) + 1):
                for j in range(-shape[1] // 2, (shape[1] // 2) + 1):
                    r = np.sqrt(np.square(i) + np.square(j))
                    kernel[i + shape[0] // 2, j + shape[1] // 2] = k * np.exp(-(r * r) / 2 * np.square(sigma))

            return kernel / np.sum(kernel)

        smoothing_kernel = gaussian_kernel((7, 7))

        def pad_with_zeros(a, kernel_shape):
            array_shape = a.shape
            new_array = np.insert(a, 0, np.zeros((kernel_shape[1] // 2, array_shape[0],), dtype=int), axis=1)
            new_array = np.insert(new_array, 0, np.zeros((kernel_shape[0] // 2, new_array.shape[1]), dtype=int), axis=0)
            new_array = np.insert(new_array, new_array.shape[1],
                                  np.zeros((kernel_shape[1] // 2, new_array.shape[0],), dtype=int), axis=1)
            new_array = np.insert(new_array, new_array.shape[0],
                                  np.zeros((kernel_shape[0] // 2, new_array.shape[1]), dtype=int), axis=0)
            return new_array

        def remove_padding(a, array_shape, kernel_shape):
            return a[kernel_shape[0] // 2:array_shape[0] + kernel_shape[0] // 2,
                   kernel_shape[0] // 2:array_shape[1] + kernel_shape[0] // 2]

        def convolve(a, kernel):
            array_shape = a.shape
            new_array = pad_with_zeros(a, kernel.shape)
            kernel = np.rot90(np.rot90(kernel))
            kernel_shape = kernel.shape
            for i in range(kernel_shape[0] // 2, array_shape[0] + kernel_shape[0] // 2):
                for j in range(kernel_shape[0] // 2, array_shape[1] + kernel_shape[0] // 2):
                    adjacent_elements = new_array[i - (kernel_shape[0] // 2):i + (kernel_shape[0] // 2) + 1,
                                        j - (kernel_shape[1] // 2):j + (kernel_shape[1] // 2) + 1]
                    new_array[i, j] = np.sum(np.multiply(kernel, adjacent_elements))
            new_array = remove_padding(new_array, array_shape, kernel_shape)
            return new_array
        out = convolve(image, smoothing_kernel)
        out.shape = (initial_shape[0], initial_shape[1])
        return out

    @staticmethod
    def laplacian_sharpening(image):
        initial_shape = image.shape
        if len(initial_shape) > 2:
            image = image.flatten()[0:image.size:initial_shape[2]]
            image.shape = (initial_shape[0], initial_shape[1])

        laplacian = np.ones((3, 3)) * -1
        laplacian[1, 1] = 8

        def pad_with_zeros(a, kernel_shape):
            array_shape = a.shape
            new_array = np.insert(a, 0, np.zeros((kernel_shape[1] // 2, array_shape[0],), dtype=int), axis=1)
            new_array = np.insert(new_array, 0, np.zeros((kernel_shape[0] // 2, new_array.shape[1]), dtype=int), axis=0)
            new_array = np.insert(new_array, new_array.shape[1],
                                  np.zeros((kernel_shape[1] // 2, new_array.shape[0],), dtype=int), axis=1)
            new_array = np.insert(new_array, new_array.shape[0],
                                  np.zeros((kernel_shape[0] // 2, new_array.shape[1]), dtype=int), axis=0)
            return new_array

        def remove_padding(a, array_shape, kernel_shape):
            return a[kernel_shape[0] // 2:array_shape[0] + kernel_shape[0] // 2,
                   kernel_shape[0] // 2:array_shape[1] + kernel_shape[0] // 2]

        def sharpen(a, kernel):
            array_shape = a.shape
            new_array = pad_with_zeros(a, kernel.shape)
            out = np.zeros(array_shape)
            kernel_shape = kernel.shape
            kernel = np.rot90(np.rot90(kernel))
            for i in range(kernel_shape[0] // 2, array_shape[0] + kernel_shape[0] // 2):
                for j in range(kernel_shape[0] // 2, array_shape[1] + kernel_shape[0] // 2):
                    adjacent_elements = new_array[i - (kernel_shape[0] // 2):i + (kernel_shape[0] // 2) + 1,
                                        j - (kernel_shape[1] // 2):j + (kernel_shape[1] // 2) + 1]
                    derivative = np.sum(np.multiply(kernel, adjacent_elements))
                    out[i - kernel_shape[0] // 2, j - kernel_shape[1] // 2] = derivative
            out = out.astype(int)
            return a - out if kernel[1, 1] < 0 else a + out

        out = sharpen(image, laplacian)
        out.shape = (initial_shape[0], initial_shape[1])
        return out

    @staticmethod
    def negative(image):
        initial_shape = image.shape
        if len(initial_shape) > 2:
            image = image.flatten()[0:image.size:initial_shape[2]]
            image.shape = (initial_shape[0], initial_shape[1])

        image = 256 - image
        image.shape = (initial_shape[0], initial_shape[1])
        return image

