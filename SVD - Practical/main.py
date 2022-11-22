# import libraries for image and matrix processing
import matplotlib.pyplot as plt
import numpy
import numpy as np
from PIL import Image
import os

# a class to hold and process a PIL image object
class image_svd(object):
    def __init__(self, path: str):
        self.path      = path
        self.image     = Image.open(path)
        self.color_map = ["Reds", "Greens", "Blues"]

    # get 2D matrix of color intensities. 0: red, 1: green, 2: blue
    def get_component(self, index: int) -> np.matrix:
        return np.matrix(self.image.getdata(index)).reshape(self.image.size[::-1])

    # get 3 2D matrices of RGB components
    def get_all_components(self) -> tuple:
        return tuple([self.get_component(index) for index in range(3)])

    # convert an index to RGB name for color map
    def __convert_index_color(self, index: int) -> str | None:
        if index in range(3):
            return self.color_map[index]
        else:
            return None

    # plot the image
    def plot_image(self) -> None:
        plt.imshow(self.image)

    # plot a 2D matrix of color intensities.
    def plot_component(self, index: int) -> None:
        plt.imshow(
            self.get_component(index),
            cmap = self.__convert_index_color(index)
        )

    # plot all components side by side
    def plot_all_components(self) -> None:
        figure = plt.figure()
        # figure.suptitle(f"RGB Components of '{os.path.basename(self.path)}'", fontsize = 20)

        # set the specific figure titles
        color_titles = ["Red Component", "Green Component", "Blue Component"]

        for index in range(3):
            axs = figure.add_subplot(1, 3, index + 1)
            axs.set_title(color_titles[index], fontsize = 24)
            axs.imshow(self.get_component(index), cmap = "Greys")# self.color_map[index])

        plt.tight_layout(w_pad = 2.5)

    # scale and center a matrix
    def __scale_matrix(self, matrix: np.matrix) -> np.matrix:
        return np.matrix((matrix - matrix.mean()) / matrix.std())

    # perform the SV decomposition
    def __perform_svd(self, matrix: np.matrix) -> tuple:
        return np.linalg.svd(matrix)

    # get the variance explained by the singular values
    def var_singular(self, matrix: np.matrix, scale: bool = True) -> numpy.ndarray:
        if scale:
            matrix = self.__scale_matrix(matrix)
        matrix_U, matrix_s, matrix_V = self.__perform_svd(matrix)

        var_explained = np.round(matrix_s ** 2 / np.sum(matrix_s ** 2), decimals = 3)
        return var_explained

    # plot a bar chart from an nd.array
    def __plot_bar_1d(self, values: np.array) -> None:
        plt.bar([str(index + 1) for index in range(len(values))], values)
        plt.xlabel("Singular Vector")
        plt.ylabel("Variance Explained")
        plt.tight_layout()

    # plot a bar chart showing the variance explained by the singular values
    def plot_singular(self, matrix: np.matrix, total: int = 20) -> None:
        self.__plot_bar_1d(self.var_singular(matrix)[:total])

    # get a rank-k approximation from an SV decomposition
    def __rank_k_approximation(self,matrix_U: np.matrix, matrix_s: np.ndarray,
                                 matrix_V: np.matrix, num_components: int) -> np.matrix:
        return np.matrix(matrix_U[:, :num_components] * np.diag(
            matrix_s[:num_components]) * np.matrix(matrix_V[:num_components, :]))

    # get a rank-k approximation for a matrix
    def rank_k_approximation(self, matrix: np.matrix, num_components: int) -> np.matrix:
        matrix_U, matrix_s, matrix_V = self.__perform_svd(matrix)
        return self.__rank_k_approximation(matrix_U, matrix_s, matrix_V, num_components)

    # recombine RGB components into an image. Can definitely be done faster but numpy matrix is a pain
    def recombine_rgb(self, components: tuple) -> np.ndarray:
        combined_array = np.zeros([self.image.size[1], self.image.size[0], 3], dtype=np.uint8)
        for row_index in range(self.image.size[1]):
            for col_index in range(self.image.size[0]):
                combined_array[row_index, col_index] = [
                    components[0][row_index, col_index],
                    components[1][row_index, col_index],
                    components[2][row_index, col_index]
                ]
        return combined_array

    # plot n approximations. The components increase according to e^x to give better results
    def plot_n_approximations(self, matrix: np.matrix, n_plots: int,
                              color_index: int, title: str = "", scale: bool = True) -> None:
        # config and parameters
        num_cols = matrix.shape[1]
        color = self.__convert_index_color(color_index)

        # figure setup
        figure = plt.figure()
        figure.suptitle(title)

        # perform the svd
        if scale:
            matrix = self.__scale_matrix(matrix)
        matrix_U, matrix_s, matrix_V = self.__perform_svd(matrix)

        # for each plot
        for plot_index in range(n_plots):
            # calculate equivalent number of components
            num_components = int((np.e ** (((plot_index + 1) / n_plots) * np.log(num_cols + 1))) - 1) + 1

            # get the approximation for that number of components
            approx_image = self.__rank_k_approximation(matrix_U, matrix_s, matrix_V, num_components)

            # plot that approximation
            axs = figure.add_subplot(int(np.ceil(n_plots / 5)), 5, n_plots - plot_index)
            axs.title.set_text(f"Number of Components: {num_components}")
            axs.imshow(approx_image, cmap = color)


    # will plot n approximations for each RGB components
    def plot_n_approximations_rgb(self, n_plots: int, title: str = "") -> None:
        # config and parameters
        matrices = self.get_all_components()
        if matrices[0].shape[1] < matrices[0].shape[0]:
            num_cols = matrices[0].shape[1]
        else:
            num_cols = matrices[0].shape[0]

        # setup figure
        figure = plt.figure()
        figure.suptitle(title)

        # store color components
        color_components = []

        # for each color component
        for color_index in range(3):
            matrix_U, matrix_s, matrix_V = self.__perform_svd(matrices[color_index])
            color = self.__convert_index_color(color_index)

            # store components for this color
            current_components = []

            # for each plotting components
            for plot_index in range(n_plots):
                grid_index     = (color_index * n_plots) + plot_index
                num_components = int((np.e ** (((plot_index + 1) / n_plots) * np.log(num_cols + 1))) - 1) + 1

                # get the approximation for that number of components
                approx_image = self.__rank_k_approximation(matrix_U, matrix_s, matrix_V, num_components)
                current_components.append(numpy.array(numpy.array(approx_image)))

                # plot that approximation
                axs = figure.add_subplot(4, n_plots, int((n_plots * (np.floor(grid_index / n_plots) + 1)) - np.mod(grid_index, n_plots)))
                if color_index == 0:
                    axs.title.set_text(f"Number of Components: {num_components}")
                axs.imshow(approx_image, cmap = color)

            color_components.append(current_components)

        # plot the recombined images
        for plot_index in range(n_plots):
            combined_array = self.recombine_rgb((
                color_components[0][plot_index],
                color_components[1][plot_index],
                color_components[2][plot_index]
            ))
            axs = figure.add_subplot(4, n_plots, (n_plots * 4) - plot_index)
            axs.imshow(combined_array)

test = image_svd("ffc_lores.jpg")
image_data = np.array(test.image.getdata()).reshape([*test.image.size[::-1], 3])
components = test.get_all_components()
recombined_data = np.array(test.recombine_rgb(components))

figure = plt.figure()

axs = figure.add_subplot(1, 3, 1)
axs.title.set_text("Original Image")
plt.imshow(image_data)

axs = figure.add_subplot(1, 3, 2)
axs.title.set_text("Recombined Image")
plt.imshow(recombined_data)

diff_image_data = recombined_data - image_data
axs = figure.add_subplot(1, 3, 3)
axs.title.set_text("Difference Image (sum = 0)")
plt.imshow(diff_image_data)

print(np.sum(diff_image_data))

plt.show()