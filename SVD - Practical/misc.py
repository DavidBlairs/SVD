# import libraries for image and matrix processing
import matplotlib.pyplot as plt
import numpy
import numpy as np
from PIL import Image

# create a test image
# test_array = np.zeros([200, 300, 3], dtype = np.uint8)
# test_array[:,:100]    = [255, 0, 0] # red bar
# test_array[:,100:200] = [0, 255, 0] # green bar
# test_array[:,200:300] = [0, 0, 255] # blue bar
# Image.fromarray(test_array).save("test_image.png")

# import the image
image_obj = Image.open("test_image.png")

figure = plt.figure()

figure.add_subplot(1, 4, 1)
plt.imshow(image_obj)

all_component_data = []

for component_index, component_color in enumerate(["Reds", "Greens", "Blues"]):
    component_data = np.matrix(image_obj.getdata(component_index)).reshape(image_obj.size[::-1])
    print(f"Current Color: {component_color}")
    print(component_data)

    figure.add_subplot(1, 5, 2 + component_index)
    all_component_data.append(component_data)
    plt.imshow(Image.fromarray(component_data))

combined_array = np.zeros([image_obj.size[1], image_obj.size[0], 3], dtype = np.uint8)
for row_index in range(image_obj.size[1]):
    for col_index in range(image_obj.size[0]):
        combined_array[row_index, col_index] = [
            all_component_data[0][row_index, col_index],
            all_component_data[1][row_index, col_index],
            all_component_data[2][row_index, col_index]
        ]
figure.add_subplot(1, 5, 5)
plt.imshow(combined_array)
plt.show()

