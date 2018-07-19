"""
Visualize the expressions in the form of images
"""
import matplotlib.pyplot as plt
from src.Models.models import ParseModelOutput

from src.utils.train_utils import prepare_input_op, beams_parser, validity, image_from_expressions

# Load the terminals symbols of the grammar
canvas_shape = [64, 64]
max_len = 13

with open("terminals.txt", "r") as file:
    unique_draw = file.readlines()
for index, e in enumerate(unique_draw):
    unique_draw[index] = e[0:-1]

expressions = ["c(32,32,28)c(32,32,24)-s(32,32,28)s(32,32,20)-+t(32,32,20)+"]

parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len, canvas_shape)
predicted_images = image_from_expressions(parser, expressions)
plt.imshow(predicted_images[0], cmap="Greys")
plt.grid("off")
plt.axis("off")
plt.show()