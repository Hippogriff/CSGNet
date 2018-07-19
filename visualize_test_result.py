"""
Visualize the expressions in the form of images
"""
import matplotlib.pyplot as plt
from src.Models.models import ParseModelOutput

from src.utils.train_utils import prepare_input_op, beams_parser, validity, image_from_expressions
import argparse
import json

# Load the terminals symbols of the grammar
canvas_shape = [64, 64]
max_len = 13

with open("terminals.txt", "r") as file:
    unique_draw = file.readlines()
for index, e in enumerate(unique_draw):
    unique_draw[index] = e[0:-1]

argparser = argparse.ArgumentParser(
    prog='visualize_expressions.py',
    usage='Visualize CSG expressions',
    description='This can show the target image and predicted image in test directory(/trained_models/results/NETWORK)',
    add_help=True,
)

argparser.add_argument('-n', '--network', help='name of the network', default='pretrained')
argparser.add_argument('-l', '--show-only-long', help='Show the result of the CSG expression longer than 50 characters', action='store_true')

args = argparser.parse_args()

with open('trained_models/results/{}/tar_prog.org'.format(args.network), 'r') as f:
    target_data = json.load(f)['true']
    
with open('trained_models/results/{}/pred_prog.org'.format(args.network), 'r') as f:
    prediction_data = json.load(f)['true']

parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len, canvas_shape)

data_num = len(target_data)
for i in range(data_num):
    if args.show_only_long:
        if len(target_data[i]) < 50:
            continue
    target_images = image_from_expressions(parser, [target_data[i]])
    prediction_images = image_from_expressions(parser, [prediction_data[i]])

    plt.subplot(121)
    plt.imshow(target_images[0], cmap='Greys')
    plt.grid('off')
    plt.axis('off')
    plt.title('target')
    plt.subplot(122)
    plt.imshow(prediction_images[0], cmap='Greys')
    plt.grid('off')
    plt.axis('off')
    plt.title('prediction')
    plt.show()






