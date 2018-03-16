"""
This script does the post processing optimization on the programs retrieved
expression after top-1 decoding from CSGNet. So if the output expressions 
(of size test_size) from the network are already calculated, then you can 
use this script.
"""
import argparse
import json
import os
import sys

import numpy as np

import read_config
from src.utils.generators.shapenet_generater import Generator
from src.utils.refine import optimize_expression
from src.utils.reinforce import Reinforce

parser = argparse.ArgumentParser()
parser.add_argument("opt_exp_path", type=str, help="path to the expressions being "
                                                   "optmized")
parser.add_argument("opt_exp_save_path", type=str, help="path to the directory where "
                                                        "optmized expressions to be "
                                                        "saved.")
args = parser.parse_args()

if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
    config = read_config.Config("config_synthetic.yml")

# Load the terminals symbols of the grammar
with open("terminals.txt", "r") as file:
    unique_draw = file.readlines()
for index, e in enumerate(unique_draw):
    unique_draw[index] = e[0:-1]

# path where results will be stored
save_optimized_exp_path = args.opt_exp_save_path

# path to load the expressions to be optimized.
expressions_to_optmize = args.opt_exp_path

test_size = 3000
max_len = 13

# maximum number of refinement iterations to be done.
max_iter = 1

# This is the path where you want to save the results and optmized expressions
os.makedirs(os.path.dirname(save_optimized_exp_path), exist_ok=True)

generator = Generator()
reinforce = Reinforce(unique_draws=unique_draw)
data_set_path = "data/cad/cad.h5"
test_gen = generator.test_gen(
    batch_size=config.batch_size, path=data_set_path, if_augment=False)

distances = 0
target_images = []
for i in range(test_size // config.batch_size):
    data_ = next(test_gen)
    target_images.append(data_[-1, :, 0, :, :])

with open(expressions_to_optmize, "r") as file:
    Predicted_expressions = file.readlines()

# remove dollars and "\n"
for index, e in enumerate(Predicted_expressions):
    Predicted_expressions[index] = e[0:-1].split("$")[0]

print("let us start the optimization party!!")
Target_images = np.concatenate(target_images, 0)
refined_expressions = []
scores = 0
for index, value in enumerate(Predicted_expressions):
    optimized_expression, score = optimize_expression(
        value,
        Target_images[index],
        metric="chamfer",
        stack_size=max_len // 2 + 1,
        steps=max_len,
        max_iter=max_iter)
    refined_expressions.append(optimized_expression)
    scores += score
    print(index, score, scores / (index + 1), flush=True)

print(
    "chamfer scores for max_iterm {}: ".format(max_iter),
    scores / len(refined_expressions),
    flush=True)
results = {
    "chamfer scores for max_iterm {}:".format(max_iter):
        scores / len(refined_expressions)
}

with open(save_optimized_exp_path +
                  "optmized_expressions_maxiter_{}.txt".format(max_iter),
          "w") as file:
    for index, value in enumerate(refined_expressions):
        file.write(value + "\n")

with open(save_optimized_exp_path + "results_max_iter_{}.org".format(max_iter),
          'w') as outfile:
    json.dump(results, outfile)
