import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd.variable import Variable
import time
from Graphs import Graph, steinertree
from grouping import EditDistance
from src.Models.models import Encoder
from src.Models.models import ImitateJoint
from src.utils import read_config
from src.utils.Grouping import GenerateGroupings, Grouping
from src.utils.reinforce import Reinforce
from src.utils.train_utils import beams_parser, prepare_input_op
from src.Models.models import ParseModelOutput
from src.utils.generators.mixed_len_generator import MixedGenerateData

max_len = 13
reward = "chamfer"
rotations = 6
config = read_config.Config("config_synthetic.yml")

# CNN encoder
encoder_net = Encoder(config.encoder_drop)
encoder_net.cuda()

# Load the terminals symbols of the grammar
with open("terminals.txt", "r") as file:
    unique_draw = file.readlines()
for index, e in enumerate(unique_draw):
    unique_draw[index] = e[0:-1]

imitate_net = ImitateJoint(
        hd_sz=config.hidden_size,
        input_size=config.input_size,
        encoder=encoder_net,
        mode=config.mode,
        num_draws=len(unique_draw),
        canvas_shape=config.canvas_shape)

imitate_net.cuda()
imitate_net.epsilon = 0
reinforce = Reinforce(unique_draws=unique_draw)
pretrained_dict = torch.load(config.pretrain_modelpath)
imitate_net_dict = imitate_net.state_dict()
pretrained_dict = {
    k: v
    for k, v in pretrained_dict.items() if k in imitate_net_dict
}
imitate_net_dict.update(pretrained_dict)
imitate_net.load_state_dict(imitate_net_dict)
imitate_net.eval()

parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len, config.canvas_shape)

editdistance = EditDistance()

config = read_config.Config("config_synthetic.yml")

generator = GenerateGroupings("../CSG_Grouping/data/four_ops/", 10000, 1000)
grouping = Grouping()

data_labels_paths = {3: "../CSG_Grouping/data/one_op/expressions.txt",
                     5: "../CSG_Grouping/data/two_ops/expressions.txt",
                     13: "../CSG_Grouping/data/six_ops/expressions.txt"}
# first element of list is num of training examples, and second is number of
# testing examples.
proportion = config.proportion  # proportion is in percentage. vary from [1, 100].
dataset_sizes = {
    3: [30000, 50 * proportion],
    5: [110000, 500 * proportion],
    13: [370000, 1000 * proportion]
}
max_len = 13

generator = MixedGenerateData(data_labels_paths=data_labels_paths,
                              batch_size=config.batch_size,
                              canvas_shape=config.canvas_shape)

from src.utils.train_utils import image_from_expressions

parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1, max_len, config.canvas_shape)

import json

with open("../CSG_Grouping/data/four_ops/train_substrings.json", "r") as file:
    train_expressions = json.load(file)

pattern1 = [[1, 120, 64, 64],
            [70, 1, 64, 64],
            [170, 1, 64, 64],
            [230, 120, 64, 64],
            [170, 230, 64, 64],
            [70, 230, 64, 64]]

pattern2 = [[1, 120, 64, 64],
            [70, 1, 64, 64],
            [170, 1, 64, 64],
            [230, 120, 64, 64],
            [170, 230, 64, 64],
            [70, 230, 64, 64],
            [75, 75, 64, 64],
            [75, 155, 64, 64],
            [155, 75, 64, 64],
            [155, 155, 64, 64]]

pattern3 = [[1, 120, 64, 64],
            [70, 1, 64, 64],
            [170, 1, 64, 64],
            [230, 120, 64, 64],
            [170, 230, 64, 64],
            [70, 230, 64, 64],
            [75, 75, 64, 64],
            [75, 155, 64, 64],
            [155, 125, 64, 64]]

pattern = {0: pattern1,
           1: pattern2,
           2: pattern3}
selected_indices = [20, 23, 40, 43, 65, 84, 117, 156, 166, 176,
                    179, 225, 231, 258, 295, 309, 375, 398, 403,
                    409, 419, 449, 505, 511]

canvases = []
for _ in range(100):
    all_expressions = []
    pat = pattern[np.random.choice(3)]
    index = np.random.choice(selected_indices, 5, replace=False)
    for i in index:
        for k in train_expressions[str(i)].keys():
            all_expressions.append(train_expressions[str(i)][k])

    indices = np.random.choice(len(all_expressions), len(pat), replace=False)
    indices, index, len(pat)

    canvas = np.zeros((300, 300), dtype=np.bool)
    images = image_from_expressions(parser, all_expressions)

    for i in range(len(pat)):
        x, y, h, w = pat[i]
        canvas[x:x + h, y:y + w] = images[i]

    plt.imshow(canvas, cmap="Greys")
    plt.grid("off")
    plt.axis("off")
    canvases.append(canvas)

for canvas_id, canvas in enumerate(canvases):
    offset_1 = 0
    Rotated_Images = []
    similarity_matrix, bbs, objects = grouping.group(canvas)
    NUM = len(objects)
    for obj in objects:
        # Rotate six times
        rotated_images = []
        for i in range(6):
            degree = 30 * i
            temp_image = np.zeros((128, 128), dtype=np.bool)
            temp_image[64 - obj.shape[0] // 2: 64 - obj.shape[0] // 2 + obj.shape[0],
            64 - obj.shape[1] // 2: 64 - obj.shape[1] // 2 + obj.shape[1], ] = obj
            M = cv2.getRotationMatrix2D((64, 64), degree, 1)
            img = cv2.warpAffine(temp_image.astype(np.float32), M, (128, 128))
            rotated_images.append(img > 0)

        #         plotall([rotated_images])
        for i in range(6):
            image = grouping.find_unique(rotated_images[i], grouping.tightboundingbox(rotated_images[i]))
            image = image[0]
            # This is to avoid the situation when the rotated
            # image's size is greater than 64.
            if image.shape[0] > 64 or image.shape[1] > 64:
                rotated_images[i] = np.zeros((64, 64), np.bool)
            else:
                rotated_images[i] = grouping.replace_in_small_canvas(image, [64, 64])
        Rotated_Images.append(rotated_images)

    data = []
    for r in Rotated_Images:
        data += r
    data = np.stack(data, 0)
    data = np.expand_dims(data, 0).astype(np.float32)
    data_ = np.expand_dims(data, 2)

    print ("Beam-Searching...")
    beam_width = 20
    config.batch_size = rotations * NUM
    labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
    one_hot_labels = prepare_input_op(labels, len(unique_draw))
    one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
    data = Variable(torch.from_numpy(data_), volatile=True).cuda()

    all_beams, next_beams_prob, all_inputs = imitate_net.beam_search(
            [data, one_hot_labels], beam_width, max_len)

    beam_labels = beams_parser(
            all_beams, data_.shape[1], beam_width=beam_width)

    beam_labels_numpy = np.zeros(
            (config.batch_size * beam_width, max_len), dtype=np.int32)
    for i in range(data_.shape[1]):
        beam_labels_numpy[i * beam_width:(i + 1) * beam_width, :] = beam_labels[i]

    # find expression from these predicted beam labels
    expressions = [""] * config.batch_size * beam_width
    for i in range(config.batch_size * beam_width):
        for j in range(max_len):
            expressions[i] += unique_draw[beam_labels_numpy[i, j]]
    for index, prog in enumerate(expressions):
        expressions[index] = prog.split("$")[0]

    predicted_images = image_from_expressions(parser, expressions)
    target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
    target_images_new = np.repeat(target_images, axis=0, repeats=beam_width)

    beam_R = np.sum(np.logical_and(target_images_new, predicted_images),
                    (1, 2)) / np.sum(np.logical_or(target_images_new, predicted_images), (1, 2))

    best_beam_indices = []
    selected_expressions = []
    rotations = 6
    rewards = []
    for r in range(config.batch_size // (rotations)):
        index = np.argmax(beam_R[r * beam_width * rotations:(r + 1) * beam_width * rotations])
        index = index // beam_width
        rewards.append(beam_R[r * rotations * beam_width + index * beam_width: \
                              r * rotations * beam_width + index * beam_width + beam_width])
        selected_expressions.append( \
                expressions[r * rotations * beam_width + index * beam_width: \
                            r * rotations * beam_width + index * beam_width + beam_width])

    weights = np.ones((len(selected_expressions), len(selected_expressions), beam_width, beam_width),
                      dtype=np.float32)
    weights *= 1000
    print("Calculating weights")

#     import time
#     t1 = time.time()
#     for i in range(len(selected_expressions)):
#         for j in range(len(selected_expressions)):
#             for b1 in range(beam_width):
#                 for b2 in range(beam_width):
#                     if i == j:
#                         continue
#                     for b in range(beam_width):
#                         prog_ib = selected_expressions[i][b1]
#                         prog_jb = selected_expressions[j][b2]
#                         weights[i, j, b1, b2] = editdistance.edit_distance(prog_jb, prog_ib, iou=0)
#     t2 = time.time() - t1
#     print ("time taken in calculating the weights ", t2)

#     root_weights = np.zeros((NUM, beam_width))
#     for i in range(NUM):
#         for b in range(beam_width):
#             program_tokens = editdistance.parse(selected_expressions[i][b])
#             root_weights[i, b] = len(program_tokens)
#     K = 20
#     nodes = NUM
#     graph = Graph()
#     vertices = {}

#     for i in range(nodes + 1):
#         graph.addVertex(i)

#     for i in range(nodes):
#         graph.addEdge(0, i + 1, root_weights[i])

#     for i in range(1, nodes + 1):
#         for j in range(1, nodes + 1):
#             graph.addEdge(i, j, weights[i - 1, j - 1])

#     graph.vertList[0].root = True
#     graph.vertex_keys()

#     Nodes = steinertree(graph, graph.vertList[0])

#     adjacency = np.zeros((NUM + 1, NUM + 1), bool)
#     exps = {}
#     for k, v in graph.vertList.items():
#         if k > 0:
#             exps[k] = selected_expressions[k - 1][v.program_id]

#     for k, v in graph.vertList.items():
#         if v.pred:
#             print(graph.vertex2keys[v.pred], "==>", k)
#             adjacency[k, graph.vertex2keys[v.pred]] = True
#     temp = [exps[k] for k in sorted(exps.keys())]
#     clustered_images = image_from_expressions(parser, temp)

#     reconstructed_canvas = np.zeros((300, 300), dtype=np.bool)
#     for i in range(NUM):
#         similarity_matrix, temp_bbs, objects = grouping.group(clustered_images[i])
#         reconstructed_canvas[bbs[i][0]:bbs[i][0] + temp_bbs[0][3],
#         bbs[i][1]:bbs[i][1] + temp_bbs[0][2]] = objects[0]
#     plt.imshow(reconstructed_canvas)
#     break