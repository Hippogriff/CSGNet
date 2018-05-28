"""
Training script specially designed for REINFORCE training.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.refine import optimize_expression
import os
import json
import numpy as np
import torch
from src.Models.models import ParseModelOutput
from src.utils import read_config
import sys
from src.Models.models import ImitateJoint
from src.Models.models import Encoder
from src.utils.generators.shapenet_generater import Generator
from src.utils.reinforce import Reinforce
from src.utils.train_utils import prepare_input_op, beams_parser, validity, image_from_expressions
from torch.autograd import Variable
from src.utils.train_utils import chamfer

REFINE = False
SAVE_VIZ = False


if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
    config = read_config.Config("config_synthetic.yml")

encoder_net = Encoder()
encoder_net.cuda()

# Load the terminals symbols of the grammar
with open("terminals.txt", "r") as file:
    unique_draw = file.readlines()
for index, e in enumerate(unique_draw):
    unique_draw[index] = e[0:-1]

# RNN decoder
imitate_net = ImitateJoint(
    hd_sz=config.hidden_size,
    input_size=config.input_size,
    encoder=encoder_net,
    mode=config.mode,
    num_draws=len(unique_draw),
    canvas_shape=config.canvas_shape)
imitate_net.cuda()
imitate_net.epsilon = config.eps

max_len = 13
beam_width = 5
config.test_size = 3000
imitate_net.eval()
imitate_net.epsilon = 0
paths = [config.pretrain_modelpath]
parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len,
                          config.canvas_shape)
for p in paths:
    print(p)
    pred_expressions = []
    image_path = "data/cad/predicted_images/{}/beam_search_{}/images/".format(
        p.split("/")[-1], beam_width)
    expressions_path = "data/cad/predicted_images/{}/beam_search_{}/expressions/".format(
        p.split("/")[-1], beam_width)
    results_path = "data/cad/predicted_images/{}/beam_search_{}/".format(
        p.split("/")[-1], beam_width)

    tweak_expressions_path = "data/cad/predicted_images/{}/tweak/expressions/".format(
        p.split("/")[-1])
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(expressions_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs(os.path.dirname(tweak_expressions_path), exist_ok=True)

    config.pretrain_modelpath = p
    print("pre loading model")
    pretrained_dict = torch.load(config.pretrain_modelpath)
    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

    generator = Generator()

    reinforce = Reinforce(unique_draws=unique_draw)
    test_gen = generator.test_gen(
        batch_size=config.batch_size,
        path="data/cad/cad.h5",
        if_augment=False)

    Rs = 0
    CDs = 0
    Target_images = []
    for batch_idx in range(config.test_size // config.batch_size):
        print(batch_idx)
        data_ = next(test_gen)
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
        Target_images.append(data_[-1, :, 0, :, :])
        for i in range(data_.shape[1]):
            beam_labels_numpy[i * beam_width:(
                i + 1) * beam_width, :] = beam_labels[i]

        # find expression from these predicted beam labels
        expressions = [""] * config.batch_size * beam_width
        for i in range(config.batch_size * beam_width):
            for j in range(max_len):
                expressions[i] += unique_draw[beam_labels_numpy[i, j]]
        for index, prog in enumerate(expressions):
            expressions[index] = prog.split("$")[0]

        pred_expressions += expressions
        predicted_images = image_from_expressions(parser, expressions)
        target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
        target_images_new = np.repeat(
            target_images, axis=0, repeats=beam_width)

        beam_R = np.sum(np.logical_and(target_images_new, predicted_images),
                        (1, 2)) / np.sum(np.logical_or(target_images_new, predicted_images), (1, 2))

        R = np.zeros((config.batch_size, 1))
        for r in range(config.batch_size):
            R[r, 0] = max(beam_R[r * beam_width:(r + 1) * beam_width])

        Rs += np.mean(R)

        beam_CD = chamfer(target_images_new, predicted_images)

        CD = np.zeros((config.batch_size, 1))
        for r in range(config.batch_size):
            CD[r, 0] = min(beam_CD[r * beam_width:(r + 1) * beam_width])

        CDs += np.mean(CD)

        if SAVE_VIZ:
            for j in range(0, config.batch_size):
                f, a = plt.subplots(1, beam_width + 1, figsize=(30, 3))
                a[0].imshow(data_[-1, j, 0, :, :], cmap="Greys_r")
                a[0].axis("off")
                a[0].set_title("target")
                for i in range(1, beam_width + 1):
                    a[i].imshow(
                        predicted_images[j * beam_width + i - 1],
                        cmap="Greys_r")
                    a[i].set_title("{}".format(i))
                    a[i].axis("off")
                plt.savefig(
                    image_path +
                    "{}.png".format(batch_idx * config.batch_size + j),
                    transparent=0)
                plt.close("all")

    print(
        "average chamfer distance: {}".format(
            CDs / (config.test_size // config.batch_size)),
        flush=True)

    if REFINE:
        Target_images = np.concatenate(Target_images, 0)
        tweaked_expressions = []
        scores = 0
        for index, value in enumerate(pred_expressions):
            prog = parser.Parser.parse(value)
            if validity(prog, len(prog), len(prog) - 1):
                optim_expression, score = optimize_expression(
                    value,
                    Target_images[index // beam_width],
                    metric="chamfer",
                    max_iter=None)
                print(value)
                tweaked_expressions.append(optim_expression)
                scores += score
            else:
                # If the predicted program is invalid
                tweaked_expressions.append(value)
                scores += 16

        print("chamfer scores", scores / len(tweaked_expressions))
        with open(
                tweak_expressions_path +
                "chamfer_tweak_expressions_beamwidth_{}.txt".format(beam_width),
                "w") as file:
            for index, value in enumerate(tweaked_expressions):
                file.write(value + "\n")

    Rs = Rs / (config.test_size // config.batch_size)
    CDs = CDs / (config.test_size // config.batch_size)
    print(p, Rs, CDs)
    if REFINE:
        results = {
            "iou": Rs,
            "chamferdistance": CDs,
            "tweaked_chamfer_distance": scores / len(tweaked_expressions)
        }
    else:
        results = {"iou": Rs, "chamferdistance": CDs}

    with open(expressions_path +
              "expressions_beamwidth_{}.txt".format(beam_width), "w") as file:
        for e in pred_expressions:
            file.write(e + "\n")

    with open(results_path + "results_beam_width_{}.org".format(beam_width),
              'w') as outfile:
        json.dump(results, outfile)
