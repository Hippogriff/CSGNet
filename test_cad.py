import matplotlib
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import torch
from torch.autograd.variable import Variable
import sys
from src.utils import read_config
from src.Models.models import ImitateJoint
from src.Models.models import Encoder
from src.utils.generators.shapenet_generater import Generator
from src.utils.reinforce import Reinforce
from src.utils.train_utils import prepare_input_op

max_len = 13
power = 20
reward = "chamfer"
if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
    config = read_config.Config("config_cad.yml")

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

test_size = 3000
# This is to find top-1 performance.
paths = [config.pretrain_modelpath]
save_viz = False
for p in paths:
    print(p, flush=True)
    config.pretrain_modelpath = p

    image_path = "data/cad/predicted_images/{}/top_1_prediction/images/".format(
        p.split("/")[-1])
    expressions_path = "data/cad/predicted_images/{}/top_1_prediction/expressions/".format(
        p.split("/")[-1])

    results_path = "data/cad/predicted_images/{}/top_1_prediction/".format(
        p.split("/")[-1])
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(expressions_path), exist_ok=True)

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
    data_set_path = "data/cad/cad.h5"
    train_gen = generator.train_gen(
        batch_size=config.batch_size, path=data_set_path, if_augment=False)
    val_gen = generator.val_gen(
        batch_size=config.batch_size, path=data_set_path, if_augment=False)
    test_gen = generator.test_gen(
        batch_size=config.batch_size, path=data_set_path, if_augment=False)

    imitate_net.epsilon = 0
    RS_iou = 0
    RS_chamfer = 0
    distances = 0
    pred_expressions = []
    for i in range(test_size // config.batch_size):
        data_ = next(test_gen)
        labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
        one_hot_labels = prepare_input_op(labels, len(unique_draw))
        one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
        data = Variable(torch.from_numpy(data_), volatile=True).cuda()
        outputs, samples = imitate_net([data, one_hot_labels, max_len])
        R, _, pred_images, expressions = reinforce.generate_rewards(
                                                                    samples,
                                                                    data_,
                                                                    time_steps=max_len,
                                                                    stack_size=max_len // 2 + 1,
                                                                    power=1,
                                                                    reward="iou")
        RS_iou += np.mean(R) / (test_size // config.batch_size)

        R, _, _, expressions, distance = reinforce.generate_rewards(samples,
                                                                    data_,
                                                                    time_steps=max_len,
                                                                    stack_size=max_len // 2 + 1,
                                                                    power=power,
                                                                    reward="chamfer")

        RS_chamfer += np.mean(R) / (test_size // config.batch_size)
        distances += np.mean(distance) / (test_size // config.batch_size)

        for index, p in enumerate(expressions):
            expressions[index] = p.split("$")[0]
        pred_expressions += expressions
        # Save images
        if save_viz:
            for j in range(config.batch_size):
                f, a = plt.subplots(1, 2, figsize=(8, 4))
                a[0].imshow(data_[-1, j, 0, :, :], cmap="Greys_r")
                a[0].axis("off")
                a[0].set_title("target")

                a[1].imshow(pred_images[j], cmap="Greys_r")
                a[1].axis("off")
                a[1].set_title("prediction")
                plt.savefig(
                    image_path + "{}.png".format(i * config.batch_size + j),
                    transparent=0)
                plt.close("all")

    print("iou is {}: ".format(RS_iou), flush=True)
    print("chamfer reward is {}: ".format(RS_chamfer), flush=True)
    print("chamfer distance is {}: ".format(distances), flush=True)

    results = {
        "iou": RS_iou,
        "chamfer distance": distances,
        "chamfer reward": distances
    }
    with open(expressions_path + "expressions.txt", "w") as file:
        for e in pred_expressions:
            file.write(e + "\n")

    with open(results_path + "results.org", 'w') as outfile:
        json.dump(results, outfile)
