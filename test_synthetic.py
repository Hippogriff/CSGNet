"""
Contains code to start the visualization process.
"""
import json
import os
import read_config
import numpy as np
import torch
from torch.autograd.variable import Variable
from src.Models.models import ImitateJoint
from src.Models.models import ParseModelOutput
from src.Models.models import Encoder
from src.utils.generators.mixed_len_generator import MixedGenerateData
from src.utils.train_utils import prepare_input_op, chamfer

config = read_config.Config("config_synthetic.yml")
model_name = config.pretrain_modelpath.split("/")[-1][0:-4]
encoder_net = Encoder()
encoder_net.cuda()

data_labels_paths = {3: "data/synthetic/one_op/expressions.txt",
                     5: "data/synthetic/two_ops/expressions.txt",
                     7: "data/synthetic/three_ops/expressions.txt",
                     9: "data/synthetic/four_ops/expressions.txt",
                     11: "data/synthetic/five_ops/expressions.txt",
                     13: "data/synthetic/six_ops/expressions.txt"}
# first element of list is num of training examples, and second is number of
# testing examples.
proportion = config.proportion  # proportion is in percentage. vary from [1, 100].
dataset_sizes = {
    3: [30000, 50 * proportion],
    5: [110000, 500 * proportion],
    7: [170000, 500 * proportion],
    9: [270000, 500 * proportion],
    11: [370000, 1000 * proportion],
    13: [370000, 1000 * proportion]
}

generator = MixedGenerateData(data_labels_paths=data_labels_paths,
                              batch_size=config.batch_size,
                              train_size=config.train_size,
                              test_size=config.test_size,
                              canvas_shape=config.canvas_shape)

imitate_net = ImitateJoint(hd_sz=config.hidden_size,
                           input_size=config.input_size,
                           encoder=encoder_net,
                           mode=config.mode,
                           num_draws=len(generator.unique_draw),
                           canvas_shape=config.canvas_shape)

imitate_net.cuda()
if config.preload_model:
    print("pre loading model")
    pretrained_dict = torch.load(config.pretrain_modelpath)
    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

imitate_net.eval()
max_len = max(data_labels_paths.keys())
parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1,
                          max_len, config.canvas_shape)

# total size according to the test batch size.
total_size = 0
config.test_size = sum(dataset_sizes[k][1] for k in dataset_sizes.keys())
for k in dataset_sizes.keys():
    test_batch_size = config.batch_size
    total_size += (dataset_sizes[k][1] // test_batch_size) * test_batch_size


imitate_net.eval()
over_all_CD = {}
Pred_Prog = []
Targ_Prog = []
metrics = {}
programs_tar = {}
programs_pred = {}

for jit in [True, False]:
    total_CD = 0
    test_gen_objs = {}
    programs_tar[jit] = []
    programs_pred[jit] = []

    for k in data_labels_paths.keys():
        test_gen_objs[k] = {}
        test_batch_size = config.batch_size
        test_gen_objs[k] = generator.get_test_data(
            test_batch_size,
            k,
            num_train_images=dataset_sizes[k][0],
            num_test_images=dataset_sizes[k][1],
            jitter_program=jit)

    for k in dataset_sizes.keys():
        test_batch_size = config.batch_size
        for _ in range(dataset_sizes[k][1] // test_batch_size):
            data_, labels = next(test_gen_objs[k])
            one_hot_labels = prepare_input_op(labels,
                                              len(generator.unique_draw))
            one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
            data = Variable(torch.from_numpy(data_), volatile=True).cuda()
            labels = Variable(torch.from_numpy(labels)).cuda()
            test_output = imitate_net.test([data, one_hot_labels, max_len])
            pred_images, correct_prog, pred_prog = parser.get_final_canvas(
                test_output,
                if_just_expressions=False,
                if_pred_images=True)
            target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
            targ_prog = parser.labels2exps(labels, k)

            programs_tar[jit] += targ_prog
            programs_pred[jit] += pred_prog
            distance = chamfer(target_images, pred_images) * 64 * 0.5 / 2
            total_CD += np.sum(distance)

    over_all_CD[jit] = total_CD / total_size

metrics["chamfer"] = over_all_CD
print(metrics, model_name)
print(over_all_CD)

results_path = "trained_models/results/{}/".format(model_name)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open("trained_models/results/{}/{}".format(model_name, "pred_prog.org"), 'w') as outfile:
    json.dump(programs_pred, outfile)

with open("trained_models/results/{}/{}".format(model_name, "tar_prog.org"), 'w') as outfile:
    json.dump(programs_tar, outfile)

with open("trained_models/results/{}/{}".format(model_name, "all_metrices.org"), 'w') as outfile:
    json.dump(metrics, outfile)