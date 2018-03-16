"""
Contains code to start the visualization process.
"""
import json
import os
import numpy as np
import torch
import sys
from torch.autograd.variable import Variable
import read_config
from src.Models.models import Encoder
from src.Models.models import ImitateJoint, validity
from src.Models.models import ParseModelOutput
from src.utils.generators.mixed_len_generator import MixedGenerateData
from src.utils.train_utils import prepare_input_op, chamfer, beams_parser

if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
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

config.test_size = sum(dataset_sizes[k][1] for k in dataset_sizes.keys())
imitate_net.eval()
Pred_Prog = []
Targ_Prog = []

# NOTE: Let us run all the programs for maximum lengths possible irrespective
# of what they actually require.
max_len = max(data_labels_paths.keys())
parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1,
                          max_len, config.canvas_shape)
metrics = {}
test_gen_objs = {}
imitate_net.eval()
imitate_net.epsilon = 0
over_all_CD = {}
programs_pred = {}
programs_tar = {}
beam_width = 10
maxx_len = max(dataset_sizes.keys())
total_size = 0

# If the batch size doesn't divide the testing set perfectly, than we ignore the last
# batch and calculate this new total test size ignoring the last batch.
for k in dataset_sizes.keys():
    test_batch_size = config.batch_size
    total_size += (dataset_sizes[k][1] // test_batch_size) * test_batch_size

for jit in [True, False]:
    total_CD = 0
    programs_pred[jit] = []
    programs_tar[jit] = []

    for k in data_labels_paths.keys():
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
            one_hot_labels = prepare_input_op(labels, len(generator.unique_draw))
            one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
            data = Variable(torch.from_numpy(data_), volatile=True).cuda()
            labels = Variable(torch.from_numpy(labels)).cuda()
            all_beams, next_beams_prob, all_inputs = imitate_net.beam_search([data, one_hot_labels], beam_width, maxx_len)

            targ_prog = parser.labels2exps(labels, k)
            beam_labels = beams_parser(all_beams, test_batch_size, beam_width=beam_width)

            beam_labels_numpy = np.zeros((test_batch_size * beam_width, maxx_len), dtype=np.int32)

            for i in range(test_batch_size):
                beam_labels_numpy[i * beam_width: (i + 1) * beam_width, :] = beam_labels[i]

            # find expression from these predicted beam labels
            expressions = [""] * test_batch_size * beam_width
            for i in range(test_batch_size * beam_width):
                for j in range(maxx_len):
                    expressions[i] += generator.unique_draw[beam_labels_numpy[i, j]]
            for index, p in enumerate(expressions):
                expressions[index] = p.split("$")[0]

            programs_tar[jit] += targ_prog
            programs_pred[jit] += expressions

            pred_images = []
            for index, exp in enumerate(expressions):
                program = parser.Parser.parse(exp)
                if validity(program, len(program), len(program) - 1):
                    stack = parser.expression2stack([exp])
                    pred_images.append(stack[-1, -1, 0, :, :])
                else:
                    pred_images.append(np.zeros(config.canvas_shape))
            pred_images = np.stack(pred_images, 0).astype(dtype=np.bool)
            target_images = data_[-1, :, 0, :, :].astype(dtype=bool)

            # repeat the target_images beamwidth times
            target_images_new = np.repeat(target_images, axis=0,
                                          repeats=beam_width)
            beam_CD = chamfer(target_images_new, pred_images)

            CD = np.zeros((test_batch_size, 1))
            for r in range(test_batch_size):
                CD[r, 0] = min(beam_CD[r * beam_width: (r + 1) * beam_width])
            total_CD += np.sum(CD)

    over_all_CD[jit] = total_CD / total_size

metrics["chamfer"] = over_all_CD
results_path = "trained_models/results/{}/".format(model_name)
os.makedirs(os.path.dirname(results_path), exist_ok=True)
print(metrics)
print(config.pretrain_modelpath)
with open("trained_models/results/{}/{}".format(model_name, "beam_{}_pred_prog.org".format(beam_width)), 'w') as outfile:
    json.dump(programs_pred, outfile)

with open("trained_models/results/{}/{}".format(model_name, "beam_{}_tar_prog.org".format(beam_width)), 'w') as outfile:
    json.dump(programs_tar, outfile)

with open("trained_models/results/{}/{}".format(model_name, "beam_{}_metrices.org".format(beam_width)), 'w') as outfile:
    json.dump(metrics, outfile)
