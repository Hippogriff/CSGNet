# This script only modifies the training, so that higher len programs are
# trained better.
"""
This trains network to predict stop symbol for variable length programs.
Note that there is no padding done in RNN in contrast to traditional RNN for 
variable length programs. This is mainly because of computational 
efficiency of forward pass, that is, each batch contains only 
programs of similar length, that implies that the program of smaller lengths 
are not processed by RNN for unnecessary time steps.
Losses from all batches of different time-lengths are combined to compute 
gradient and updated in the network in one go. This ensures that every update to
the network has equal contribution (or weighted by the ratio of their 
batch sizes) coming from programs of different lengths.
"""

import logging

import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import configure, log_value
from torch.autograd.variable import Variable

from src.Models.loss import losses_joint
from src.Models.models import Encoder
from src.Models.models import ImitateJoint, ParseModelOutput
from src.utils import read_config
from src.utils.generators.mixed_len_generator import MixedGenerateData
from src.utils.learn_utils import LearningRate
from src.utils.train_utils import prepare_input_op, cosine_similarity, chamfer

config = read_config.Config("config_synthetic.yml")

model_name = config.model_path.format(config.mode)
print(config.config, flush=True)
config.write_config("log/configs/{}_config.json".format(model_name))
configure("log/tensorboard/{}".format(model_name), flush_secs=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(
    'log/logger/{}.log'.format(model_name), mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Encoder
encoder_net = Encoder(config.encoder_drop)
encoder_net.cuda()
logger.info(config.config)

data_labels_paths = {
    3: "data/synthetic/one_op/expressions.txt",
    5: "data/synthetic/two_ops/expressions.txt",
    7: "data/synthetic/three_ops/expressions.txt"
}

# proportion is in percentage. vary from [1, 100].
proportion = config.proportion
dataset_sizes = {
    3: [proportion * 250, proportion * 50],
    5: [proportion * 1000, proportion * 100],
    7: [proportion * 1500, proportion * 200]
}

generator = MixedGenerateData(
    data_labels_paths=data_labels_paths,
    batch_size=config.batch_size,
    canvas_shape=config.canvas_shape)

imitate_net = ImitateJoint(
    hd_sz=config.hidden_size,
    input_size=config.input_size,
    encoder=encoder_net,
    mode=config.mode,
    num_draws=len(generator.unique_draw),
    canvas_shape=config.canvas_shape)
imitate_net.cuda()

if config.preload_model:
    imitate_net.load_state_dict(torch.load(config.pretrain_modelpath))

for param in imitate_net.parameters():
    param.requires_grad = True

for param in encoder_net.parameters():
    param.requires_grad = True

max_len = max(data_labels_paths.keys())
parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1, max_len,
                          config.canvas_shape)

optimizer = optim.Adam(
    [para for para in imitate_net.parameters() if para.requires_grad],
    weight_decay=config.weight_decay,
    lr=config.lr)

reduce_plat = LearningRate(
    optimizer,
    init_lr=config.lr,
    lr_dacay_fact=0.2,
    patience=config.patience,
    logger=logger)
types_prog = len(dataset_sizes)
train_gen_objs = {}
test_gen_objs = {}
config.train_size = sum(dataset_sizes[k][0] for k in dataset_sizes.keys())
config.test_size = sum(dataset_sizes[k][1] for k in dataset_sizes.keys())
total_importance = sum(k for k in dataset_sizes.keys())
for k in data_labels_paths.keys():
    test_batch_size = int(config.batch_size * dataset_sizes[k][1] / \
                          config.test_size)
    # Acts as a curriculum learning
    train_batch_size = config.batch_size // types_prog
    train_gen_objs[k] = generator.get_train_data(
        train_batch_size,
        k,
        num_train_images=dataset_sizes[k][0],
        jitter_program=True)
    test_gen_objs[k] = generator.get_test_data(
        test_batch_size,
        k,
        num_train_images=dataset_sizes[k][0],
        num_test_images=dataset_sizes[k][1],
        jitter_program=True)

prev_test_loss = 1e20
prev_test_cd = 1e20
prev_test_iou = 0
for epoch in range(config.epochs):
    train_loss = 0
    Accuracies = []
    imitate_net.train()
    for batch_idx in range(config.train_size //
                           (config.batch_size * config.num_traj)):
        optimizer.zero_grad()
        loss = Variable(torch.zeros(1)).cuda().data
        for _ in range(config.num_traj):
            for k in data_labels_paths.keys():
                data, labels = next(train_gen_objs[k])
                data = data[:, :, 0:1, :, :]
                one_hot_labels = prepare_input_op(labels,
                                                  len(generator.unique_draw))
                one_hot_labels = Variable(
                    torch.from_numpy(one_hot_labels)).cuda()
                data = Variable(torch.from_numpy(data)).cuda()
                labels = Variable(torch.from_numpy(labels)).cuda()
                outputs = imitate_net([data, one_hot_labels, k])

                loss_k = (losses_joint(outputs, labels, time_steps=k + 1) / (
                    k + 1)) / len(data_labels_paths.keys()) / config.num_traj
                loss_k.backward()
                loss += loss_k.data
                del loss_k

        optimizer.step()
        train_loss += loss
        log_value('train_loss_batch',
                  loss.cpu().numpy(),
                  epoch * (config.train_size //
                           (config.batch_size * config.num_traj)) + batch_idx)

    mean_train_loss = train_loss / (config.train_size // (config.batch_size))
    log_value('train_loss', mean_train_loss.cpu().numpy(), epoch)
    imitate_net.eval()
    loss = Variable(torch.zeros(1)).cuda()
    metrics = {"cos": 0, "iou": 0, "cd": 0}
    IOU = 0
    COS = 0
    CD = 0
    for batch_idx in range(config.test_size // (config.batch_size)):
        for k in data_labels_paths.keys():
            data_, labels = next(test_gen_objs[k])
            one_hot_labels = prepare_input_op(labels, len(
                generator.unique_draw))
            one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
            data = Variable(torch.from_numpy(data_), volatile=True).cuda()
            labels = Variable(torch.from_numpy(labels)).cuda()
            test_outputs = imitate_net([data, one_hot_labels, k])
            loss += (losses_joint(test_outputs, labels, time_steps=k + 1) /
                     (k + 1)) / types_prog
            test_output = imitate_net.test([data, one_hot_labels, max_len])
            pred_images, correct_prog, pred_prog = parser.get_final_canvas(
                test_output, if_just_expressions=False, if_pred_images=True)
            target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
            iou = np.sum(np.logical_and(target_images, pred_images),
                         (1, 2)) / \
                  np.sum(np.logical_or(target_images, pred_images),
                         (1, 2))
            cos = cosine_similarity(target_images, pred_images)
            CD += np.sum(chamfer(target_images, pred_images))
            IOU += np.sum(iou)
            COS += np.sum(cos)

    metrics["iou"] = IOU / config.test_size
    metrics["cos"] = COS / config.test_size
    metrics["cd"] = CD / config.test_size

    log_value('test_iou', metrics["iou"], epoch)
    log_value('test_cosine', metrics["cos"], epoch)
    log_value('test_CD', metrics["cd"], epoch)

    test_losses = loss.data
    test_loss = test_losses.cpu().numpy() / (config.test_size //
                                             (config.batch_size))
    log_value('test_loss', test_loss, epoch)
    reduce_plat.reduce_on_plateu(metrics["cd"])
    logger.info("Epoch {}/{}=>  train_loss: {}, iou: {}, cd: {},"
                "test_mse: {}".format(epoch, config.epochs,
                                      mean_train_loss.cpu().numpy(), test_loss,
                                      metrics["iou"], metrics["cd"]))

    del test_losses, test_outputs
    if prev_test_cd > metrics["cd"]:
        logger.info("Saving the Model weights based on CD")
        print("Saving the Model weights based on CD", flush=True)
        torch.save(imitate_net.state_dict(),
                   "trained_models/{}.pth".format(model_name))
        prev_test_cd = metrics["cd"]
