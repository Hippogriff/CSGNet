"""
Training script specially designed for REINFORCE training.
"""

import logging
import numpy as np
import torch
import torch.optim as optim
import sys
import read_config
from tensorboard_logger import configure, log_value
from torch.autograd.variable import Variable
from src.Models.models import ImitateJoint
from src.Models.models import Encoder
from src.utils.generators.shapenet_generater import Generator
from src.utils.learn_utils import LearningRate
from src.utils.reinforce import Reinforce
from src.utils.train_utils import prepare_input_op

if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
    config = read_config.Config("config_cad.yml")

max_len = 15
reward = "chamfer"
power = 20
DATA_PATH = "data/cad/cad.h5"
model_name = config.model_path.format(config.mode)
config.write_config("log/configs/{}_config.json".format(model_name))
config.train_size = 10000
config.test_size = 3000
print(config.config)

# Setup Tensorboard logger
configure("log/tensorboard/{}".format(model_name), flush_secs=5)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(
    'log/logger/{}.log'.format(model_name), mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info(config.config)

# CNN encoder
encoder_net = Encoder(config.encoder_drop)
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

for param in imitate_net.parameters():
    param.requires_grad = True

for param in encoder_net.parameters():
    param.requires_grad = True
generator = Generator()
reinforce = Reinforce(unique_draws=unique_draw)

if config.optim == "sgd":
    optimizer = optim.SGD(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay,
        momentum=0.9,
        lr=config.lr,
        nesterov=False)
elif config.optim == "adam":
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

train_gen = generator.train_gen(
    batch_size=config.batch_size, path=DATA_PATH, if_augment=True, shuffle=True)
val_gen = generator.val_gen(
    batch_size=config.batch_size, path=DATA_PATH, if_augment=False)

prev_test_reward = 0
imitate_net.epsilon = config.eps
# Number of batches to accumulate before doing the gradient update.
num_traj = config.num_traj
training_reward_save = 0

for epoch in range(config.epochs):
    train_loss = 0
    total_reward = 0
    imitate_net.epsilon = 1
    imitate_net.train()
    for batch_idx in range(config.train_size // (config.batch_size)):
        optimizer.zero_grad()
        loss_sum = Variable(torch.zeros(1)).cuda().data
        Rs = np.zeros((config.batch_size, 1))
        for _ in range(num_traj):
            labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
            data_ = next(train_gen)
            one_hot_labels = prepare_input_op(labels, len(unique_draw))
            one_hot_labels = Variable(
                torch.from_numpy(one_hot_labels)).cuda()
            data = Variable(torch.from_numpy(data_), volatile=False).cuda()
            outputs, samples = imitate_net([data, one_hot_labels, max_len])
            R = reinforce.generate_rewards(
                samples,
                data_,
                time_steps=max_len,
                stack_size=max_len // 2 + 1,
                reward=reward,
                power=power)
            R = R[0]
            loss = reinforce.pg_loss_var(
                R, samples, outputs) / num_traj
            loss.backward()

            if reward == "chamfer":
                Rs = Rs + R
            elif reward == "iou":
                Rs = Rs + (R ** (1 / power))

            loss_sum += loss.data
        Rs = Rs / (num_traj)

        # Clip gradient to avoid explosions
        logger.info(torch.nn.utils.clip_grad_norm(imitate_net.parameters(), 10))
        # take gradient step only after having accumulating all gradients.
        optimizer.step()
        l = loss_sum
        train_loss += l
        log_value('train_loss_batch',
                  l.cpu().numpy(),
                  epoch * (config.train_size //
                           (config.batch_size)) + batch_idx)
        total_reward += np.mean(Rs)

        log_value('train_reward_batch', np.mean(Rs),
                  epoch * (config.train_size //
                           (config.batch_size)) + batch_idx)

    mean_train_loss = train_loss / (config.train_size // (config.batch_size))
    log_value('train_loss', mean_train_loss.cpu().numpy(), epoch)
    log_value('train_reward',
              total_reward / (config.train_size //
                              (config.batch_size)), epoch)

    test_losses = 0
    total_reward = 0
    imitate_net.eval()
    imitate_net.epsilon = 0
    for batch_idx in range(config.test_size // config.batch_size):
        loss = Variable(torch.zeros(1)).cuda()
        Rs = np.zeros((config.batch_size, 1))
        labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
        data_ = next(val_gen)
        one_hot_labels = prepare_input_op(labels, len(unique_draw))
        one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
        data = Variable(torch.from_numpy(data_), volatile=True).cuda()
        outputs, samples = imitate_net([data, one_hot_labels, max_len])
        R = reinforce.generate_rewards(
            samples,
            data_,
            time_steps=max_len,
            stack_size=max_len // 2 + 1,
            reward=reward,
            power=power)
        R = R[0]
        loss = loss + reinforce.pg_loss_var(R, samples, outputs)

        if reward == "chamfer":
            Rs = Rs + R

        elif reward == "iou":
            Rs = Rs + (R**(1 / power))

        test_losses += (loss.data)
        Rs = Rs
        total_reward += (np.mean(Rs))
    total_reward = total_reward / (config.test_size // config.batch_size)

    test_loss = test_losses.cpu().numpy() / (config.test_size // config.batch_size)
    log_value('test_loss', test_loss, epoch)
    log_value('test_reward', total_reward, epoch)
    if config.lr_sch:
        # Negative of the rewards should be minimized
        reduce_plat.reduce_on_plateu(-total_reward)

    logger.info("Epoch {}/{}=>  train_loss: {}, test_loss: {}, train_mse: {},"
                "test_mse: {}".format(epoch, config.epochs,
                                      mean_train_loss.cpu().numpy(), test_loss,
                                      1, 1))
    del test_losses
    
    # Save when test reward is increased
    if total_reward > prev_test_reward:
        logger.info("Saving the Model weights")
        torch.save(imitate_net.state_dict(),
                   "trained_models/{}.pth".format(model_name))
        prev_test_reward = total_reward
