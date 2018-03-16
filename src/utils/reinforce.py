"""
This file defines helper classes to implement REINFORCE algorithm.
"""
import numpy as np
import torch
from torch.autograd.variable import Variable
from .generators.mixed_len_generator import Parser
from ..Models.models import ParseModelOutput, validity
from ..utils.train_utils import chamfer


class Reinforce:
    def __init__(self,
                 unique_draws,
                 canvas_shape=[64, 64],
                 rolling_average_const=0.7):
        """
        This class defines does all the work to create the final canvas from
        the prediction of RNN and also defines the loss to back-propagate in.
        :param canvas_shape: Canvas shape
        :param rolling_average_const: constant to be used in creating running average 
        baseline.
        :param stack_size: Maximum size of Stack required
        :param time_steps: max len of program
        :param unique_draws: Number of unique_draws in the dataset
        penalize longer predicted programs in variable length case training.
        """
        self.canvas_shape = canvas_shape
        self.unique_draws = unique_draws
        self.max_reward = Variable(torch.zeros(1)).cuda()
        self.rolling_baseline = Variable(torch.zeros(1)).cuda()
        self.alpha_baseline = rolling_average_const

    def generate_rewards(self,
                         samples,
                         data,
                         time_steps,
                         stack_size,
                         reward="chamfer",
                         if_stack_calculated=False,
                         pred_images=None,
                         power=20):
        """
        This function will parse the predictions of RNN into final canvas,
        and define the rewards for individual examples.
        :param samples: Sampled actions from output of RNN
        :param labels: GRound truth labels
        :param power: returns R ** power, to give more emphasis on higher
        powers.
        """
        if not if_stack_calculated:
            parser = ParseModelOutput(self.unique_draws, stack_size,
                                      time_steps, [64, 64])
            samples = torch.cat(samples, 1)
            expressions = parser.labels2exps(samples, time_steps)

            # Drain all dollars down the toilet!
            for index, exp in enumerate(expressions):
                expressions[index] = exp.split("$")[0]

            pred_images = []
            for index, exp in enumerate(expressions):
                program = parser.Parser.parse(exp)
                if validity(program, len(program), len(program) - 1):
                    stack = parser.expression2stack([exp])
                    pred_images.append(stack[-1, -1, 0, :, :])
                else:
                    pred_images.append(np.zeros(self.canvas_shape))
            pred_images = np.stack(pred_images, 0).astype(dtype=np.bool)
        else:
            # in stack_CNN we calculate it in the forward pass
            # convert the torch tensor to numpy
            pred_images = pred_images[-1, :, 0, :, :].data.cpu().numpy()
        target_images = data[-1, :, 0, :, :].astype(dtype=np.bool)
        image_size = target_images.shape[-1]

        if reward == "iou":
            R = np.sum(np.logical_and(target_images, pred_images), (1, 2)) / \
                (np.sum(np.logical_or(target_images, pred_images), (1,
                                                                    2)) + 1.0)
            R = R**power

        elif reward == "chamfer":
            distance = chamfer(target_images, pred_images)
            # normalize the distance by the diagonal of the image
            R = (1.0 - distance / image_size / (2**0.5))
            R = np.clip(R, a_min=0.0)
            R[R > 1.0] = 0
            R = R**power

        R = np.expand_dims(R, 1).astype(dtype=np.float32)
        if (reward == "chamfer"):
            if if_stack_calculated:
                return R, samples, pred_images, 0, distance
            else:
                return R, samples, pred_images, expressions, distance

        elif reward == "iou":
            if if_stack_calculated:
                return R, samples, pred_images, 0
            else:
                return R, samples, pred_images, expressions

    def pg_loss_var(self, R, samples, probs):
        """
        Reinforce loss for variable length program setting, where we stop at maximum
        length programs or when stop symbol is encountered. The baseline is calculated
        using rolling average baseline.
        :return: 
        :param R: Rewards for the minibatch
        :param samples: Sampled actions for minibatch at every time step
        :param probs: Probability corresponding to every sampled action.
        :return loss: reinforce loss
        """
        batch_size = R.shape[0]
        R = Variable(torch.from_numpy(R)).cuda()
        T = len(samples)
        samples = [s.data.cpu().numpy() for s in samples]

        Parse_program = Parser()
        parser = ParseModelOutput(self.unique_draws, T // 2 + 1, T, [64, 64])
        samples_ = np.concatenate(samples, 1)
        expressions = parser.labels2exps(samples_, T)

        for index, exp in enumerate(expressions):
            expressions[index] = exp.split("$")[0]

        # Find the length of programs. If len of program is lesser than T,
        # then we include stop symbol in len_programs to backprop through
        # stop symbol.
        len_programs = np.zeros((batch_size), dtype=np.int32)
        for index, exp in enumerate(expressions):
            p = Parse_program.parse(exp)
            if len(p) == T:
                len_programs[index] = len(p)
            else:
                # Include one more step for stop symbol.
                try:
                    len_programs[index] = len(p) + 1
                except:
                    print(len(expressions), batch_size, samples_.shape)
        self.rolling_baseline = self.alpha_baseline * self.rolling_baseline + (1 - self.alpha_baseline) * torch.mean(R)
        baseline = self.rolling_baseline.view(1, 1).repeat(batch_size, 1)
        baseline = baseline.detach()
        advantage = R - baseline

        temp = []
        for i in range(batch_size):
            neg_log_prob = Variable(torch.zeros(1)).cuda()
            # Only summing the probs before stop symbol
            for j in range(len_programs[i]):
                neg_log_prob = neg_log_prob + probs[j][i, samples[j][i, 0]]
            temp.append(neg_log_prob)

        loss = -torch.cat(temp).view(batch_size, 1)
        loss = loss.mul(advantage)
        loss = torch.mean(loss)
        return loss
