"""
Defines Neural Networks
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
from ..utils.generators.mixed_len_generator import Parser, \
    SimulateStack
from typing import List


class Encoder(nn.Module):
    def __init__(self, dropout=0.2):
        """
        Encoder for 2D CSGNet.
        :param dropout: dropout
        """
        super(Encoder, self).__init__()
        self.p = dropout
        self.conv1 = nn.Conv2d(1, 8, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, 3, padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.drop = nn.Dropout(dropout)

    def encode(self, x):
        x = F.max_pool2d(self.drop(F.relu(self.conv1(x))), (2, 2))
        x = F.max_pool2d(self.drop(F.relu(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ImitateJoint(nn.Module):
    def __init__(self,
                 hd_sz,
                 input_size,
                 encoder,
                 mode,
                 num_layers=1,
                 time_steps=3,
                 num_draws=None,
                 canvas_shape=[64, 64],
                 dropout=0.5):
        """
        Defines RNN structure that takes features encoded by CNN and produces program
        instructions at every time step.
        :param num_draws: Total number of tokens present in the dataset or total number of operations to be predicted + a stop symbol = 400
        :param canvas_shape: canvas shape
        :param dropout: dropout
        :param hd_sz: rnn hidden size
        :param input_size: input_size (CNN feature size) to rnn
        :param encoder: Feature extractor network object
        :param mode: Mode of training, RNN, BDRNN or something else
        :param num_layers: Number of layers to rnn
        :param time_steps: max length of program
        """
        super(ImitateJoint, self).__init__()
        self.hd_sz = hd_sz
        self.in_sz = input_size
        self.num_layers = num_layers
        self.encoder = encoder
        self.time_steps = time_steps
        self.mode = mode
        self.canvas_shape = canvas_shape
        self.num_draws = num_draws

        # Dense layer to project input ops(labels) to input of rnn
        self.input_op_sz = 128
        self.dense_input_op = nn.Linear(
            in_features=self.num_draws + 1, out_features=self.input_op_sz)

        self.rnn = nn.GRU(
            input_size=self.in_sz + self.input_op_sz,
            hidden_size=self.hd_sz,
            num_layers=self.num_layers,
            batch_first=False)

        # adapt logsoftmax and softmax for different versions of pytorch
        self.pytorch_version = torch.__version__[2]
        if self.pytorch_version == "1":
            self.logsoftmax = nn.LogSoftmax()
            self.softmax = nn.Softmax()

        elif self.pytorch_version == "3":
            self.logsoftmax = nn.LogSoftmax(1)
            self.softmax = nn.Softmax(1)
        self.dense_fc_1 = nn.Linear(
            in_features=self.hd_sz, out_features=self.hd_sz)
        self.dense_output = nn.Linear(
            in_features=self.hd_sz, out_features=(self.num_draws))
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x: List):
        """
        Forward pass for  all architecture
        :param x: Has different meaning with different mode of training
        :return:
        """

        if self.mode == 1:
            '''
            Variable length training. This mode runs for one
            more than the length of program for producing stop symbol. Note
            that there is no padding as is done in traditional RNN for
            variable length programs. This is done mainly because of computational
            efficiency of forward pass, that is, each batch contains only
            programs of same length and losses from all batches of
            different time-lengths are combined to compute gradient and
            update in the network. This ensures that every update of the
            network has equal contribution coming from programs of different lengths.
            Training is done using the script train_synthetic.py
            '''
            data, input_op, program_len = x

            assert data.size()[0] == program_len + 1, "Incorrect stack size!!"
            batch_size = data.size()[1]
            h = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
            x_f = self.encoder.encode(data[-1, :, 0:1, :, :])
            x_f = x_f.view(1, batch_size, self.in_sz)
            outputs = []
            for timestep in range(0, program_len + 1):
                # X_f is always input to the RNN at every time step
                # along with previous predicted label
                input_op_rnn = self.relu(
                    self.dense_input_op(input_op[:, timestep, :]))
                input_op_rnn = input_op_rnn.view(1, batch_size,
                                                 self.input_op_sz)
                input = torch.cat((self.drop(x_f), input_op_rnn), 2)
                h, _ = self.rnn(input, h)
                hd = self.relu(self.dense_fc_1(self.drop(h[0])))
                output = self.logsoftmax(self.dense_output(self.drop(hd)))
                outputs.append(output)
            return outputs

        elif self.mode == 2:
            '''Train variable length RL'''
            # program length in this case is the maximum time step that RNN runs
            data, input_op, program_len = x
            batch_size = data.size()[1]
            h = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
            x_f = self.encoder.encode(data[-1, :, 0:1, :, :])
            x_f = x_f.view(1, batch_size, self.in_sz)
            outputs = []
            samples = []
            temp_input_op = input_op[:, 0, :]
            for timestep in range(0, program_len):
                # X_f is the input to the RNN at every time step along with previous
                # predicted label
                input_op_rnn = self.relu(self.dense_input_op(temp_input_op))
                input_op_rnn = input_op_rnn.view(1, batch_size,
                                                 self.input_op_sz)
                input = torch.cat((x_f, input_op_rnn), 2)
                h, _ = self.rnn(input, h)
                hd = self.relu(self.dense_fc_1(self.drop(h[0])))
                dense_output = self.dense_output(self.drop(hd))
                output = self.logsoftmax(dense_output)
                # output for loss, these are log-probabs
                outputs.append(output)

                output_probs = self.softmax(dense_output)
                # Get samples from output probabs based on epsilon greedy way
                # Epsilon will be reduced to 0 gradually following some schedule
                if np.random.rand() < self.epsilon:
                    # This is during training
                    sample = torch.multinomial(output_probs, 1)
                else:
                    # This is during testing
                    if self.pytorch_version == "1":
                        sample = torch.max(output_probs, 1)[1]
                    elif self.pytorch_version == "3":
                        sample = torch.max(output_probs, 1)[1].view(
                            batch_size, 1)

                # Stopping the gradient to flow backward from samples
                sample = sample.detach()
                samples.append(sample)

                # Create next input to the RNN from the sampled instructions
                arr = Variable(
                    torch.zeros(batch_size, self.num_draws + 1).scatter_(
                        1, sample.data.cpu(), 1.0)).cuda()
                arr = arr.detach()
                temp_input_op = arr
            return [outputs, samples]
        else:
            assert False, "Incorrect mode!!"

    def test(self, data: List):
        """
        Testing different modes of network
        :param data: Has different meaning for different modes
        :param draw_uniques:
        :return:
        """
        if self.mode == 1:
            data, input_op, program_len = data
            batch_size = data.size()[1]
            h = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
            x_f = self.encoder.encode(data[-1, :, 0:1, :, :])
            x_f = x_f.view(1, batch_size, self.in_sz)
            outputs = []
            last_output = input_op[:, 0, :]
            for timestep in range(0, program_len):
                # X_f is always input to the network at every time step
                # along with previous predicted label
                input_op_rnn = self.relu(self.dense_input_op(last_output))
                input_op_rnn = input_op_rnn.view(1, batch_size,
                                                 self.input_op_sz)
                input = torch.cat((self.drop(x_f), input_op_rnn), 2)
                h, _ = self.rnn(input, h)
                hd = self.relu(self.dense_fc_1(self.drop(h[0])))
                output = self.logsoftmax(self.dense_output(self.drop(hd)))
                if self.pytorch_version == "1":
                    next_input_op = torch.max(output, 1)[1]
                elif self.pytorch_version == "3":
                    next_input_op = torch.max(output, 1)[1].view(batch_size, 1)
                arr = Variable(
                    torch.zeros(batch_size, self.num_draws + 1).scatter_(
                        1, next_input_op.data.cpu(), 1.0)).cuda()

                last_output = arr
                outputs.append(output)
            return outputs

        else:
            assert False, "Incorrect mode!!"

    def beam_search(self, data: List, w: int, max_time: int):
        """
        Implements beam search for different models.
        :param data: Input data
        :param w: beam width
        :param max_time: Maximum length till the program has to be generated
        :return all_beams: all beams to find out the indices of all the
        """
        data, input_op = data

        # Beam, dictionary, with elements as list. Each element of list
        # containing index of the selected output and the corresponding
        # probability.
        batch_size = data.size()[1]
        h = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
        # Last beams' data
        B = {0: {"input": input_op, "h": h}, 1: None}
        next_B = {}
        x_f = self.encoder.encode(data[-1, :, 0:1, :, :])
        x_f = x_f.view(1, batch_size, self.in_sz)
        # List to store the probs of last time step
        prev_output_prob = [
            Variable(torch.ones(batch_size, self.num_draws)).cuda()
        ]
        all_beams = []
        all_inputs = []
        for timestep in range(0, max_time):
            outputs = []
            for b in range(w):
                if not B[b]:
                    break
                input_op = B[b]["input"]

                h = B[b]["h"]
                input_op_rnn = self.relu(
                    self.dense_input_op(input_op[:, 0, :]))
                input_op_rnn = input_op_rnn.view(1, batch_size,
                                                 self.input_op_sz)
                input = torch.cat((x_f, input_op_rnn), 2)
                h, _ = self.rnn(input, h)
                hd = self.relu(self.dense_fc_1(self.drop(h[0])))
                dense_output = self.dense_output(self.drop(hd))
                output = self.logsoftmax(dense_output)
                # Element wise multiply by previous probabs
                output = torch.nn.Softmax(1)(output)

                output = output * prev_output_prob[b]
                outputs.append(output)
                next_B[b] = {}
                next_B[b]["h"] = h
            if len(outputs) == 1:
                outputs = outputs[0]
            else:
                outputs = torch.cat(outputs, 1)

            next_beams_index = torch.topk(outputs, w, 1, sorted=True)[1]
            next_beams_prob = torch.topk(outputs, w, 1, sorted=True)[0]
            # print (next_beams_prob)
            current_beams = {
                "parent":
                next_beams_index.data.cpu().numpy() // (self.num_draws),
                "index": next_beams_index % (self.num_draws)
            }
            # print (next_beams_index % (self.num_draws))
            next_beams_index %= (self.num_draws)
            all_beams.append(current_beams)

            # Update previous output probabilities
            temp = Variable(torch.zeros(batch_size, 1)).cuda()
            prev_output_prob = []
            for i in range(w):
                for index in range(batch_size):
                    temp[index, 0] = next_beams_prob[index, i]
                prev_output_prob.append(temp.repeat(1, self.num_draws))
            # hidden state for next step
            B = {}
            for i in range(w):
                B[i] = {}
                temp = Variable(torch.zeros(h.size())).cuda()
                for j in range(batch_size):
                    temp[0, j, :] = next_B[current_beams["parent"][j, i]]["h"][
                        0, j, :]
                B[i]["h"] = temp

            # one_hot for input to the next step
            for i in range(w):
                arr = Variable(
                    torch.zeros(batch_size, self.num_draws + 1).scatter_(
                        1, next_beams_index[:, i:i + 1].data.cpu(),
                        1.0)).cuda()
                B[i]["input"] = arr.unsqueeze(1)
            all_inputs.append(B)

        return all_beams, next_beams_prob, all_inputs


class ParseModelOutput:
    def __init__(self, unique_draws: List, stack_size: int, steps: int,
                 canvas_shape: List):
        """
        This class parses complete output from the network which are in joint
        fashion. This class can be used to generate final canvas and
        expressions.
        :param unique_draws: Unique draw/op operations in the current dataset
        :param stack_size: Stack size
        :param steps: Number of steps in the program
        :param canvas_shape: Shape of the canvases
        """
        self.canvas_shape = canvas_shape
        self.stack_size = stack_size
        self.steps = steps
        self.Parser = Parser()
        self.sim = SimulateStack(self.stack_size, self.canvas_shape)
        self.unique_draws = unique_draws
        self.pytorch_version = torch.__version__[2]

    def get_final_canvas(self,
                         outputs: List,
                         if_just_expressions=False,
                         if_pred_images=False):
        """
        Takes the raw output from the network and returns the predicted
        canvas. The steps involve parsing the outputs into expressions,
        decoding expressions, and finally producing the canvas using
        intermediate stacks.
        :param if_just_expressions: If only expression is required than we
        just return the function after calculating expressions
        :param outputs: List, each element correspond to the output from the
        network
        :return: stack: Predicted final stack for correct programs
        :return: correct_programs: Indices of correct programs
        """
        batch_size = outputs[0].size()[0]

        # Initialize empty expression string, len equal to batch_size
        correct_programs = []
        expressions = [""] * batch_size
        labels = [
            torch.max(outputs[i], 1)[1].data.cpu().numpy()
            for i in range(self.steps)
        ]

        if self.pytorch_version == "1":
            for j in range(batch_size):
                for i in range(self.steps):
                    expressions[j] += self.unique_draws[labels[i][j, 0]]
        elif self.pytorch_version == "3":
            for j in range(batch_size):
                for i in range(self.steps):
                    expressions[j] += self.unique_draws[labels[i][j]]

        # Remove the stop symbol and later part of the expression
        for index, exp in enumerate(expressions):
            expressions[index] = exp.split("$")[0]
        if if_just_expressions:
            return expressions
        stacks = []
        for index, exp in enumerate(expressions):
            program = self.Parser.parse(exp)
            if validity(program, len(program), len(program) - 1):
                correct_programs.append(index)
            else:
                if if_pred_images:
                    # if you just want final predicted image
                    stack = np.zeros((self.canvas_shape[0],
                                      self.canvas_shape[1]))
                else:
                    stack = np.zeros(
                        (self.steps + 1, self.stack_size, self.canvas_shape[0],
                         self.canvas_shape[1]))
                stacks.append(stack)
                continue
                # Check the validity of the expressions

            self.sim.generate_stack(program)
            stack = self.sim.stack_t
            stack = np.stack(stack, axis=0)
            if if_pred_images:
                stacks.append(stack[-1, 0, :, :])
            else:
                stacks.append(stack)
        if len(stacks) == 0:
            return None
        if if_pred_images:
            stacks = np.stack(stacks, 0).astype(dtype=np.bool)
        else:
            stacks = np.stack(stacks, 1).astype(dtype=np.bool)
        return stacks, correct_programs, expressions

    def expression2stack(self, expressions: List):
        """Assuming all the expression are correct and coming from
        groundtruth labels. Helpful in visualization of programs
        :param expressions: List, each element an expression of program
        """
        stacks = []
        for index, exp in enumerate(expressions):
            program = self.Parser.parse(exp)
            self.sim.generate_stack(program)
            stack = self.sim.stack_t
            stack = np.stack(stack, axis=0)
            stacks.append(stack)
        stacks = np.stack(stacks, 1).astype(dtype=np.float32)
        return stacks

    def labels2exps(self, labels: np.ndarray, steps: int):
        """
        Assuming grountruth labels, we want to find expressions for them
        :param labels: Grounth labels batch_size x time_steps
        :return: expressions: Expressions corresponding to labels
        """
        if isinstance(labels, np.ndarray):
            batch_size = labels.shape[0]
        else:
            batch_size = labels.size()[0]
            labels = labels.data.cpu().numpy()
        # Initialize empty expression string, len equal to batch_size
        correct_programs = []
        expressions = [""] * batch_size
        for j in range(batch_size):
            for i in range(steps):
                expressions[j] += self.unique_draws[labels[j, i]]
        return expressions


def validity(program: List, max_time: int, timestep: int):
    """
    Checks the validity of the program. In short implements a pushdown automaton that accepts valid strings.
    :param program: List of dictionary containing program type and elements
    :param max_time: Max allowed length of program
    :param timestep: Current timestep of the program, or in a sense length of
    program
    # at evey index
    :return:
    """
    num_draws = 0
    num_ops = 0
    for i, p in enumerate(program):
        if p["type"] == "draw":
            # draw a shape on canvas kind of operation
            num_draws += 1
        elif p["type"] == "op":
            # +, *, - kind of operation
            num_ops += 1
        elif p["type"] == "stop":
            # Stop symbol, no need to process further
            if num_draws > ((len(program) - 1) // 2 + 1):
                return False
            if not (num_draws > num_ops):
                return False
            return (num_draws - 1) == num_ops

        if num_draws <= num_ops:
            # condition where number of operands are lesser than 2
            return False
        if num_draws > (max_time // 2 + 1):
            # condition for stack over flow
            return False
    if (max_time - 1) == timestep:
        return (num_draws - 1) == num_ops
    return True
