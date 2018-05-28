""""
Contains small utility functions helpful in making the training interesting
"""
import h5py
import numpy as np
import torch
from torch.autograd.variable import Variable
from sklearn.preprocessing import normalize
from ..Models.models import validity
import cv2
from typing import List
import copy
from matplotlib import pyplot as plt
from typing import List

def pytorch_data(_generator, if_volatile=False):
    """Converts numpy tensor input data to pytorch tensors"""
    data_, labels = next(_generator)
    data = Variable(torch.from_numpy(data_))
    data.volatile = if_volatile
    data = data.cuda()
    labels = [Variable(torch.from_numpy(i)).cuda() for i in labels]
    return data, labels


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_draw_set(expressions):
    """
    Find a sorted set of draw type from the entire dataset. The idea is to
    use only the plausible position, scale and shape combinations and
    reject that are not possible because of the restrictions we have in
    the dataset.
    :param expressions: List containing entire dataset in the form of
    expressions.
    :return: unique_chunks: Unique sorted draw operations in the dataset.
    """
    shapes = ["s", "c", "t"]
    chunks = []
    for expression in expressions:
        for i, e in enumerate(expression):
            if e in shapes:
                index = i
                last_index = expression[index:].index(")")
                chunks.append(expression[index:index + last_index + 1])
    return list(set(chunks))


def prepare_input_op(arr, maxx):
    """
    This creates one-hot input for RNN that typically stores what happened
    in the immediate past. The first input to the RNN is
    start-of-the-sequence symbol. It is to be noted here that Input to the
    RNN in the form of one-hot contains one more element in comparison to
    the output from the RNN. This is because we don't want the
    start-of-the-sequence symbol in the output space of the program. arr
    here contains all the possible output that the RNN should/can produce,
    including stop-symbol. The stop symbol is represented by maxx-1 in the
    arr, but not to be bothered about here. Here, we make sure that the
    first input the RNN is start-of-the-sequence symbol by making maxx
    element of the array 1.
    :param arr: labels array
    :param maxx: maximum value in the labels
    :return:
    """
    s = arr.shape
    array = np.zeros((s[0], s[1] + 1, maxx + 1), dtype=np.float32)
    # Start of sequence token.
    array[:, 0, maxx] = 1
    for i in range(s[0]):
        for j in range(s[1]):
            array[i, j + 1, arr[i, j]] = 1
    return array


def to_one_hot(vector, max_category):
    """
    Converts a 1 d vector to one-hot representation
    :param vector:
    :param max_category:
    :return:
    """
    batch_size = vector.size()[0]
    vector_np = vector.data.cpu().numpy()
    array = np.zeros((batch_size, max_category))
    for j in range(batch_size):
        array[j, vector_np[j]] = 1
    return Variable(torch.from_numpy(array)).cuda()


def cosine_similarity(arr1, arr2):
    arr1 = np.reshape(arr1, (arr1.shape[0], -1))
    arr2 = np.reshape(arr2, (arr2.shape[0], -1))
    arr1 = normalize(arr1, "l2", 1)
    arr2 = normalize(arr2, "l2", 1)
    similarity = np.multiply(arr1, arr2)
    similarity = np.sum(similarity, 1)
    return similarity


def chamfer(images1, images2):
    """
    Chamfer distance on a minibatch, pairwise.
    :param images1: Bool Images of size (N, 64, 64). With background as zeros
    and forground as ones
    :param images2: Bool Images of size (N, 64, 64). With background as zeros
    and forground as ones
    :return: pairwise chamfer distance
    """
    # Convert in the opencv data format
    images1 = images1.astype(np.uint8)
    images1 = images1 * 255
    images2 = images2.astype(np.uint8)
    images2 = images2 * 255
    N = images1.shape[0]
    size = images1.shape[-1]

    D1 = np.zeros((N, size, size))
    E1 = np.zeros((N, size, size))

    D2 = np.zeros((N, size, size))
    E2 = np.zeros((N, size, size))
    summ1 = np.sum(images1, (1, 2))
    summ2 = np.sum(images2, (1, 2))

    # sum of completely filled image pixels
    filled_value = int(255 * size**2)
    defaulter_list = []
    for i in range(N):
        img1 = images1[i, :, :]
        img2 = images2[i, :, :]

        if (summ1[i] == 0) or (summ2[i] == 0) or (summ1[i] == filled_value) or (summ2[\
                i] == filled_value):
            # just to check whether any image is blank or completely filled
            defaulter_list.append(i)
            continue
        edges1 = cv2.Canny(img1, 1, 3)
        sum_edges = np.sum(edges1)
        if (sum_edges == 0) or (sum_edges == size**2):
            defaulter_list.append(i)
            continue
        dst1 = cv2.distanceTransform(
            ~edges1, distanceType=cv2.DIST_L2, maskSize=3)

        edges2 = cv2.Canny(img2, 1, 3)
        sum_edges = np.sum(edges2)
        if (sum_edges == 0) or (sum_edges == size**2):
            defaulter_list.append(i)
            continue

        dst2 = cv2.distanceTransform(
            ~edges2, distanceType=cv2.DIST_L2, maskSize=3)
        D1[i, :, :] = dst1
        D2[i, :, :] = dst2
        E1[i, :, :] = edges1
        E2[i, :, :] = edges2
    distances = np.sum(D1 * E2, (1, 2)) / (
        np.sum(E2, (1, 2)) + 1) + np.sum(D2 * E1, (1, 2)) / (np.sum(E1, (1, 2)) + 1)
    # TODO make it simpler
    distances = distances / 2.0
    # This is a fixed penalty for wrong programs
    distances[defaulter_list] = 16
    return distances


def image_from_expressions(parser, expressions):
    """This take a generic expression as input and returns the final image for
    this. The expressions need not be valid.
    :param parser: Object of the class parseModelOutput
    :expression: List of expression
    :return images: Last elements of the stack.
    """
    stacks = []
    for index, exp in enumerate(expressions):
        program = parser.Parser.parse(exp)
        if validity(program, len(program), len(program) - 1):
            pass
        else:
            stack = np.zeros((parser.canvas_shape[0], parser.canvas_shape[1]))
            stacks.append(stack)
            continue
        parser.sim.generate_stack(program)
        stack = parser.sim.stack_t
        stack = np.stack(stack, axis=0)[-1, 0, :, :]
        stacks.append(stack)
    images = np.stack(stacks, 0).astype(dtype=np.bool)
    return images


def stack_from_expressions(parser, expression: List):
    """This take a generic expression as input and returns the complete stack for
    this. The expressions need not be valid.
    :param parser: Object of the class parseModelOutput
    :expression: an expression
    :return stack: Stack from execution of the expression.
    """
    program = parser.Parser.parse(expression)
    if validity(program, len(program), len(program) - 1):
        pass
    else:
        stack = np.zeros((parser.canvas_shape[0], parser.canvas_shape[1]))
    parser.sim.generate_stack(program)
    stack = parser.sim.stack_t
    stack = np.stack(stack, axis=0)
    return stack


def plot_stack(stack):
    import matplotlib.pyplot as plt
    plt.ioff()
    T, S = stack.shape[0], stack.shape[1]
    f, ar = plt.subplots(
        stack.shape[0], stack.shape[1], squeeze=False, figsize=(S, T))
    for j in range(T):
        for k in range(S):
            ar[j, k].imshow(stack[j, k, :, :], cmap="Greys_r")
            ar[j, k].axis("off")


def summary(model):
    """
    given the model, it returns a summary of learnable parameters
    :param model: Pytorch nn model
    :return: summary
    """
    state_dict = model.state_dict()
    total_param = 0
    num_parameters = {}
    for k in state_dict.keys():
        num_parameters[k] = np.prod([i for i in state_dict[k].size()])
        total_param += num_parameters[k]
    return num_parameters, total_param


def beams_parser(all_beams, batch_size, beam_width=5):
    # all_beams = [all_beams[k].data.numpy() for k in all_beams.keys()]
    all_expression = {}
    W = beam_width
    T = len(all_beams)
    for batch in range(batch_size):
        all_expression[batch] = []
        for w in range(W):
            temp = []
            parent = w
            for t in range(T - 1, -1, -1):
                temp.append(all_beams[t]["index"][batch, parent].data.cpu()
                            .numpy()[0])
                parent = all_beams[t]["parent"][batch, parent]
            temp = temp[::-1]
            all_expression[batch].append(np.array(temp))
        all_expression[batch] = np.squeeze(np.array(all_expression[batch]))
    return all_expression


def valid_permutations(prog, permutations=[], stack=[], start=False):
    """
    Takes the prog, and returns valid permutation such that the final output
    shape remains same. Mainly permuate the operands in union and intersection
    open"""
    for index, p in enumerate(prog):
        if p["type"] == "draw":
            stack.append(p["value"])

        elif p["type"] == "op" and (p["value"] == "+" or p["value"] == "*"):
            second = stack.pop()
            first = stack.pop()

            first_stack = copy.deepcopy(stack)
            first_stack.append(first + second + p["value"])

            second_stack = copy.deepcopy(stack)
            second_stack.append(second + first + p["value"])

            program1 = valid_permutations(prog[index + 1:], permutations, first_stack, start=False)
            program2 = valid_permutations(prog[index + 1:], permutations, second_stack, start=False)
            permutations.append(program1)
            permutations.append(program2)

            stack.append(first + second + p["value"])

        elif p["type"] == "op" and p["value"] == "-":
            second = stack.pop()
            first = stack.pop()
            stack.append(first + second + p["value"])
            if index == len(prog) - 1:
                permutations.append(copy.deepcopy(stack[0]))
    if start:
        return list(permutations)
    else:
        return stack[0]


def plotall(images: List, cmap="Greys_r"):
    """
    Awesome function to plot figures in list of list fashion.
    Every list inside the list, is assumed to be drawn in one row.
    :param images: List of list containing images
    :param cmap: color map to be used for all images
    :return: List of figures.
    """
    figures = []
    num_rows = len(images)
    for r in range(num_rows):
        cols = len(images[r])
        f, a = plt.subplots(1, cols)
        for c in range(cols):
            a[c].imshow(images[r][c], cmap=cmap)
            a[c].title.set_text("{}".format(c))
            a[c].axis("off")
            a[c].grid("off")
        figures.append(f)
    return figures