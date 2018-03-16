# Training and testing dataset generator

import string
from typing import List
import numpy as np
from skimage import draw
from ...utils.image_utils import ImageDataGenerator

datagen = ImageDataGenerator(
    width_shift_range=3 / 64,
    height_shift_range=3 / 64,
    zoom_range=[1 - 2 / 64, 1 + 2 / 64],
    data_format="channels_first")


class MixedGenerateData:
    def __init__(self,
                 data_labels_paths,
                 batch_size=32,
                 train_size=4000,
                 test_size=1000,
                 stack_size=2,
                 canvas_shape=[64, 64]):
        """
        Primary function of this generator is to generate variable length
        dataset for training variable length programs. It creates a generator
        object for every length of program that you want to generate. This
        process allow finer control of every batch that you feed into the
        network. This class can also be used in fixed length training.
        :param stack_size: size of the stack
        :param canvas_shape: canvas shape
        :param time_steps: Max time steps for generated programs
        :param num_operations: Number of possible operations
        :param image_path: path to images
        :param data_labels_paths: dictionary containing paths for different
        lengths programs
        :param batch_size: batch_size
        :param train_size: number of training instances
        :param test_size: number of test instances
        """
        self.batch_size = batch_size
        self.canvas_shape = canvas_shape

        self.programs = {}
        self.data_labels_path = data_labels_paths
        for index in data_labels_paths.keys():
            with open(data_labels_paths[index]) as data_file:
                self.programs[index] = data_file.readlines()
        all_programs = []
        # print (self.programs)
        for k in self.programs.keys():
            all_programs += self.programs[k]

        self.unique_draw = self.get_draw_set(all_programs)
        self.unique_draw.sort()
        # Append ops in the end and the last one is for stop symbol
        self.unique_draw += ["+", "*", "-", "$"]

    def get_draw_set(self, expressions):
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

    def parse(self, expression):
        """
        NOTE: This method is different from parse method in Parser class
        Takes an expression, returns a serial program
        :param expression: program expression in postfix notation
        :return program:
        """
        self.shape_types = ["c", "s", "t"]
        self.op = ["*", "+", "-"]
        program = []
        for index, value in enumerate(expression):
            if value in self.shape_types:
                program.append({})
                program[-1]["type"] = "draw"

                # find where the parenthesis closes
                close_paren = expression[index:].index(")") + index
                program[-1]["value"] = expression[index:close_paren + 1]
            elif value in self.op:
                program.append({})
                program[-1]["type"] = "op"
                program[-1]["value"] = value
            else:
                pass
        return program

    def get_train_data(self,
                       batch_size: int,
                       program_len: int,
                       num_train_images=None,
                       stack_size=None,
                       jitter_program=False,
                       if_randomize=True):
        """
        This is a special generator that can generate dataset for any length.
        Since, this is a generator, you need to make a generator different object for
        different program length and use them as required. Training data is shuffled
        once per epoch.
        :param num_train_images: Number of training examples from a particular program
        length.
        :param jitter_program: whether to jitter the final output or not
        :param if_randomize: If randomize the training dataset
        :param batch_size: batch size for the current program
        :param program_len: which program length dataset to sample
        :param stack_size: program_len  // 2 + 1
        :return data: image and label pair for a minibatch
        """
        # The last label corresponds to the stop symbol and the first one to
        labels = np.zeros((batch_size, program_len + 1), dtype=np.int64)
        if stack_size == None:
            sim = SimulateStack(program_len // 2 + 1, self.canvas_shape)
        else:
            sim = SimulateStack(stack_size, self.canvas_shape)
        parser = Parser()
        while True:
            # Random things to select random indices
            IDS = np.arange(num_train_images)
            if if_randomize:
                np.random.shuffle(IDS)
            for rand_id in range(0, num_train_images - batch_size, batch_size):
                image_ids = IDS[rand_id:rand_id + batch_size]
                stacks = []
                for index, value in enumerate(image_ids):
                    program = parser.parse(self.programs[program_len][value])
                    sim.generate_stack(program)
                    stack = sim.stack_t
                    stack = np.stack(stack, axis=0)
                    stacks.append(stack)
                stacks = np.stack(stacks, 1).astype(dtype=np.float32)

                for index, value in enumerate(image_ids):
                    # Get the current program
                    exp = self.programs[program_len][value]

                    program = self.parse(exp)
                    for j in range(program_len):
                        labels[index, j] = self.unique_draw.index(
                            program[j]["value"])

                    # Stop symbol
                    labels[:, -1] = len(self.unique_draw) - 1
                if jitter_program:
                    stacks[-1, :, 0:1, :, :] = next(
                        datagen.flow(
                            stacks[-1, :, 0:1, :, :],
                            batch_size=self.batch_size,
                            shuffle=False))
                yield [stacks.copy(), labels]

    def get_test_data(self,
                      batch_size: int,
                      program_len: int,
                      if_randomize=False,
                      num_train_images=None,
                      num_test_images=None,
                      stack_size=None,
                      jitter_program=False):
        """
        This is a special generator that can generate dataset for any length.
        Since, this is a generator, you need to make a generator different object for
        different program length and use them as required.
        :param num_train_images: Number of training examples from a particular program
        length.
        :param num_test_images: Number of Testing examples from a particular program
        length.
        :param jitter_program: Whether to jitter programs or not
        :param batch_size: batch size of dataset to yielded
        :param program_len: length of program to be generated
        :param stack_size: program_len // 2 + 1
        :param if_randomize: if randomize
        :return:
        """
        labels = np.zeros((batch_size, program_len + 1), dtype=np.int64)
        if stack_size == None:
            sim = SimulateStack(program_len // 2 + 1, self.canvas_shape)
        else:
            sim = SimulateStack(stack_size, self.canvas_shape)
        parser = Parser()
        while True:
            IDS = np.arange(num_train_images,
                            num_test_images + num_train_images)
            if if_randomize:
                np.random.shuffle(IDS)
            for rand_id in range(0, num_test_images - batch_size, batch_size):
                image_ids = IDS[rand_id:rand_id + batch_size]
                stacks = []
                for index, value in enumerate(image_ids):
                    program = parser.parse(self.programs[program_len][value])
                    # if jitter_program:
                    #     program = self.jitter_program(program)
                    sim.generate_stack(program)
                    stack = sim.stack_t
                    stack = np.stack(stack, axis=0)
                    stacks.append(stack)
                stacks = np.stack(stacks, 1).astype(dtype=np.float32)

                for index, value in enumerate(image_ids):
                    # Get the current program
                    exp = self.programs[program_len][value]
                    program = self.parse(exp)
                    for j in range(program_len):
                        labels[index, j] = self.unique_draw.index(
                            program[j]["value"])
                    # Stop symbol
                    labels[:, -1] = len(self.unique_draw) - 1
                if jitter_program:
                    stacks[-1, :, 0:1, :, :] = next(
                        datagen.flow(
                            stacks[-1, :, 0:1, :, :],
                            batch_size=self.batch_size,
                            shuffle=False))
                yield [stacks.copy(), labels]


class Draw:
    def __init__(self, canvas_shape=[64, 64]):
        """
        Helper function for drawing the canvases.
        :param canvas_shape: shape of the canvas on which to draw objects
        """
        self.canvas_shape = canvas_shape

    def draw_circle(self, center: List, radius: int):
        """
        Draw a circle
        :param center: center of the circle
        :param radius: radius of the circle
        :return:
        """
        arr = np.zeros(self.canvas_shape, dtype=bool)
        xp = [center[0] + radius, center[0], center[0], center[0] - radius]
        yp = [center[1], center[1] + radius, center[1] - radius, center[1]]

        rr, cc = draw.circle(*center, radius=radius, shape=self.canvas_shape)
        arr[cc, rr] = True
        return arr

    def draw_triangle(self, center: List, length: int):
        """
        Draw a triangle
        :param center: center of the triangle
        :param radius: radius of the triangle
        :return:
        """
        arr = np.zeros(self.canvas_shape, dtype=bool)
        length = 1.732 * length
        rows = [
            int(center[1] + length / (2 * 1.732)),
            int(center[1] + length / (2 * 1.732)),
            int(center[1] - length / 1.732)
        ]
        cols = [
            int(center[0] - length / 2.0),
            int(center[0] + length / 2.0), center[0]
        ]

        rr_inner, cc_inner = draw.polygon(rows, cols, shape=self.canvas_shape)
        rr_boundary, cc_boundary = draw.polygon_perimeter(
            rows, cols, shape=self.canvas_shape)

        ROWS = np.concatenate((rr_inner, rr_boundary))
        COLS = np.concatenate((cc_inner, cc_boundary))
        arr[ROWS, COLS] = True
        return arr

    def draw_square(self, center: list, length: int):
        """
        Draw a square
        :param center: center of square
        :param length: length of square
        :return:
        """
        arr = np.zeros(self.canvas_shape, dtype=bool)
        length *= 1.412
        # generate the row vertices
        rows = np.array([
            int(center[0] - length / 2.0),
            int(center[0] + length / 2.0),
            int(center[0] + length / 2.0),
            int(center[0] - length / 2.0)
        ])
        cols = np.array([
            int(center[1] + length / 2.0),
            int(center[1] + length / 2.0),
            int(center[1] - length / 2.0),
            int(center[1] - length / 2.0)
        ])

        # generate the col vertices
        rr_inner, cc_inner = draw.polygon(rows, cols, shape=self.canvas_shape)
        rr_boundary, cc_boundary = draw.polygon_perimeter(
            rows, cols, shape=self.canvas_shape)

        ROWS = np.concatenate((rr_inner, rr_boundary))
        COLS = np.concatenate((cc_inner, cc_boundary))

        arr[COLS, ROWS] = True
        return arr


class CustomStack(object):
    """Simple Stack implements in the form of array"""

    def __init__(self, max_len, canvas_shape):
        _shape = [max_len] + canvas_shape
        self.max_len = max_len
        self.canvas_shape = canvas_shape
        self.items = np.zeros(_shape, dtype=bool)
        self.pointer = -1
        self.max_len = max_len

    def push(self, item):
        if self.pointer > self.max_len - 1:
            assert False, "{} exceeds max len for stack!!".format(self.pointer)
        self.pointer += 1
        self.items[self.pointer, :, :] = item.copy()

    def pop(self):
        if self.pointer <= -1:
            assert False, "below min len of stack!!"
        item = self.items[self.pointer, :, :].copy()
        self.items[self.pointer, :, :] = np.zeros(
            self.canvas_shape, dtype=bool)
        self.pointer -= 1
        return item

    def clear(self):
        """Re-initializes the stack"""
        self.pointer = -1
        _shape = [self.max_len] + self.canvas_shape
        self.items = np.zeros(_shape, dtype=bool)


class PushDownStack(object):
    """Simple Stack implements in the form of array"""

    def __init__(self, max_len, canvas_shape):
        """
        Simulates a push down stack for canvases. Idea can be taken to build
        generic stacks.
        :param max_len: Max length of stack
        :param canvas_shape: shape of canvas
        """
        _shape = [max_len] + canvas_shape
        self.max_len = max_len
        self.canvas_shape = canvas_shape
        self.items = []
        self.max_len = max_len

    def push(self, item):
        if len(self.items) >= self.max_len:
            assert False, "exceeds max len for stack!!"
        self.items = [item.copy()] + self.items

    def pop(self):
        if len(self.items) == 0:
            assert False, "below min len of stack!!"
        item = self.items[0]
        self.items = self.items[1:]
        return item

    def get_items(self):
        # In this we create a fixed shape tensor amenable for further usage
        # we basically create a fixed length stack and fill up the empty
        # space with zero elements
        zero_stack_element = [
            np.zeros(self.canvas_shape, dtype=bool)
            for _ in range(self.max_len - len(self.items))
        ]
        items = np.stack(self.items + zero_stack_element)
        return items.copy()

    def clear(self):
        """Re-initializes the stack"""
        _shape = [self.max_len] + self.canvas_shape
        self.items = []


class Parser:
    """
    Parser to parse the program written in postfix notation
    """

    def __init__(self):
        self.shape_types = ["c", "s", "t"]
        self.op = ["*", "+", "-"]

    def parse(self, expression: string):
        """
        Takes an empression, returns a serial program
        :param expression: program expression in postfix notation
        :return program:
        """
        program = []
        for index, value in enumerate(expression):
            if value in self.shape_types:
                # draw shape instruction
                program.append({})
                program[-1]["value"] = value
                program[-1]["type"] = "draw"
                # find where the parenthesis closes
                close_paren = expression[index:].index(")") + index
                program[-1]["param"] = expression[index + 2:close_paren].split(
                    ",")
            elif value in self.op:
                # operations instruction
                program.append({})
                program[-1]["type"] = "op"
                program[-1]["value"] = value
            elif value == "$":
                # must be a stop symbol
                program.append({})
                program[-1]["type"] = "stop"
                program[-1]["value"] = "$"
        return program


class SimulateStack:
    def __init__(self, max_len, canvas_shape):
        """
        Takes the program and simulate stack for it.
        :param max_len: Maximum size of stack
        :param canvas_shape: canvas shape, for elements of stack
        """
        self.draw_obj = Draw(canvas_shape=canvas_shape)
        self.draw = {
            "c": self.draw_obj.draw_circle,
            "s": self.draw_obj.draw_square,
            "t": self.draw_obj.draw_triangle
        }
        self.canvas_shape = canvas_shape
        self.op = {"*": self._and, "+": self._union, "-": self._diff}
        # self.stack = CustomStack(max_len, canvas_shape)
        self.stack = PushDownStack(max_len, canvas_shape)
        self.stack_t = []
        self.stack.clear()
        self.stack_t.append(self.stack.get_items())

    def generate_stack(self, program: list, start_scratch=True):
        """
        Executes the program step-by-step and stores all intermediate stack
        states.
        :param program: List with each item a program step
        :param start_scratch: whether to start creating stack from scratch or
        stack already exist and we are appending new instructions. With this
        set to False, stack can be started from its previous state.
        """
        # clear old garbage
        if start_scratch:
            self.stack_t = []
            self.stack.clear()
            self.stack_t.append(self.stack.get_items())

        for index, p in enumerate(program):
            if p["type"] == "draw":
                # normalize it so that it fits for every dimension multiple
                # of 64, because the programs are generated for dimension of 64
                x = int(p["param"][0]) * self.canvas_shape[0] // 64
                y = int(p["param"][1]) * self.canvas_shape[1] // 64
                scale = int(p["param"][2]) * self.canvas_shape[0] // 64
                # Copy to avoid over-write
                layer = self.draw[p["value"]]([x, y], scale)
                self.stack.push(layer)

                # Copy to avoid orver-write
                # self.stack_t.append(self.stack.items.copy())
                self.stack_t.append(self.stack.get_items())
            else:
                # operate
                obj_2 = self.stack.pop()
                obj_1 = self.stack.pop()
                layer = self.op[p["value"]](obj_1, obj_2)
                self.stack.push(layer)
                # Copy to avoid over-write
                # self.stack_t.append(self.stack.items.copy())
                self.stack_t.append(self.stack.get_items())

    def _union(self, obj1: np.ndarray, obj2: np.ndarray):
        return np.logical_or(obj1, obj2)

    def _and(self, obj1: np.ndarray, obj2: np.ndarray):
        return np.logical_and(obj1, obj2)

    def _diff(self, obj1: np.ndarray, obj2: np.ndarray):
        return (obj1 * 1. - np.logical_and(obj1, obj2) * 1.).astype(np.bool)
