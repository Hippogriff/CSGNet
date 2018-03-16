import numpy as np
import string
from scipy.optimize import minimize
from src.Models.models import ParseModelOutput
from src.utils.train_utils import chamfer
from src.utils.train_utils import validity


class Optimize:
    """
    Post processing visually guided search using Powell optimizer.
    """

    def __init__(self, query_expression, metric="iou", stack_size=7, steps=15):
        """
        Post processing visually guided search.
        :param query_expression: expression to be optimized
        :param metric: metric to be minimized, like chamfer
        :param stack_size: max stack size required in any program
        :param steps: max tim step of any program
        """
        self.parser = ParseModelOutput(canvas_shape=[64, 64], stack_size=stack_size, unique_draws=None, steps=steps)
        self.query_expression = query_expression
        self.get_graph_structure(query_expression)
        self.metric = metric
        self.errors = []

    def get_target_image(self, image: np.ndarray):
        """
        Gets the target image.
        :param image: target image
        :return: 
        """
        self.target_image = image
        
    def get_graph_structure(self, expression):
        """
        returns the nodes (terminals) of the program
        :param expression: input query expression
        :return: 
        """
        program = self.parser.Parser.parse(expression)
        self.graph_str = []
        for p in program:
            self.graph_str.append(p["value"])

    def make_expression(self, x: np.ndarray):
        expression = ""
        index = 0
        for e in self.graph_str:
            if e in ["c", "s", "t"]:
                expression += e + "({},{},{})".format(x[index], x[index + 1], x[index + 2])
                index += 3
            else:
                expression += e
        return expression

    def objective(self, x: np.ndarray):
        """
        Objective to minimize.
        :param x: input program parameters in numpy array format
        :return: 
        """
        x = x.astype(np.int)
        x = np.clip(x, 8, 56)

        query_exp = self.make_expression(x)
        query_image = self.parser.expression2stack([query_exp])[-1, 0, 0, :, :]
        if self.metric == "iou":
            error = -np.sum(
                np.logical_and(self.target_image, query_image)) / np.sum(
                np.logical_or(self.target_image, query_image))
        elif self.metric == "chamfer":
            error = chamfer(np.expand_dims(self.target_image, 0),
                            np.expand_dims(query_image, 0))
        return error


def validity(program, max_time, timestep):
    """
    Checks the validity of the program.
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


def optimize_expression(query_exp: string, target_image: np.ndarray, metric="iou", stack_size=7, steps=15, max_iter = 100):
    """
    A helper function for visually guided search. This takes the target image (or test 
    image) and predicted expression from CSGNet and returns the final chamfer distance 
    and optmized program with least chamfer distance possible.
    :param query_exp: program expression 
    :param target_image: numpy array of test image
    :param metric: metric to minimize while running the optimizer, "chamfer"
    :param stack_size: max stack size of the program required
    :param steps: max number of time step present in any program
    :param max_iter: max iteration for which to run the program.
    :return: 
    """
    # a parser to parse the input expressions.
    parser = ParseModelOutput(canvas_shape=[64, 64], stack_size=stack_size,
                              unique_draws=None, steps=steps)

    program = parser.Parser.parse(query_exp)
    if not validity(program, len(program), len(program) - 1):
        return query_exp, 16

    x = []
    for p in program:
        if p["value"] in ["c", "s", "t"]:
            x += [int(t) for t in p["param"]]

    optimizer = Optimize(query_exp, metric=metric, stack_size=stack_size, steps=steps)
    optimizer.get_target_image(target_image)

    if max_iter == None:
        # None will stop when tolerance hits, not based on maximum iterations
        res = minimize(optimizer.objective, x, method="Powell", tol=0.0001,
                       options={"disp": False, 'return_all': False})
    else:
        # This will stop when max_iter hits
        res = minimize(optimizer.objective, x, method="Powell", tol=0.0001, options={"disp": False, 'return_all': False, "maxiter": max_iter})

    final_value = res.fun
    res = res.x.astype(np.int)
    for i in range(2, res.shape[0], 3):
        res[i] = np.clip(res[i], 8, 32)
    res = np.clip(res, 8, 56)
    predicted_exp = optimizer.make_expression(res)
    return predicted_exp, final_value