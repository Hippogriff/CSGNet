import numpy as np
from src.utils import read_config
from src.utils import train_utils
from src.Models.models import ParseModelOutput

max_len = 13
config = read_config.Config("config_synthetic.yml")
valid_permutations = train_utils.valid_permutations

# Load the terminals symbols of the grammar
with open("terminals.txt", "r") as file:
    unique_draw = file.readlines()
for index, e in enumerate(unique_draw):
    unique_draw[index] = e[0:-1]

parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len, config.canvas_shape)


class EditDistance:
    """
    Defines edit distance between two programs. Following criterion are used
    to find edit distance:
    1. Done: Subset string
    2. % Subset
    3. Primitive type based subsetting
    4. Done: Permutation invariant subsetting
    """

    def __init__(self):
        pass

    def edit_distance(self, prog1, prog2, iou):
        """
        Calculates edit distance between two programs
        :param prog1:
        :param prog2:
        :param iou:
        :return:
        """
        prog1_tokens = self.parse(prog1)
        prog2_tokens = self.parse(prog2)

        all_valid_programs1 = list(set(valid_permutations(prog1_tokens, permutations=[], stack=[], start=True)))
        all_valid_programs2 = list(set(valid_permutations(prog2_tokens, permutations=[], stack=[], start=True)))
        if iou == 1:
            return 0

        # if prog1 in prog2:
        #     return len(prog2_tokens) - len(prog1_tokens)
        #
        # elif prog2 in prog1:
        #     return len(prog1_tokens) - len(prog2_tokens)
        # else:
        #     return 100

        if len(prog1_tokens) <= len(prog2_tokens):
            subsets1 = self.exhaustive_subsets_edit_distance(all_valid_programs1, all_valid_programs2)
            return np.min(subsets1)
        else:
            subsets2 = self.exhaustive_subsets_edit_distance(all_valid_programs2, all_valid_programs1)
            return np.min(subsets2)
        # return np.min([np.min(subsets1), np.min(subsets2)])

    def exhaustive_subsets_edit_distance(self, progs1, progs2):
        len_1 = len(progs1)
        len_2 = len(progs2)
        subset_flag = np.zeros((len_1, len_2))
        for index1, p1 in enumerate(progs1):
            for index2, p2 in enumerate(progs2):
                if p1 in p2:
                    prog1_tokens = self.parse(p1)
                    prog2_tokens = self.parse(p2)
                    subset_flag[index1, index2] = len(prog2_tokens) - len(prog1_tokens)
                else:
                    subset_flag[index1, index2] = 100
        return subset_flag

    def subset_program_structure_primitives(self, prog1, prog2):
        """
        Define edit distance based on partial program structure and primitive
        types. If the partial program structure is same and the position of the
        primitives is same, then edit distance is positive.
        """
        pass

    def parse(self, expression):
        """
        NOTE: This method is different from parse method in Parser class
        Takes an expression, returns a serial program
        :param expression: program expression in postfix notation
        :return program:
        """
        shape_types = ["c", "s", "t"]
        op = ["*", "+", "-"]
        program = []
        for index, value in enumerate(expression):
            if value in shape_types:
                program.append({})
                program[-1]["type"] = "draw"

                # find where the parenthesis closes
                close_paren = expression[index:].index(")") + index
                program[-1]["value"] = expression[index:close_paren + 1]
            elif value in op:
                program.append({})
                program[-1]["type"] = "op"
                program[-1]["value"] = value
            else:
                pass
        return program