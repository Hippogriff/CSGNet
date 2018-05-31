import numpy as np
import cv2
from src.utils.train_utils import image_from_expressions, validity
import json
from typing import List


class GenerateGroupings:
    def __init__(self, root_path, train_size, test_size, image_dim=64):
        """
        Generates programs for Grouping task. It generates programs for a cluster
        containing different objects. A cluster is a tree, where parent program is sub-string
        of children program. In this way it generates a forest of trees (clusters).
        :param root_path: root path where programs are stored
        :param train_size: train size
        :param test_size: test size
        :param image_dim: canvas dimension
        """
        with open(root_path + "train_substrings.json", "r") as file:
            self.train_substrings = json.load(file)

        with open(root_path + "test_substrings.json", "r") as file:
            self.test_substrings = json.load(file)

        self.train_substrings = {k: self.train_substrings[str(k)] for k in range(train_size)}
        self.test_substrings = {k: self.test_substrings[str(k)] for k in range(test_size)}
        self.train_sz = train_size
        self.test_sz = test_size
        self.image_dim = image_dim

    def train_gen(self, number_of_objects, number_of_trees):
        """
        Generates cluster programs to be drawn in one image.
        :param number_of_objects: Total number of objects to draw in one image
        :param number_of_trees: total number of cluster to draw in one image
        :return:
        """
        num_objs = 0
        programs = []
        while num_objs < number_of_objects:
            index = np.random.choice(len(self.train_substrings))
            if num_objs + len(self.train_substrings[index].keys()) > number_of_objects:
                required_indices = sorted(self.train_substrings[index].keys())[0:number_of_objects - num_objs]
                cluster = {}
                for r in required_indices:
                    p = self.train_substrings[index][r]
                    image = image_from_expressions([p,], stack_size=9, canvas_shape=[64, 64])

                    # Makes sure that the object created doesn't have disjoint parts,
                    # don't include the program, because it makes the analysis difficult.
                    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        np.array(image[0], dtype=np.uint8))
                    if nlabels > 2:
                        continue
                    cluster[r] = self.train_substrings[index][r]
                if cluster:
                    programs.append(cluster)
                    num_objs += len(cluster.keys())
                num_objs += len(cluster.keys())
            else:
                cluster = {}
                for k, p in self.train_substrings[index].items():
                    image = image_from_expressions([p], stack_size=9, canvas_shape=[64, 64])
                    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        np.array(image[0], dtype=np.uint8))
                    if nlabels > 2:
                        continue
                    cluster[k] = p
                if cluster:
                    programs.append(cluster)
                    num_objs += len(cluster.keys())
        return programs

    def place_on_canvas(self, programs):
        """
        Places objects from progams one by one on bigger canvas randomly
        such there is no intersection between objects.
        """
        canvas = np.zeros((240, 240), dtype=bool)
        grid = np.arange(0, 16)
        valid_objects = 0
        images = image_from_expressions(programs, stack_size=9, canvas_shape=[64, 64])

        objects_done = 0
        xi, yj = np.meshgrid(np.arange(3), np.arange(3))
        xi = np.reshape(xi, 9)
        yj = np.reshape(yj, 9)
        random_index = np.random.choice(np.arange(9), len(programs), replace=False)
        for index in range(len(programs)):
            x, y = np.random.choice(grid, 2)
            canvas[xi[random_index[index]] * 80 + x: xi[random_index[index]] * 80 + x + 64,
            yj[random_index[index]] * 80 + y: yj[random_index[index]] * 80 + y + 64] = images[index]
        return canvas


class Grouping:
    def __init__(self):
        pass

    def group(self, image):
        bbs = self.tightboundingbox(image)
        num_objects = len(bbs)
        similarity_matrix = np.zeros((num_objects, num_objects))
        objects = self.find_unique(image, bbs)
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                _, _, w1, h1 = bbs[i]
                _, _, w2, h2 = bbs[j]
                # if w1 == w2 and h1 == h2:
                #     ob1 = objects[i]
                #     ob2 = objects[j]
                #     iou = np.sum(np.logical_and(ob1, ob2)) / np.sum(np.logical_or(ob1, ob2))
                #     if iou == 1:
                #         similarity_matrix[i, j] = True
        return similarity_matrix, bbs, objects

    def similarity_to_cluster(self, similarity):
        """
        Takes similarity matrix and returns cluster
        """
        clusters = []
        num_objs = similarity.shape[0]
        non_zero_x, non_zero_y = np.nonzero(similarity == 1.0)
        for x in range(non_zero_x.shape[0]):
            if len(clusters) == 0:
                clusters.append([non_zero_x[x], non_zero_y[x]])
            else:
                found = False
                for c in clusters:
                    if non_zero_x[x] in c or non_zero_y[x] in c:
                        c.append(non_zero_x[x])
                        c.append(non_zero_y[x])
                        found = True
                        break
                if not found:
                    clusters.append([non_zero_x[x], non_zero_y[x]])

        diff_sets = set(np.arange(num_objs))
        for index, c in enumerate(clusters):
            clusters[index] = list(set(c))
            diff_sets = diff_sets - set(c)
        clusters += [s for s in diff_sets]
        return clusters

    def find_unique(self, image, bbs):
        objects = [self.object_from_bb(image, bb) for bb in bbs]
        return objects

    def object_from_bb(self, image, bb):
        x, y, w, h = bb
        return image[x:x + h, y:y + w]

    def nms(self, bbs):
        """
        No maximal suppressions
        :param bbs: list containing bounding boxes
        :return: pruned list containing bounding boxes
        """
        for index1, b1 in enumerate(bbs):
            for index2, b2 in enumerate(bbs):
                if index1 == index2:
                    continue
                if self.inside(b1, b2):
                    _, _, w1, h1 = b1
                    _, _, w2, h2 = b2
                    if w1 * h1 >= w2 * h2:
                        del bbs[index2]
                    else:
                        del bbs[index1]
        return bbs

    def tightboundingbox(self, image):
        ret, thresh = cv2.threshold(np.array(image, dtype=np.uint8), 0, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bb = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # +1 is done to encapsulate entire figure
            w += 2
            h += 2
            x -= 1
            y -= 1
            x = np.max([0, x])
            y = np.max([0, y])
            bb.append([y, x, w, h])
        bb = self.nms(bb)
        return bb

    def replace_in_small_canvas(self, img, canvas_shape:List):
        canvas = np.zeros(canvas_shape, dtype=np.bool)
        h, w = img.shape
        diff_h = canvas_shape[0] - h
        diff_w = canvas_shape[1] - w
        canvas[diff_h // 2:diff_h // 2 + h, diff_w // 2:diff_w// 2 + w] = img
        return canvas

    def inside(self, bb1, bb2):
        """
        check if the bounding box 1 is inside bounding box 2
        """
        x1, y1, w1, h1 = bb1
        x, y, w, h = bb2
        coor2 = [[x, y],
                 [x + w, y],
                 [x + w, y + w],
                 [x, y + w]]
        for x2, y2 in coor2:
            cond1 = (x1 <= x2) and (x2 <= x1 + w1)
            cond2 = (y1 <= y2) and (y2 <= y1 + h1)
            if cond1 and cond2:
                return True
        return False


def transform(rot, trans, mean, image):
    M = cv2.getRotationMatrix2D((mean[0, 0], mean[1, 0]), np.arcsin(rot[0, 1]) * 180 / np.pi, 1)
    M[0, 2] += trans[0]
    M[1, 2] += trans[1]
    image = cv2.warpAffine(image.astype(np.float32), M, (64, 64))
    return image