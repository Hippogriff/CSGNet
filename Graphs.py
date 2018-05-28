import numpy as np

class PriorityQueue:
    def __init__(self):
        self.heapArray = [(0, 0)]
        self.currentSize = 0

    def buildHeap(self, alist):
        self.currentSize = len(alist)
        self.heapArray = [(0, 0)]
        for i in alist:
            self.heapArray.append(i)
        i = len(alist) // 2
        while (i > 0):
            self.percDown(i)
            i = i - 1

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapArray[i][0] > self.heapArray[mc][0]:
                tmp = self.heapArray[i]
                self.heapArray[i] = self.heapArray[mc]
                self.heapArray[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 > self.currentSize:
            return -1
        else:
            if i * 2 + 1 > self.currentSize:
                return i * 2
            else:
                if self.heapArray[i * 2][0] < self.heapArray[i * 2 + 1][0]:
                    return i * 2
                else:
                    return i * 2 + 1

    def percUp(self, i):
        while i // 2 > 0:
            if self.heapArray[i][0] < self.heapArray[i // 2][0]:
                tmp = self.heapArray[i // 2]
                self.heapArray[i // 2] = self.heapArray[i]
                self.heapArray[i] = tmp
            i = i // 2

    def add(self, k):
        self.heapArray.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def delMin(self):
        retval = self.heapArray[1][1]
        self.heapArray[1] = self.heapArray[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapArray.pop()
        self.percDown(1)
        return retval

    def isEmpty(self):
        if self.currentSize == 0:
            return True
        else:
            return False

    def decreaseKey(self, val, amt):
        # this is a little wierd, but we need to find the heap thing to decrease by
        # looking at its value
        done = False
        i = 1
        myKey = 0
        while not done and i <= self.currentSize:
            if self.heapArray[i][1] == val:
                done = True
                myKey = i
            else:
                i = i + 1
        if myKey > 0:
            self.heapArray[myKey] = (amt, self.heapArray[myKey][1])
            self.percUp(myKey)

    def __contains__(self, vtx):
        for pair in self.heapArray:
            if pair[1] == vtx:
                return True
        return False


# class Vertex:
#     def __init__(self, key):
#         self.id = key
#         self.connectedTo = {}
#         self.distance = 1e4
#         self.pred = None
#         self.program = []
#         self.program_selected = False
#
#     def addNeighbor(self, nbr, weight=0):
#         self.connectedTo[nbr] = weight
#
#     def __str__(self):
#         return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])
#
#     def getConnections(self):
#         return self.connectedTo.keys()
#
#     def getId(self):
#         return self.id
#
#     def getWeight(self, nbr):
#         return self.connectedTo[nbr]
#
#     def getDistance(self):
#         return self.distance
#
#     def setDistance(self, distance):
#         self.distance = distance
#
#     def setPred(self, pred):
#         self.pred = pred


class Node:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.distance = 1e2
        self.pred = None

        # Whether a program is selected or not
        self.program_id = None
        self.root = False
        self.selected = False
        self.best_weight = None

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def getDistance(self):
        return self.distance

    def setDistance(self, distance):
        self.distance = distance

    def setPred(self, pred):
        self.pred = pred


class Graph:
    """
    Creates a directed graph
    """
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Node(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertList

    def addEdge(self, f, t, weights):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)

        self.vertList[f].addNeighbor(self.vertList[t], weights)
        self.vertList[t].addNeighbor(self.vertList[f], weights)

    def getVertices(self):
        return self.vertList.keys()

    def vertex_keys(self):
        self.vertex2keys = {}
        for k, v in self.vertList.items():
            self.vertex2keys[v] = k

    def getIndex(self, vertex):
        for k, v in self.vertList.items():
            if v == vertex:
                return k
        return None

    def getEdgesWeight(self, vertex1, vertex2):
        """
        Get the minimum weight from vertex1 to vertex2
        :param vertex1: is the vertex that is already selected to be part of the MST
        :param program_id1: Program id that is selected for the vertex1
        :param vertex2: the vertex not selected yet
        :return:
        """
        key1 = self.vertex2keys[vertex1]
        key2 = self.vertex2keys[vertex2]
        program_id1 = vertex1.program_id
        if vertex1.root:
            # vertex is a root
            weight = np.min(vertex1.connectedTo[vertex2])
            program_id = np.argmin(vertex1.connectedTo[vertex2])
            return weight, program_id

        else:
            weight = np.min(vertex1.connectedTo[vertex2][program_id1, :])
            program_id = np.argmin(vertex1.connectedTo[vertex2][program_id1, :])
            return weight, program_id

    def __iter__(self):
        return iter(self.vertList.values())


def dijkstra(aGraph,start):
    pq = PriorityQueue()
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v) for v in aGraph])
    while not pq.isEmpty():
        currentVert = pq.delMin()
        for nextVert in currentVert.getConnections():
            newDist = currentVert.getDistance() + currentVert.getWeight(nextVert)
            if newDist < nextVert.getDistance():
                nextVert.setDistance( newDist )
                nextVert.setPred(currentVert)
                pq.decreaseKey(nextVert, newDist)


def steinertree(G,start):
    Nodes = []
    pq = PriorityQueue()
    for v in G:
        v.setDistance(1e2)
        v.setPred(None)
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v) for v in G])
    while not pq.isEmpty():
        currentVert = pq.delMin()
        currentVert.selected = True
        Nodes.append(G.vertex2keys[currentVert])
        for nextVert in currentVert.getConnections():
            if nextVert.selected:
                continue
            newCost, program_id = G.getEdgesWeight(currentVert, nextVert)
            if nextVert in pq and newCost < nextVert.getDistance():
                nextVert.setPred(currentVert)
                nextVert.setDistance(newCost)
                nextVert.program_id = program_id
                nextVert.best_weight = newCost
                pq.decreaseKey(nextVert, newCost)
    return Nodes

def prim(G,start):
    pq = PriorityQueue()
    for v in G:
        v.setDistance(1e2)
        v.setPred(None)
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v) for v in G])
    while not pq.isEmpty():
        currentVert = pq.delMin()
        for nextVert in currentVert.getConnections():
            newCost = currentVert.getWeight(nextVert)
            if nextVert in pq and newCost < nextVert.getDistance():
              nextVert.setPred(currentVert)
              nextVert.setDistance(newCost)
              pq.decreaseKey(nextVert,newCost)


# prim(graph, graph.vertList[0])
# graph.vertex_keys()
#
# new_graph = Graph()
# for k, v in graph.vertList.items():
#     mini_dist = 1e5
#     mini_neigh = None
#     for neighbour in v.getConnections():
#         if neighbour.getDistance() < mini_dist:
#             mini_dist = neighbour.getDistance()
#             mini_neigh = neighbour
#     neigh_key = graph.getIndex(mini_neigh)
#     print (k, neigh_key)
#     if not neigh_key == None:
#         new_graph.addEdge(k, neigh_key, v.connectedTo[mini_neigh])