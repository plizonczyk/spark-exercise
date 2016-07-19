import math


def euclidean_norm(a):
    squares = [x**2 for x in a]
    return math.sqrt(sum(squares))


def euclidean_distance(a, b):
    distances = [(x - y)**2 for x, y in zip(a, b)]
    return math.sqrt(sum(distances))


class VectorWithNorm:
    def __init__(self, vec, norm):
        self.vector = vec
        self.norm = norm

    def __repr__(self):
        return "Vector: " + str(self.vector) + ", Norm: " + str(self.norm)
