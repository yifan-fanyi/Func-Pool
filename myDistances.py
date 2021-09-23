import numpy as np

class myDistances:
    @staticmethod
    def canberra_distance(point1, point2):
        dist = 0
        for i in range(len(point1)):
            if np.abs(point1[i]) + np.abs(point2[i]) != 0:
                dist += np.abs(point1[i]-point2[i]) / (np.abs(point1[i]) + np.abs(point2[i]))
        return dist

    @staticmethod
    def minkowski_distance(point1, point2):
        point1, point2 = np.array(point1), np.array(point2)
        return np.power(np.sum(np.power(np.abs(point1-point2), 3)), 1/3)

    @staticmethod
    def euclidean_distance(point1, point2):
        point1, point2 = np.array(point1), np.array(point2)
        return np.power(np.sum(np.power(np.abs(point1-point2), 2)), 1/2)

    @staticmethod
    def inner_product_distance(point1, point2):
        point1, point2 = np.array(point1), np.array(point2)
        return np.inner(point1, point2)

    @staticmethod
    def cosine_similarity_distance(point1, point2):
        point1, point2 = np.array(point1), np.array(point2)
        if (np.sqrt(np.sum(np.square(point1))) * np.sqrt(np.sum(np.square(point2)))) < 1e-20:
            return 1.
        return 1. - np.inner(point1, point2) / (np.sqrt(np.sum(np.square(point1))) * np.sqrt(np.sum(np.square(point2))))

    @staticmethod
    def gaussian_kernel_distance(point1, point2):
        point1, point2 = np.array(point1), np.array(point2)
        return -np.exp(-0.5 * np.inner(point1-point2, point1-point2))

