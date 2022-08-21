import numpy


class KalmanFilter(object):

    def __init__(self):
        self.dt = 0.005
        self.A = numpy.array([[1, 0], [0, 1]])
        self.u = numpy.zeros((2, 1))
        self.b = numpy.array([[0], [255]])
        self.P = numpy.diag((3.0, 3.0))
        self.F = numpy.array([[1.0, self.dt], [0.0, 1.0]])
        self.Q = numpy.eye(self.u.shape[0])
        self.R = numpy.eye(self.b.shape[0])
        self.lastResult = numpy.array([[0], [255]])

    def predict(self):
        self.u = numpy.round(numpy.dot(self.F, self.u))
        self.P = numpy.dot(self.F, numpy.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u
        return self.u

    def correct(self, b, flag):
        if not flag:
            self.b = self.lastResult
        else:
            self.b = b
        C = numpy.dot(self.A, numpy.dot(self.P, self.A.T)) + self.R
        K = numpy.dot(self.P, numpy.dot(self.A.T, numpy.linalg.inv(C)))
        self.u = numpy.round(self.u + numpy.dot(K, (self.b - numpy.dot(self.A, self.u))))
        self.P = self.P - numpy.dot(K, numpy.dot(C, K.T))
        self.lastResult = self.u
        return self.u
