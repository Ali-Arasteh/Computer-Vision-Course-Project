import cv2
import numpy
from kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Track(object):

    def __init__(self, prediction, trackIdCount):
        self.track_id = trackIdCount
        self.KF = KalmanFilter()
        self.prediction = numpy.asarray(prediction)
        self.skipped_frames = 0
        self.frame = []
        self.first = []
        self.v = 0


class Tracker(object):

    def __init__(self, dist_thresh, max_frames_to_skip, min_frames_count):
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.min_frames_count = min_frames_count
        self.trackIdCount = 0
        self.tracks = []
        self.finalTracks = []

    def Update(self, frame, contours, centers, counter):
        if len(self.tracks) == 0:
            for i in range(len(centers)):
                track = Track(centers[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)
        N = len(self.tracks)
        M = len(centers)
        cost = numpy.zeros(shape=(N, M))
        for i in range(len(self.tracks)):
            for j in range(len(centers)):
                try:
                    diff = self.tracks[i].prediction - centers[j]
                    cost[i][j] = numpy.sqrt(diff[0][0] * diff[0][0] + diff[1][0] * diff[1][0]) / 2
                except:
                    pass
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]
        for i in range(len(assignment)):
            if assignment[i] != -1:
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
                pass
            else:
                self.tracks[i].skipped_frames += 1
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)
                if len(self.tracks[i].frame) > self.min_frames_count:
                    self.finalTracks.append(self.tracks[i])
        if len(del_tracks) > 0:
            for j in del_tracks:
                if j < len(self.tracks):
                    del self.tracks[j]
                    del assignment[j]
        un_assigned_detects = []
        for i in range(len(centers)):
            if i not in assignment:
                un_assigned_detects.append(i)
        if len(un_assigned_detects) > 0:
            for i in range(len(un_assigned_detects)):
                track = Track(centers[un_assigned_detects[i]], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()
            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                new_predition = self.tracks[i].KF.correct(centers[assignment[i]], 1)
                """
                new_v = numpy.sqrt((self.tracks[i].prediction[0][0] - new_predition[0][0]) ** 2 + (self.tracks[i].prediction[1][0] - new_predition[1][0]) ** 2)
                if new_v > 20:
                    new_v = 20
                self.tracks[i].v = (self.tracks[i].v + new_v) / 2
                if len(self.tracks[i].frame) == 10:
                    self.tracks[i].first = new_predition
                """
                self.tracks[i].prediction = new_predition
                mask = numpy.zeros(frame.shape).astype(dtype=numpy.uint8)
                cv2.drawContours(mask, [contours[assignment[i]]], -1, (255, 255, 255), cv2.FILLED)
                """
                mask = mask.astype(dtype=numpy.int)
                mask_sum = numpy.sum(mask) / (255 * 3)
                mask = mask.astype(dtype=numpy.uint8)
                """
                mask = numpy.where(mask, frame, 0)
                """
                mask = mask.astype(dtype=numpy.int)
                pixels_sum = numpy.array([numpy.sum(mask[:, :, 0]) / mask_sum, numpy.sum(mask[:, :, 1]) / mask_sum, numpy.sum(mask[:, :, 2]) / mask_sum]).astype(dtype=numpy.int)
                mask = mask.astype(dtype=numpy.uint8)
                cv2.waitKey(0)
                cv2.circle(mask, (centers[assignment[i]][0][0], centers[assignment[i]][1][0]), 5, (int(pixels_sum[0]), int(pixels_sum[1]), int(pixels_sum[2])), -1)
                """
                cv2.putText(mask, str(int(counter / 1800)) + ":" + str(int(numpy.remainder(counter / 30, 60))), (centers[assignment[i]][0][0], centers[assignment[i]][1][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                #cv2.putText(mask, str(int(self.tracks[i].v)), (centers[assignment[i]][0][0], centers[assignment[i]][1][0] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                self.tracks[i].frame.append(mask)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(numpy.array([[0], [0]]), 0)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction

    def Finish(self):
        for i in range(len(self.tracks)):
            if len(self.tracks[i].frame) > self.min_frames_count:
                self.finalTracks.append(self.tracks[i])
