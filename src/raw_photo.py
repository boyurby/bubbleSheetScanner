import cv2
import numpy as np
import json
from paper_scan import PaperScan
from options import DEBUG

# Adaptive threshold
THR_MAX_VAL = 255
THR_BLOCK_SIZE = 29
THR_OFFSET = 8

# Extract paper
SIZE_VARIANCE_FACTOR = 0.7
TEMPLATE_KEY_PTS = np.float32([[764, 307], [49, 307], [49, 1128], [764, 1128]])
PAPER_SIZE = (875, 1240)
REF_PT_RATIO = 1.053
REF_PT_RANGE = 0.012


class RawPhoto:
    """
    Represents each physical, raw photo taken by the user that is to be processed.
    Each raw photo can contains multiple test papers, so there is a list of PaperScan objects in each RawPhoto object.
    When used, initialize a RawPhoto object and call dump_data() on it to get the results.
    """
    raw_img = None
    thr_img = None
    paper_objs = None
    metadata = ''

    def __init__(self, raw_image, num_papers):
        """
        Initializes the RawPhoto object.
        The only function that needs to be call (to the RawPhoto object itself) when processing a new photo. All other
        functions (other than dump_data()) are called by this top-level initializer.
        :param raw_image: image loaded by cv2.imread()
        :param num_papers: number of papers in the photo
        """
        self.metadata = ''
        self.num_papers = []
        self.raw_img = raw_image

        # Threshold original image
        self.thr_img = cv2.adaptiveThreshold(self.raw_img, THR_MAX_VAL, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, THR_BLOCK_SIZE, THR_OFFSET)
        if DEBUG:
            cv2.imwrite("../tmp/self_th.png", self.thr_img)

        # Find contours
        # contours = cv2.findContours(self.thr_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, _ = cv2.findContours(self.thr_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # temporary fix to known issue of a certain version of OpenCV. Depending on OpenCV version, might need to
        # change this line to read `contours, _ = ...`
        # CV_RETR_LIST retrieves all of the contours without establishing any hierarchical relationships.
        # CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only end points.

        # Approximate all rectangles
        # dictionary that maps areas to their corresponding contours
        approximations = {}
        for i in range(len(contours)):
            # approximate contours to polygons
            approx_curve = True
            approx = cv2.approxPolyDP(contours[i], 4, approx_curve)
            # has 4 sides? is convex?
            if (len(approx) != 4) or (not cv2.isContourConvex(approx)):
                continue
            approximations[cv2.contourArea(approx)] = approx

        # Extract each individual paper
        self.paper_objs = self.extract_papers(approximations, num_papers)

    def extract_papers(self, approximations, num_papers):
        """
        Identify the papers in the photo and initialize the list of PaperScan objects.
        :param approximations: a dictionary of rectangles in the image (rectangle area -> rectangle vertices)
                               where each rectangle is the outer edge of the table on the paper
        :param num_papers: number of papers in the image
        :return: a list of paper objects
        """
        sizes = sorted(list(approximations.keys()))
        sizes.reverse()
        papers = []
        for i in range(num_papers):
            # Break if rectangle is clearly not big enough
            # factor of smallest allowed rectangle to largest rectangle in the picture
            if sizes[i] < SIZE_VARIANCE_FACTOR * sizes[1]:
                print('%d paper(s) not detected.' % (num_papers - i))
                self.metadata += '%d paper(s) not detected.\n' % (num_papers - i)
                break
            approx = approximations[sizes[i]]
            raw_refs = self.orientate_vertices(approx)
            ref_pts = np.float32([[raw_refs[0][0], raw_refs[0][1]],
                                  [raw_refs[1][0], raw_refs[1][1]],
                                  [raw_refs[2][0], raw_refs[2][1]],
                                  [raw_refs[3][0], raw_refs[3][1]]])
            trans_matrix = cv2.getPerspectiveTransform(ref_pts, TEMPLATE_KEY_PTS)
            paper = cv2.warpPerspective(self.raw_img, trans_matrix, PAPER_SIZE)
            # Need to re-threshold raw photo (in PaperScan) because warpPerspective() largely reduces the quality of
            # the original binary image
            papers.append(PaperScan(paper))
        return papers

    def orientate_vertices(self, approx):
        """
        Orientate a paper region based on the black bock besides the left edge.
        :param approx: vertices of the paper rectangle
        :return: transformed vertices of the paper rectangle
        """
        # Define corner points
        A = approx[0][0]
        B = approx[1][0]
        C = approx[2][0]
        D = approx[3][0]

        # Define center points of edges
        E = (int((A[0] + B[0]) / 2), int((A[1] + B[1]) / 2))
        F = (int((B[0] + C[0]) / 2), int((B[1] + C[1]) / 2))
        G = (int((C[0] + D[0]) / 2), int((C[1] + D[1]) / 2))
        H = (int((D[0] + A[0]) / 2), int((D[1] + A[1]) / 2))

        # Define ref points (points just a bit outside the mid points)
        I = (int(G[0] + REF_PT_RATIO * (E[0] - G[0])),
             int(G[1] + REF_PT_RATIO * (E[1] - G[1])))
        J = (int(H[0] + REF_PT_RATIO * (F[0] - H[0])),
             int(H[1] + REF_PT_RATIO * (F[1] - H[1])))
        K = (int(E[0] + REF_PT_RATIO * (G[0] - E[0])),
             int(E[1] + REF_PT_RATIO * (G[1] - E[1])))
        L = (int(F[0] + REF_PT_RATIO * (H[0] - F[0])),
             int(F[1] + REF_PT_RATIO * (H[1] - F[1])))

        # Check brightnesses
        brightnesses = []
        offset = int(REF_PT_RANGE * ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5)
        for pt in [I, J, K, L]:
            i_base = pt[0] - offset
            j_base = pt[1] - offset
            brightness = 0
            for i in range(offset * 2):
                for j in range(offset * 2):
                    brightness += self.raw_img[j_base + j][i_base + i]
            brightnesses.append(brightness)

        # Orientate rectangle
        r_id = brightnesses.index(min(brightnesses))
        transform = {0: B, 1: C, 2: D, 3: A}
        return (transform[r_id], transform[(r_id + 1) % 4],
                transform[(r_id + 2) % 4], transform[(r_id + 3) % 4])

    def dump_data(self):
        """
        Dumps data to a JSON string
        :return: JSON string
        """
        data_dict = {'papers': [],
                     'metadata': self.metadata}
        for paper in self.paper_objs:
            data_dict['papers'].append(json.loads(paper.json_res))
        return json.dumps(data_dict)
