import cv2
import numpy as np
import json
from pylibdmtx.pylibdmtx import decode
from options import DEBUG

# Adaptive threshold
THR_MAX_VAL = 255
THR_BLOCK_SIZE = 29
THR_OFFSET = 8

# Datamatrix
READ_DATAMATRIX_TIMEOUT = 3
DATAMATRIX_REGION = (40, 140, 740, 840)
DATAMATRIX_SPLIT = 5

# Paper segmentation
NUM_ROWS = 15
NUM_COLS = 4
NUM_QUESTIONS = NUM_ROWS * NUM_COLS
SEGMENT_OFFSET = 12
COL_END_PTS = [[87, 227], [266, 406], [446, 586], [624, 764]]
ROW_RANGE = (306, 1126)
TRIM_ROWS_TO_SCAN = 22
TRIM_THRESHOLD = 0.5

# Answer scanning
NUM_OPTIONS = 5
LR_MARGIN = 7
TB_MARGIN = 17
THRESHOLD_SD_MULTIPLIER = 1.5
SD_TO_MEAN_RATIO_THRESHOLD = 0.065
IDX_TO_LETTER = ['A', 'B', 'C', 'D', 'E']


def remove_edges(ans_img_raw, ans_img_thr):
    """
    Trims the edges of each block by traversing lines inverse until we _hit and pass_ a black line.
    Four sides are rather repetitive and could be refactored.
    :param ans_img_raw: raw image of the answer block
    :param ans_img_thr: binary image of the answer block
    :return: trimmed image of the answer block, raw and binary (we need the raw picture for reading the answers)
    """
    h, w = ans_img_thr.shape[:2]

    # Top
    t = 0
    flag = False
    for t in range(TRIM_ROWS_TO_SCAN):
        if sum(ans_img_thr[t]) > TRIM_THRESHOLD * w * 255:
            flag = True
        else:
            if flag:
                break
    if t == TRIM_ROWS_TO_SCAN - 1:
        t = 0

    # Bottom
    b = 0
    flag = False
    for b in range(TRIM_ROWS_TO_SCAN):
        if sum(ans_img_thr[- (b + 1)]) > TRIM_THRESHOLD * w * 255:
            flag = True
        else:
            if flag:
                break
    if b == TRIM_ROWS_TO_SCAN - 1:
        b = 0

    # Left
    l = 0
    flag = False
    for l in range(TRIM_ROWS_TO_SCAN):
        if sum(ans_img_thr[:, l]) > TRIM_THRESHOLD * h * 255:
            flag = True
        else:
            if flag:
                break
    if l == TRIM_ROWS_TO_SCAN - 1:
        l = 0

    # Right
    r = 0
    flag = False
    for r in range(TRIM_ROWS_TO_SCAN):
        if sum(ans_img_thr[:, - (r + 1)]) > TRIM_THRESHOLD * h * 255:
            flag = True
        else:
            if flag:
                break
    if r == TRIM_ROWS_TO_SCAN - 1:
        r = 0

    return ans_img_raw[t:h - b, l:w - r], ans_img_thr[t:h - b, l:w - r]


class PaperScan:
    """
    Models each answer sheet paper.
    """
    raw_img = None
    thr_img = None
    test_id = None
    paper_id = None
    ans_imgs_raw = [None] * NUM_QUESTIONS
    ans_imgs_thr = [None] * NUM_QUESTIONS
    marked_ans = [0] * NUM_QUESTIONS
    metadata = ''
    json_res = None

    def __init__(self, raw_img):
        """
        Initializes and processes the paper. This is the only top-level function that needs to be called.
        Called only by its parent RawPhoto object.
        :param raw_img: raw paper image
        """
        self.raw_img = raw_img
        self.thr_img = cv2.adaptiveThreshold(self.raw_img, THR_MAX_VAL, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, THR_BLOCK_SIZE, THR_OFFSET)
        self.read_datamatrix()
        self.segment()
        self.read_answer()
        data_dict = {'test_id': self.test_id,
                     'paper_id': self.paper_id,
                     'answers': self.marked_ans,
                     'metadata': self.metadata}
        self.json_res = json.dumps(data_dict)

    def read_datamatrix(self):
        """
        Reads the content datamatrix on the paper.
        """
        datamatrix_region = self.thr_img[DATAMATRIX_REGION[0]:DATAMATRIX_REGION[1],
                                         DATAMATRIX_REGION[2]:DATAMATRIX_REGION[3]]
        try:
            content = decode(datamatrix_region, timeout=READ_DATAMATRIX_TIMEOUT)[0][0]
            self.test_id = content[:DATAMATRIX_SPLIT]
            self.paper_id = content[DATAMATRIX_SPLIT:]
        except:
            self.test_id = '?????'
            self.paper_id = '???'
            self.metadata += 'Could not decode datamatrix.\n'
        if DEBUG:
            print('Test id: %s\tPaper id: %s' % (self.test_id, self.paper_id))

    def segment(self):
        """
        Segment the image into pieces of answer blocks
        """
        for i in range(NUM_QUESTIONS):
            col = int(i / NUM_ROWS)
            row = i % NUM_ROWS
            up = int(ROW_RANGE[0] + row * ((ROW_RANGE[1] - ROW_RANGE[0]) / (NUM_ROWS * 1.0))) - SEGMENT_OFFSET
            down = int(ROW_RANGE[0] + (row + 1) * ((ROW_RANGE[1] - ROW_RANGE[0]) / NUM_ROWS)) + SEGMENT_OFFSET
            left = COL_END_PTS[col][0] - SEGMENT_OFFSET
            right = COL_END_PTS[col][1] + SEGMENT_OFFSET
            ans_img_thr = self.thr_img[up:down, left:right]
            ans_img_raw = self.raw_img[up:down, left:right]
            ans_img_raw, ans_img_thr = remove_edges(ans_img_raw, ans_img_thr)
            self.ans_imgs_thr[i] = ans_img_thr
            self.ans_imgs_raw[i] = ans_img_raw
            if DEBUG:
                cv2.imwrite("../tmp/ans-img-thr-%s-%s-%d.png" % (self.test_id, self.paper_id, i), ans_img_thr)
                cv2.imwrite("../tmp/ans-img-raw-%s-%s-%d.png" % (self.test_id, self.paper_id, i), ans_img_raw)

    def read_answer(self):
        """
        Read the selection of each answer block
        Read from the raw image because Gaussian Adaptive Threshold treats any large block of content, either dark or
        bright, as background. Therefore, only the edges of the circles is detected. We need the inside content of the
        circle for accuracy.
        """
        for i in range(NUM_QUESTIONS):
            img = self.ans_imgs_raw[i]

            # Cut region into NUM_OPTIONS pieces
            img_height, img_width = img.shape[:2]
            block_width = int((img_width - 2 * LR_MARGIN) / NUM_OPTIONS)
            boundary_pts = [LR_MARGIN] + [0] * NUM_OPTIONS
            for j in range(NUM_OPTIONS):
                boundary_pts[j + 1] = boundary_pts[0] + (j + 1) * block_width

            # Count brightness
            brightness = [0] * NUM_OPTIONS
            for j in range(NUM_OPTIONS):
                for x in range(TB_MARGIN, img_height - TB_MARGIN):
                    for y in range(boundary_pts[j], boundary_pts[j + 1]):
                        brightness[j] += img[x, y]

            # Select answers
            mean = np.mean(brightness)
            sd = np.std(brightness)
            ans = []
            # if every field look very much alike, it's impossible that there is a selected answer
            # SD_TO_MEAN_RATIO_THRESHOLD minimizes the chance of misinterpretation in both directions by balancing two
            # normally distributed events (false positive and false negative), assuming iid distribution
            if sd / mean > SD_TO_MEAN_RATIO_THRESHOLD:
                threshold = mean - THRESHOLD_SD_MULTIPLIER * sd
                for j in range(NUM_OPTIONS):
                    if brightness[j] < threshold:
                        ans.append(IDX_TO_LETTER[j])
            self.marked_ans[i] = ans
            if DEBUG:
                print('Test %s, Paper %s, Question %d: %s' % (self.test_id, self.paper_id, i + 1, ans))
