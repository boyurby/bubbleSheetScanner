import cv2
import json

import sys
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
MAX_NUM_QUESTIONS = NUM_ROWS * NUM_COLS
SEGMENT_OFFSET = 12
COL_END_PTS = [[87, 227], [266, 406], [446, 586], [624, 764]]
ROW_RANGE = (306, 1126)
TRIM_ROWS_TO_SCAN = 22
TRIM_THRESHOLD = 0.5

# Answer scanning
NUM_OPTIONS = 5
LEFT_RIGHT_MARGIN = 7
VERTICAL_SCAN_RANGE = [7, 30]
IDX_TO_LETTER = ['A', 'B', 'C', 'D', 'E']
NORMALIZE_SCAN_RANGE_X = [0, 0.5]
NORMALIZE_SCAN_RANGE_Y = [0, 1]
NORMALIZE_TAIL_PROPORTION = 0.01
GAP_THRESHOLD = 0.74


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


def max_and_min(img):
    """
    Finds the 3% percentiles of the brightest and darkest point in a picture
    :param img: picture to check
    :return: (max, min) as a tuple
    """
    brightness = []
    h, w = img.shape[:2]
    for i in range(int(h * NORMALIZE_SCAN_RANGE_X[0]), int(h * NORMALIZE_SCAN_RANGE_X[1])):
        for j in range(int(w * NORMALIZE_SCAN_RANGE_Y[0]), int(w * NORMALIZE_SCAN_RANGE_Y[1])):
            brightness.append(img[i, j])
    brightness.sort()
    max_ref = brightness[int(len(brightness) * (1 - NORMALIZE_TAIL_PROPORTION))]
    min_ref = brightness[int(len(brightness) * NORMALIZE_TAIL_PROPORTION)]
    return max_ref, min_ref


class PaperScan:
    """
    Models each answer sheet paper.
    """
    raw_img = None
    thr_img = None
    test_id = None
    paper_id = None
    num_questions = 0
    ans_imgs_raw = [None] * MAX_NUM_QUESTIONS
    ans_imgs_thr = [None] * MAX_NUM_QUESTIONS
    marked_ans = [0] * MAX_NUM_QUESTIONS
    metadata = ''
    json_res = None

    def __init__(self, raw_img, num_questions=MAX_NUM_QUESTIONS):
        """
        Initializes and processes the paper. This is the only top-level function that needs to be called.
        Called only by its parent RawPhoto object.
        :param raw_img: raw paper image
        :param num_questions: number of questions in the paper
        """
        self.raw_img = raw_img
        self.num_questions = num_questions
        self.thr_img = cv2.adaptiveThreshold(self.raw_img, THR_MAX_VAL, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, THR_BLOCK_SIZE, THR_OFFSET)
        self.read_datamatrix()
        self.segment()
        self.read_all_answers_single()
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
        for i in range(MAX_NUM_QUESTIONS):
            col = int(i / NUM_ROWS)
            row = i % NUM_ROWS
            up = int(ROW_RANGE[0] + row * ((ROW_RANGE[1] - ROW_RANGE[0]) / (NUM_ROWS * 1.0))) - SEGMENT_OFFSET
            down = int(ROW_RANGE[0] + (row + 1) * ((ROW_RANGE[1] - ROW_RANGE[0]) / NUM_ROWS)) + SEGMENT_OFFSET
            left = COL_END_PTS[col][0] - SEGMENT_OFFSET
            right = COL_END_PTS[col][1] + SEGMENT_OFFSET
            ans_img_thr = self.thr_img[up:down, left:right]
            ans_img_raw = self.raw_img[up:down, left:right]
            self.ans_imgs_thr[i] = ans_img_thr
            self.ans_imgs_raw[i] = ans_img_raw

    def read_all_answers_single(self):
        """
        Read the selection of each answer block
        Only one selections allowed, high reliability
        Read from the raw image because Gaussian Adaptive Threshold treats any large block of content, either dark or
        bright, as background. Therefore, only the edges of the circles is detected. We need the inside content of the
        circle for accuracy.
        """
        for i in range(self.num_questions):
            raw_img = self.ans_imgs_raw[i]
            thr_img = self.ans_imgs_thr[i]
            img, _ = remove_edges(raw_img, thr_img)
            img_height, img_width = img.shape[:2]

            # Cut region into NUM_OPTIONS pieces
            img_height, img_width = img.shape[:2]
            block_width = int((img_width - 2 * LEFT_RIGHT_MARGIN) / NUM_OPTIONS)
            boundary_pts = [LEFT_RIGHT_MARGIN] + [0] * NUM_OPTIONS
            for j in range(NUM_OPTIONS):
                boundary_pts[j + 1] = boundary_pts[0] + (j + 1) * block_width

            # Count brightness
            lowest_val, lowest_idx = sys.maxint, 0
            for j in range(NUM_OPTIONS):
                this_brightness = 0
                for x in range(VERTICAL_SCAN_RANGE[0], VERTICAL_SCAN_RANGE[1]):
                    for y in range(boundary_pts[j], boundary_pts[j + 1]):
                        this_brightness += img[x, y]
                if this_brightness < lowest_val:
                    lowest_val, lowest_idx = this_brightness, j
            self.marked_ans[i] = IDX_TO_LETTER[lowest_idx]

    def raad_all_answers_multiple(self):
        """
        Read the selection of each answer block
        Multiple selections allowed, low reliability
        Read from the raw image because Gaussian Adaptive Threshold treats any large block of content, either dark or
        bright, as background. Therefore, only the edges of the circles is detected. We need the inside content of the
        circle for accuracy.
        """
        for i in range(self.num_questions):
            raw_img = self.ans_imgs_raw[i]
            thr_img = self.ans_imgs_thr[i]
            max_ref, min_ref = max_and_min(raw_img)
            if DEBUG:
                print max_ref, min_ref
            img, _ = remove_edges(raw_img, thr_img)
            if DEBUG:
                cv2.imwrite("tmp/ans-img-raw-%s-%s-%d.png" % (self.test_id, self.paper_id, i), img)

            # Cut region into NUM_OPTIONS pieces
            img_height, img_width = img.shape[:2]
            block_width = int((img_width - 2 * LEFT_RIGHT_MARGIN) / NUM_OPTIONS)
            boundary_pts = [LEFT_RIGHT_MARGIN] + [0] * NUM_OPTIONS
            for j in range(NUM_OPTIONS):
                boundary_pts[j + 1] = boundary_pts[0] + (j + 1) * block_width

            # Count brightness
            brightness = [0] * NUM_OPTIONS
            temp = [0] * NUM_OPTIONS        # logs (this_brightness - black) / (white - black) values
            for j in range(NUM_OPTIONS):
                for x in range(VERTICAL_SCAN_RANGE[0], VERTICAL_SCAN_RANGE[1]):
                    for y in range(boundary_pts[j], boundary_pts[j + 1]):
                        brightness[j] += img[x, y]
                num_pts = (VERTICAL_SCAN_RANGE[1] - VERTICAL_SCAN_RANGE[0]) * (boundary_pts[j + 1] - boundary_pts[j])
                brightness[j] /= num_pts
                temp[j] = '%.4f' % ((brightness[j] - min_ref) / float(max_ref - min_ref))
            if DEBUG:
                print brightness, temp

            # Select answers
            threshold = min_ref + (max_ref - min_ref) * GAP_THRESHOLD
            ans = ''
            for j in range(NUM_OPTIONS):
                if brightness[j] < threshold:
                    ans += IDX_TO_LETTER[j]
            self.marked_ans[i] = ans
            if DEBUG:
                print('Test %s, Paper %s, Question %d: %s' % (self.test_id, self.paper_id, i + 1, ans))
