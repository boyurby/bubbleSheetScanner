# Bubble Sheet Scanner

Allows teachers to scan multiple-choice question answer sheets by taking a
pictures of them by their phones.

`RawPhoto` and `PaperScan` are the implementation of the core functionalities
and `RequestHandler` acts as an API that takes input from GET request when
hosted on a server.

This is the computer vision module of the project. An Android app is being
developed.


## Implementation

- The raw image is binarized by Gaussian adaptive thresholding
- Contours in the image are identified
- Each contour is approximated to a polygon
- Those polygons that have four sides and are convex are selected
- The `num_papers` largest such polygons are identified as the main tables of
  the papers, unless they are significantly smaller than the largest identified
  table (since all papers should have approximately the same size)
- For each of these tables:
  - Based on the four corner points, the midpoints of each side is identified
  - Based on the midpoints of each side, the points just outside the midpoints
    are identified as reference points
  - The reference point with the darkest neighbor is on the left side due to
    the black rectangle on the bubble sheet template
  - The table is oriented to the correct direction
  - A perspective transformation is applied to the rectangle and the corners
    are mapped to the corresponding locations of the template
  - A transformed image of this paper is obtained
  - The region that contains the datamatrix is trimmed and decoded
  - The answer table is segmented to each answer field, with some margin
    conservatively reserved
  - For each answer field:
    - By traversing inwards past a black line, the margin is trimmed
    - The answer field is now accurately aligned with the template
    - The answer field is segmented into 5 regions (one for each option)
    - The darkest region is considered to be marked by the student. The answer
      for this question is obtained
  - All data associated with this paper are obtained
- All data associated with this raw image are obtained


## Example usage

[Sample bubble sheet](bubble_sheet/sample.jpeg)

This scanner can be used by creating a RawPhoto class from a raw photo taken:

```Python
def direct_use(image_path):
    test_img = cv2.imread(image_path, 0)
    rp = RawPhoto(test_img, 2, 30)
    res = rp.dump_data()
```

The API accepts request in the following format:

```
http://host:port/scanner/check-now?url=file_url&num_papers=m&num_questions=n&key=api_key
```

where `file_url` is the url to the image file; `m` is the number of papers in
the image, `n` is the number of questions of the test, and `api_key` is the
issued API key.

Both the `dump_data()` method of the `RawPhoto` class and the API returns the
scanned result as a JSON string. The following is an example of the returned
data.

```JSON
{
  "metadata": "1 paper(s) not detected.\n",
  "papers": [{
    "test_id": "00000",
    "paper_id": "000",
    "answers": ["A", "B", "C", "D", "E", "E", "D", "C", "B", "A", "A", "B", "C", "D", "E", "E", "D", "C", "B", "C", "E", "A", "D", "C", "D", "A", "A", "B", "E", "D", "E", "A", "E", "C", "E", "D", "A", "E", "E", "C", "E", "E", "A", "E", "B", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "metadata": ""
  }, {
    "test_id": "?????",
    "paper_id": "???",
    "answers": ["A", "C", "D", "D", "A", "C", "B", "A", "C", "D", "D", "A", "C", "D", "D", "B", "C", "A", "E", "D", "D", "D", "E", "E", "E", "E", "E", "D", "D", "D", "E", "A", "E", "C", "E", "D", "A", "E", "E", "C", "E", "E", "A", "E", "B", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "metadata": "Could not decode datamatrix.\n"
  }]
}
```

In this example, the user requested 3 papers with 45 questions, where there are
only 2 papers, and one of them has incomplete datamatrix.

The top level metadata indicates at least one paper that the user requests was
not detected. The lower level metadata of the second paper indicates that it
was unable to decode its datamatrix. The test ID and paper ID are in turn set
to question marks.

Because there are 60 fields on the bubble sheets but the user only requests 45
answers, the last 15 answers are set to `0`.

In practice, being unable to decode the datamatrix usually indicates that the
entire paper is not read correctly. The rest of the returned data of that paper
are hence not reliable. If the program reads the datamatrix correctly, however
the answers are very likely to be correctly read.


## TODOs
- Refactor bubble sheet generator code
- Android app development
