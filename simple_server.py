import cv2
import SocketServer
import urllib
from os import remove
from time import time
from BaseHTTPServer import BaseHTTPRequestHandler
from urlparse import parse_qs
from src.options import DEBUG, API_KEYS
from src.raw_photo import RawPhoto

PORT = 8012
URL_HEAD = '/scanner/check-now'

##
# requires `options.py` in src which contains API_KEY and DEBUG option
##


class RequestHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_GET(self):
        if DEBUG:
            print(self.path, self.path[:len(URL_HEAD)])

        # has to start with URL_HEAD
        if self.path[:len(URL_HEAD)] != URL_HEAD:
            print('> != URL_HEAD')
            return
        # has to have GET parameter
        if len(self.path) <= len(URL_HEAD) + 1:  # 1 filters `?`
            print('> NO GET PARAMETER')
            return
        # has to have url in GET parameter
        get_param = parse_qs(self.path[len(URL_HEAD) + 1:])
        if 'url' not in get_param or 'key' not in get_param or \
                        'num_papers' not in get_param or 'num_questions' not in get_param:
            print('> `url`, `key`, `num_papers`, OR `num_questions` NOT IN PARAMETERS')
            return

        url = get_param['url'][0]
        key = get_param['key'][0]
        num_papers = get_param['num_papers'][0]
        num_questions = get_param['num_questions'][0]

        if key not in API_KEYS:
            print('> WRONG API KEY')
            return

        print('> PROCESSING REQUEST...')
        try:
            print('  - downloading raw photo')
            filepath = 'tmp/%d.jpg' % int(time() * 1000)
            f = open(filepath, 'wb+')
            f.write(urllib.urlopen(url).read())
            f.close()
        except:
            print('> CANNOT DOWNLOAD PICTURE %s' % url)
            return

        # CV stuff
        print('  - cv module starts')
        test_img = cv2.imread(filepath, 0)
        rp = RawPhoto(test_img, num_papers, num_questions)
        res = rp.dump_data()
        rp.paper_objs = []
        print('  - cv module done')
        print(res)

        # clean up
        remove(filepath)

        # send header
        self.send_response(200)
        self.send_header("Content-type", "text/json")
        self.end_headers()

        self.wfile.write(res)
        self.wfile.close()


if __name__ == '__main__':
    httpd = SocketServer.TCPServer(("", PORT), RequestHandler)
    httpd.serve_forever()
