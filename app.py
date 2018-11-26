from flask import Flask, request
from os.path import join

import subprocess
import signal
import time
import json
import atexit

STATIC_PATH = 'static'

app = Flask(__name__)

capture_prog = None
finish_sig_sent = False
dim_sig_sent = False
run_w_debug = 0

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/test')
def test():
    return app.send_static_file('test.html')

@app.route('/test_data')
def test_data():
    with open('test_data.txt') as f:
        js = json.loads(f.read())
    return js, 200

@app.route('/dimensions')
def dims():
    global capture_prog, dim_sig_sent

    if capture_prog is None:
        return 'failure', 500

    width = request.args.get('width')
    height = request.args.get('height')
    dims = {
        'width': width,
        'height': height
    }

    with open('dimensions.txt', 'w') as f:
        f.write(json.dumps(dims))

    if not dim_sig_sent:
        capture_prog.send_signal(signal.SIGUSR1)
        dim_sig_sent = True

    return 'success', 200

@app.route('/scroll_info', methods=['POST'])
def scroll():
    scrollInfo = json.loads(request.data)

    with open('scroll_data.txt', 'a') as f:
        f.write('\n'.join(['%d,%d' % (d['diff'], d['timestamp']) for d in scrollInfo]))
        f.write('\n')

    return 'success', 200
 
@app.route('/start_capture')
def start():
    global capture_prog
    capture_prog = subprocess.Popen(['python', 'capture_scroll.py'])
    return 'success', 200

@app.route('/finish_calibration')
def finish():
    global capture_prog, finish_sig_sent

    if capture_prog is None:
        return 'failure', 500

    if not finish_sig_sent:
        capture_prog.send_signal(signal.SIGUSR1)
        finish_sig_sent = True

    return 'success', 200

def compile_data():
    global run_w_debug

    if run_w_debug == 0:
        return

    scrollf = open('scroll_data.txt', 'r')
    dispf = open('displacement_data.txt', 'r')

    with open('data.txt', 'a'):
        for line in scrollf:
            vals = line.split(',')
            amt_scrolled, ts = float(vals[0]), int(vals[1])

    scroll.close()
    dispf.close()

if __name__ == '__main__':
    #atexit.register(compile_data)
    app.run(port=8080)