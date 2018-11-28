from Quartz.CoreGraphics import CGEventCreateScrollWheelEvent, CGEventPost, kCGHIDEventTap
from multiprocessing.connection import Listener
from time import sleep

_scroll_amt = 0
_scrolling = False

def is_scrolling():
    global _scrolling
    return _scrolling

def change_scroll_amt(scroll_amt):
    global _scroll_amt
    
    sign_orig = _scroll_amt/abs(_scroll_amt)
    sign_new = scroll_amt/abs(scroll_amt)

    if sign_orig != sign_new:
        scroll_by(scroll_amt)
    else:
        _scroll_amt += scroll_amt

def scroll_by(scroll_amt):
    if scroll_amt == 0:
        return

    global _scroll_amt, _scrolling
    _scroll_amt = scroll_amt

    #print(_scroll_amt)

    step = int(scroll_amt/abs(scroll_amt))
    amt_scrolled = 0
    _scrolling = True

    while amt_scrolled < abs(_scroll_amt):
        sleep(.005)
        #sleep(.01)

        multiplier = 1 - (float(amt_scrolled+1) / scroll_amt)
        speed = 4*multiplier*step
        event = CGEventCreateScrollWheelEvent(None, 0, 1, speed)
        CGEventPost(kCGHIDEventTap, event)

        amt_scrolled += 1

    _scrolling = False


if __name__ == '__main__':
    listener = Listener(('localhost', 6000), authkey=b'password')
    conn = listener.accept()

    while True:
        msg = conn.recv()

        if 'scroll' in msg:
            scroll_amt = float(msg[msg.index(':')+1:])

            # if _scrolling:
            #     scroll_by(scroll_amt)
            # else:
            #     change_scroll_amt(scroll_amt)

            scroll_by(scroll_amt)

        if msg == 'close':
            conn.close()
            break

    listener.close()
    