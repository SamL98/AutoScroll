from matplotlib.patches import Circle, Rectangle, Ellipse
import matplotlib.pyplot as plt

def circle(x, y, r):
    return Circle((x, y), r, edgecolor='r', facecolor='None')

def ellipse(x, y, w, h, t):
    return Ellipse((x, y), w, h, angle=t)

def rect(x, y, w, h):
    return Rectangle((x, y), w, h, edgecolor='r', facecolor='None')

def retax(n):
    _, ax = plt.subplots(1, n)
    return ax