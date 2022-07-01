from kmeans import kmeans

DTYPE = float

# FILE = './inputs/input_2d.txt'
FILE = './inputs/2d_100.txt'

MAX_TRIES = 10
MAX_ITER = 10

SHOW_INTERMEDIATE_PLOT = False
SHOW_FINAL_PLOT = True

n_clusters = 4

kmeans(FILE, n_clusters)