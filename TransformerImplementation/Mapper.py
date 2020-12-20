
import copy

def MapLinearLayer(from_, to_):
    to_.weight = copy.deepcopy(from_.weight)
    to_.bias = copy.deepcopy(from_.bias)

def MapLayerNorm(from_, to_):
    to_.elementwise_affine = from_.elementwise_affine
    to_.weight = copy.deepcopy(from_.weight)
    to_.bias = copy.deepcopy(from_.bias)
    to_.eps = from_.eps

def MapEmbedding(from_, to_):
    to_.weight = copy.deepcopy(from_.weight)