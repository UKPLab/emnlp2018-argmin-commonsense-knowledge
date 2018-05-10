import pickle
import numpy as np

all_zeroes = "ALL_ZERO"
unknown_el = "_UNKNOWN"


def load_fn_embeddings(path_to_file):
    """
    @return (embeddings as an numpy array, entity2idx)
    """

    frame2idx = {}
    with open(path_to_file, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        emb_dict = u.load()

    embeddings = []
    idx = 2
    for key, emb in emb_dict.items():
        frame2idx[key] = idx
        embeddings.append(emb)
        idx += 1

    frame2idx[all_zeroes] = 0  # 0 is reserved for padding
    frame2idx[unknown_el] = 1  # 1 is reserved for OOV
    embedding_size = len(embeddings[0])
    vector_oov = 2 * 0.1 * np.random.rand(embedding_size) - 0.1
    embeddings = np.asarray([[0.0]*embedding_size, vector_oov] + embeddings, dtype='float32')

    print("FrameNet embeddings loaded: {}".format(embeddings.shape))

    return frame2idx, embeddings
