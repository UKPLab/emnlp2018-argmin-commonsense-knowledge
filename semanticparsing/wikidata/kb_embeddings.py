import numpy as np

all_zeroes = "ALL_ZERO"
unknown_el = "_UNKNOWN"


def load_kb_embeddings(path_to_folder):
    """
    Loads pre-trained KB embeddings from the specified path.

    @return (embeddings as an numpy array, relation embeddings, entity2idx, relation2idx)
    """

    entity2idx = {}
    allowed_indices = set()
    with open("../data/entity2id.filtered.txt", 'r') as f:
        f.readline()
        for l in f.readlines():
            k, v, idx = tuple(l.strip().split("\t"))
            entity2idx[k] = int(idx) + 2
            allowed_indices.add(int(v))

    embeddings = []
    with open(path_to_folder + "/entity2vec.vec", 'r') as f:
        idx = 0
        for line in f.readlines():
            if idx in allowed_indices:
                split = line.strip().split('\t')
                embeddings.append([float(num) for num in split])
            idx += 1

    entity2idx[all_zeroes] = 0  # 0 is reserved for padding
    entity2idx[unknown_el] = 1  # 1 is reserved for OOV
    embedding_size = len(embeddings[0])
    vector_oov = 2 * 0.1 * np.random.rand(embedding_size) - 0.1
    embeddings = np.asarray([[0.0]*embedding_size, vector_oov] + embeddings, dtype='float32')

    print("KB embeddings loaded: {}".format(embeddings.shape))

    return entity2idx, embeddings
