import torch
from torch.utils.tensorboard import SummaryWriter

from feature_extractor import feature_extractor
from prototypes import prototype
from prototypes import prototype_ops
from feature_extractor.net10 import Net10
from feature_extractor.net20 import Net20
from feature_extractor.net100 import Net100
from prototypes import prototype_learning
from common import common

def main():
    print(150*"-")
    device = common.get_device()

    # Settings
    INITIAL_GOODNESS = 10.0 # goodness is inverse to learning rate
    NUM_PROTOTYPES = 40
    BATCH_SIZE = 1
    TAU = 0.7 #similarity threshold
    M = 7 # number of neighbors to update on false prediction

    LABELDNESS = 1.0 # leave at 1.0, is not put into metadata

    FEATURE_EXTRACTOR_TAG = "mnist_full_20dim_10epo"

    LOAD_PROTOTYPES = False # for retraining
    LOAD_PROTOTYPES_TAG = "labels_0-4"
    SAVE_PROTOTYPES = False
    SAVE_PROTOTYPES_TAG = "demo_labels_7-1"

    TRAIN_LABELS = [7, 1]

    # load dataset
    data_loader = common.load_mnist_by_labels(BATCH_SIZE, TRAIN_LABELS, train=True)

    # feature extractor
    fe = feature_extractor.FeatureExtractor(feature_extractor.map_tag_to_net(FEATURE_EXTRACTOR_TAG), FEATURE_EXTRACTOR_TAG, device)
    print(f"prepared {fe.dim_featurespace}-dimensional feature extractor")

    # init or load prototypes
    prototypes = []
    total_trained_labels = TRAIN_LABELS
    if LOAD_PROTOTYPES:
        metadata = common.load_metadata(LOAD_PROTOTYPES_TAG)
        for label in metadata["labels"]:
            if label not in total_trained_labels:
                total_trained_labels.append(label)

        prototypes = prototype_ops.load_prototypes(LOAD_PROTOTYPES_TAG, device)
        print(f"loaded {len(prototypes)} prototypes")
    else:
        prototypes = [prototype.Prototype(fe.dim_featurespace, i, INITIAL_GOODNESS) for i in range(1, NUM_PROTOTYPES+1)]
        print(f"initialized {len(prototypes)} prototypes with goodness value {INITIAL_GOODNESS}")

    prototypes = prototype_learning.continual_learning(M, TAU, prototypes, data_loader, LABELDNESS, fe)

    if SAVE_PROTOTYPES:
        common.save_metadata(SAVE_PROTOTYPES_TAG, INITIAL_GOODNESS, TAU, M, NUM_PROTOTYPES, total_trained_labels, [], "mnist", FEATURE_EXTRACTOR_TAG, False) #TODO stack training labels when retraining
        prototype_ops.save_prototypes(SAVE_PROTOTYPES_TAG, prototypes)
        print(f"saved prototypes and metadata with tag: {SAVE_PROTOTYPES_TAG}")

if __name__ == '__main__':
    main()
