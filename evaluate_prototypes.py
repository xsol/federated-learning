import torch
from torch.utils.tensorboard import SummaryWriter

from feature_extractor import feature_extractor
from prototypes import prototype
from prototypes import prototype_ops
from prototypes import prototype_learning
from feature_extractor.net10 import Net10
from common import common
import pathlib

def main():
    print(150*"-")
    device = common.get_device()

    # Settings
    BATCH_SIZE = 1

    FEATURE_EXTRACTOR_TAG = "mnist_full_10epo"

    PROTOTYPES_TAG = "fed;58;2;[0, 2, 3, 5, 7, 9]"
    AUTOMATIC_PROTOTYPE = True

    # eval labels are extracted from metadata
    # EVAL_LABELS = [0, 1, 2, 3, 4]

    # load prototypes and metadata
    settings = common.load_metadata(PROTOTYPES_TAG, AUTOMATIC_PROTOTYPE)
    prototypes = prototype_ops.load_prototypes(PROTOTYPES_TAG, device, AUTOMATIC_PROTOTYPE)
    print(f"loaded {len(prototypes)} prototypes")
    print(f'prototypes were trained on labels: {settings["labels"]} with tau={settings["tau"]}, m={settings["m"]}, init_goodness={settings["goodness"]}')

    # load dataset
    assert (BATCH_SIZE == 1)
    data_loader = common.load_mnist_by_labels(BATCH_SIZE, settings["labels"], train=False)

    # feature extractor
    fe = feature_extractor.FeatureExtractor(Net10(), FEATURE_EXTRACTOR_TAG, device)
    print(f"prepared {fe.dim_featurespace}-dimensional feature extractor")

    prediction_stats = prototype_learning.inference(data_loader, settings, prototypes, fe, device)

    base_dir = pathlib.Path(__file__).parent.resolve()
    writer = SummaryWriter(f'{base_dir}/runs/{PROTOTYPES_TAG}')
    _, outputs, labels = fe.extract_features_dataset(data_loader, FEATURE_EXTRACTOR_TAG)
    common.visualize_proto_and_out(writer, outputs, labels, prototypes)
    common.visualize_accuracy(writer, prediction_stats)
    writer.close()

if __name__ == '__main__':
    main()