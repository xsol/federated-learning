from common import common
from prototypes import prototype_ops
from federated_learning import federated_learning

def main():

    # settings
    PROTOTYPES_TAGS = ["demo_labels_5-6", "demo_labels_7-1"] # prototypes to federate
    SAVE_TAG = "fed_demo_1,5,6,7"
    THRESHOLD_GOODNESS = 10.0 # all prototypes below will be removed
    THRESHOLD_SIMILARITY = 0.95 # threshold above which prototypes can be merged

    # load prototypes
    clients = []
    device = common.get_device()
    for tag in PROTOTYPES_TAGS:
        clients.append({"tag": tag, "meta": common.load_metadata(tag), "prototypes": prototype_ops.load_prototypes(tag, device)})
    
    federated_client, _ = federated_learning.federate(clients, THRESHOLD_GOODNESS, THRESHOLD_SIMILARITY, device)

    meta = clients[0]["meta"]
    additionale_labels = [proto.label for proto in federated_client if proto.label not in meta["labels"]]
    meta["labels"].extend(additionale_labels)
    common.save_metadata(SAVE_TAG, meta["goodness"], meta["tau"], meta["m"], meta["num_prototypes"], meta["labels"], PROTOTYPES_TAGS, meta["dataset_tag"], meta["feature_extractor_tag"], False)
    prototype_ops.save_prototypes(SAVE_TAG, federated_client)




if (__name__ == "__main__"):
    main()