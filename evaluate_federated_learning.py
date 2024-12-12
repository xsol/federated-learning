from feature_extractor import feature_extractor
from common import common
from prototypes import prototype_learning
from prototypes import prototype
from prototypes import prototype_ops
from federated_learning import federated_learning
import random, pathlib
from torch.utils.tensorboard import SummaryWriter

def main():
    
    param = {
        "EVAL_TYPE":"GENERIC",                          # type of evaluation: "GENERIC" or "APPLICATION"

        # general
        "DATASET_TAG": "mnist",                         # Train and eval dataset
        "FEATURE_EXTRACTOR_TAG": "mnist_full_10epo",  # static feature extractor cnn
        "LABELDNESS_STEPS": [0.1, 0.5, 0.7],            # train prototypes on datasets with set percentage of labeled data 
        "NUM_DATASET_VARIATIONS": 50,                   # num of different selections of classes, samples for training
        "NUM_FEDERATED_CLIENTS": 500,                   # number of clients to federate
        "MAX_CLASSES_COEFF": 0.5,                       # number of classes for data subset is random.uniform(1,num_classes*MAX_CLASSES_COEFF). use to reduce number of trained classes

        # continual prototype learning
        "M": 7,                                         # number of closest prototypes to shift                             
        "TAU": 0.7,                                     # novelty detection threshold
        "INITIAL_GOODNESS": 10.0,                       # 
        "NUM_PROTOTYPES": 40,                           # 
        "BATCH_SIZE": 1,                                # 

        # federated learning
        "THRESHOLD_SIMILARITY": 0.95,                   # prototypes below threshold are merged
        "THRESHOLD_GOODNESS": 1.0,                    # remove prototypes below threshold
        "MAX_FEDERATION_PARTICIPANTS": 7,               # 
        "RESOLVE_CONFLICTS": False,                      # resolve conflict between close prototypes of different label
    }

    #train_prototypes(param)
    #federate_prototypes(param)
    evaluate_clients(param)

def evaluate_client(base_dir, client_tags, i, batch_size, meta, device, clients, parent_evals):
    writer = SummaryWriter(f'{base_dir}/runs/auto/{client_tags[i]}')

    # prepare dataset
    num_classes = common.map_tag_to_num_classes(meta[i]["dataset_tag"])
    data_loader = common.load_dataset(batch_size, meta[i]["dataset_tag"], num_classes, False, 0.0, False, meta[i]["labels"], False)
    
    # prepare feature extractor
    fe = feature_extractor.FeatureExtractor(feature_extractor.map_tag_to_net(meta[i]["feature_extractor_tag"]), meta[i]["feature_extractor_tag"], device)

    # eval parents
    parent_stats = []
    num_unlabeled_prototypes = 0
    parents_num_prototypes = []
    new_parent_evals = {}
    for parent_tag in meta[i]["parents"]:
        parent_meta = common.load_metadata(parent_tag, True)
        parent_proto = prototype_ops.load_prototypes(parent_tag, device, True)
        parent_data_loader = common.load_dataset(batch_size, parent_meta["dataset_tag"], num_classes, False, 0.0, False, parent_meta["labels"], False)
        if parent_tag in parent_evals.keys():
            parent_stat = parent_evals[parent_tag]
        else:
            parent_stat = prototype_learning.inference(parent_data_loader, parent_meta, parent_proto, fe, device)
            new_parent_evals.update({parent_tag: parent_stat})
            parents_num_prototypes.append(len([proto for proto in parent_proto if proto.allocated]))

        common.visualize_accuracy(writer, parent_stat, f"parent {parent_tag}")
        common.visualize_proto(writer, parent_proto, f"parent {parent_tag}")
        common.visualize_goodness_distribution(writer, [proto.goodness for proto in parent_proto if proto.allocated], f"parent {parent_tag}")
        parent_stats.append(parent_stat)
        num_unlabeled_prototypes += len([proto.label for proto in parent_proto if proto.allocated and proto.label < 0])

    # eval client
    stats = prototype_learning.inference(data_loader, meta[i], clients[i], fe, device)
    best_possible_stats = common.best_possible_stats(parent_stats)
    # loss per class by number of participating clients for that label

    common.visualize_accuracy(writer, stats, "federated client")
    common.visualize_accuracy(writer, best_possible_stats, "best possible accuracy")
    common.visualize_proto(writer, clients[i], "federated client")
    loss_info = {"loss":common.calculate_loss(stats, best_possible_stats),
                "num_fed_participants":len(meta[i]["parents"]),
                "num_prototypes":len([proto for proto in clients[i] if proto.allocated]),
                "num_merges":meta[i]["num_merges"],
                "num_conflicts_resolved":meta[i]["num_conflicts_resolved"],
                "num_unlabeled_prototypes_parents":num_unlabeled_prototypes,
                "goodness":[proto.goodness for proto in clients[i] if proto.allocated],
                "num_prototypes_parents": parents_num_prototypes}
    writer.close()
    #q.put([loss_info, new_parent_evals])
    return loss_info, new_parent_evals

def evaluate_clients(param):

    batch_size = param["BATCH_SIZE"]
    device = common.get_device()
    # eval a federated client,
    # then eval the parents and take max of respective class accuracy on test data
    # calculate difference of max(parents) and eval results

    # get all clients (fed or not)
    client_tags = common.get_tags_auto()
    clients = [prototype_ops.load_prototypes(tag, device, True) for tag in client_tags]
    meta = [common.load_metadata(tag, True) for tag in client_tags]

    federated_indices = [i for i in range(len(client_tags)) if meta[i]["parents"] != []]
    base_dir = pathlib.Path(__file__).parent.resolve()

    writer_total = SummaryWriter(f'{base_dir}/runs/auto/total')
    loss_infos = []
    parent_evals = {}

    for i in federated_indices:
        loss_info, parent_eval = evaluate_client(base_dir, client_tags, i, batch_size, meta, device, clients, parent_evals,)
        loss_infos.append(loss_info)
        parent_evals.update(parent_eval)   

    common.visualize_loss_distribution(writer_total, loss_infos)
    common.visualize_loss_by_federation_participants(writer_total, loss_infos)
    common.visualize_federation_participants_distribution(writer_total, loss_infos)
    common.visualize_loss_by_num_prototypes(writer_total, loss_infos)
    common.visualize_loss_by_num_merges(writer_total, loss_infos)
    common.visualize_loss_by_num_conflicts_resolved(writer_total, loss_infos)
    common.visualize_loss_by_num_unlabeled_prototypes_parents(writer_total, loss_infos)
    common.visualize_num_prototypes_distribution(writer_total, loss_infos)
    common.visualize_num_unlabeled_prototypes_parents_distribution(writer_total, loss_infos)
    common.visualize_num_merges_distribution(writer_total, loss_infos)
    common.visualize_num_conflicts_resolved_distribution(writer_total, loss_infos)
    common.visualize_num_prototypes_parents_distribution(writer_total, loss_infos)
    common.write_params(writer_total, param)

    # TODO visualize where false classifications are going

def federate_prototypes(param):
    NUM_FEDERATED_CLIENTS = param["NUM_FEDERATED_CLIENTS"]
    THRESHOLD_GOODNESS = param["THRESHOLD_GOODNESS"]
    MAX_FEDERATION_PARTICIPANTS = param["MAX_FEDERATION_PARTICIPANTS"]
    THRESHOLD_SIMILARITY = param["THRESHOLD_SIMILARITY"]
    RESOLVE_CONFLICTS = param["RESOLVE_CONFLICTS"]

    device = common.get_device()
    assert MAX_FEDERATION_PARTICIPANTS > 1

    # load all prototypes and metadata
    client_tags = common.get_tags_auto()
    clients = [prototype_ops.load_prototypes(tag, device, True) for tag in client_tags]
    meta = [common.load_metadata(tag, True) for tag in client_tags]
    permutation = [i for i in range(len(clients))]

    for i in range(NUM_FEDERATED_CLIENTS):
        # pick number of participants
        num_participants = round(random.uniform(2, MAX_FEDERATION_PARTICIPANTS))

        # pick participants
        while True:
            random.shuffle(permutation)
            participant_indices = permutation[0:num_participants]
            if common.check_dataset_compatability(meta, participant_indices):
                break


        fed_clients = []
        labels_all = []
        for p in participant_indices:
            fed_clients.append({"tag": client_tags[p], "meta": meta[p], "prototypes": clients[p]})
            labels_all.extend(meta[p]["labels"])

        federated_client, stats = federated_learning.federate(fed_clients, THRESHOLD_GOODNESS, THRESHOLD_SIMILARITY, device, do_resolve_conflicts=RESOLVE_CONFLICTS)
        labels_unique = list(set(labels_all))
        labels_unique.sort()

        # get all labels the federated client should have seen at some point
        meta_base = fed_clients[0]["meta"]

        prototypes_tag = f"fed;{i};{num_participants};{labels_unique}"
        common.save_metadata(prototypes_tag, meta_base["goodness"], meta_base["tau"], meta_base["m"], meta_base["num_prototypes"], labels_unique, [client_tags[p] for p in participant_indices], meta_base["dataset_tag"], meta_base["feature_extractor_tag"], True, stats["num_merges"], stats["num_conflicts_resolved"])
        prototype_ops.save_prototypes(prototypes_tag, federated_client, True)

def train_prototypes(param):
    DATASET_TAG = param["DATASET_TAG"]
    FEATURE_EXTRACTOR_TAG = param["FEATURE_EXTRACTOR_TAG"]
    LABELDNESS_STEPS = param["LABELDNESS_STEPS"]
    NUM_DATASET_VARIATIONS = param["NUM_DATASET_VARIATIONS"]
    BATCH_SIZE = param["BATCH_SIZE"]
    M = param["M"]
    TAU = param["TAU"]
    INITIAL_GOODNESS = param["INITIAL_GOODNESS"]
    NUM_PROTOTYPES = param["NUM_PROTOTYPES"]
    MAX_CLASSES_COEFF = param["MAX_CLASSES_COEFF"]

    device = common.get_device()
    fe = feature_extractor.FeatureExtractor(feature_extractor.map_tag_to_net(FEATURE_EXTRACTOR_TAG), FEATURE_EXTRACTOR_TAG, device)
    num_classes = common.map_tag_to_num_classes(DATASET_TAG)

    # get subset of dataset
    for v in range(NUM_DATASET_VARIATIONS):
        data_loader = common.load_dataset(BATCH_SIZE, DATASET_TAG, num_classes, True, MAX_CLASSES_COEFF, True, [], True)
        
        for labeldness in LABELDNESS_STEPS:
            prototypes = [prototype.Prototype(fe.dim_featurespace, i, INITIAL_GOODNESS) for i in range(1, NUM_PROTOTYPES+1)]
            prototypes = prototype_learning.continual_learning(M, TAU, prototypes, data_loader, labeldness, fe)
            trained_labels = data_loader.dataset.dataset.targets[data_loader.dataset.indices].unique().tolist()
            
            prototypes_tag = f"{DATASET_TAG};{FEATURE_EXTRACTOR_TAG};{v};{labeldness};{trained_labels}"
            prototype_ops.save_prototypes(prototypes_tag, prototypes, True)
            common.save_metadata(prototypes_tag, INITIAL_GOODNESS, TAU, M, NUM_PROTOTYPES, trained_labels, [], DATASET_TAG, FEATURE_EXTRACTOR_TAG, True) #TODO stack training labels when retraining
            print(f"saved prototypes and metadata with tag: {prototypes_tag}")

if __name__ == "__main__":
    main()