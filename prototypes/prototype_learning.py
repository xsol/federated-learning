import torch
from torch.utils.tensorboard import SummaryWriter

from prototypes import prototype
from prototypes import prototype_ops
from common import common
import random

def continual_learning(m, tau, prototypes, data_loader, labeldness, feature_extractor):
    device = common.get_device()

    # continual learning
    assert (data_loader.batch_size == 1 and tau > 0)
    correct_predictions = num_predictions = 0

    for data, label in data_loader:
        if num_predictions % 5000 == 0:
            print(f"{num_predictions}/{len(data_loader)}")
        num_predictions +=1

        # extract features
        output = feature_extractor.extract_features(data)
        
        # labeldness
        if random.uniform(0, 1) < labeldness:
            label = label.item() 
        else:
            label = None

        # calculate similarity to each prototype
        output_norm = torch.nn.functional.normalize(output, dim=0)
        allocated_prototypes = [prototype for prototype in prototypes if prototype.allocated]
        similarities = [torch.dot(output_norm, prototype.center.to(device)) for prototype in allocated_prototypes]
        
        # determine highest similarity (winner) protype
        highest_similarity = 0
        if len(similarities) != 0:
            highest_similarity = torch.max(torch.tensor(similarities))
            # print(f"highest similarity for sample of label {label} is {highest_similarity.item()}")

        # novelty detection
        if (highest_similarity < tau):
            print("novelty detected, allocating new prototype.")
            if not prototype_ops.allocate_new_protoype(prototypes, output, label):
                print("No more prototypes left to allocate. Aborting")
                raise OverflowError
                break

        else:
            # find index of highest similarity (winner) prototype
            star = common.max_index(similarities)
            winner = allocated_prototypes[star]
            # print(f"most similar protoype is of label {winner.label}")
            psi = 0.5
            if label is not None:
                # supervised update
                winner.assign_label(label)
                psi = prototype_ops.get_psi(winner.label, label)
                winner.update(output, psi)
            else:
                # unsupervised update
                winner.update(output, psi)
            # print(f"updated most similar prototype, psi={psi}, supervised={label is not None}")

            if (psi == -1):
                if (not prototype_ops.update_m_closest_prototypes(m, similarities, allocated_prototypes, label, output, prototypes)):
                    print("No more prototypes left to allocate. Aborting")
                    #break
                    continue
            else:
                correct_predictions += 1

    print(f"protoype (label, goodness): {[(proto.label, proto.goodness) for proto in prototypes]}")
    print(f"{torch.tensor([prototype.allocated for prototype in prototypes]).sum().item()} prototypes allocated")
    print(f"total accuracy: {correct_predictions/num_predictions}")

    return prototypes

def inference(data_loader, settings, prototypes, fe, device):
    # classification (no learning)
    prediction_stats = [{"label":label, "total":0, "correct":0, "incorrect":0} for label in settings["labels"]] # for every label, keep tupel of (correct, incorrect) prediction count
    for data, label in data_loader:
        num_predictions = common.count_predictions(prediction_stats, filter_label=None)["total"]
        if num_predictions % 1000 == 0:
            print(f"{num_predictions}/{len(data_loader)}")

        # extract features
        output = fe.extract_features(data)
        label = label.item()

        # calculate similarity to each prototype
        output_norm = torch.nn.functional.normalize(output, dim=0)
        allocated_prototypes = [prototype for prototype in prototypes if prototype.allocated]
        similarities = [torch.dot(output_norm, prototype.center.to(device)) for prototype in allocated_prototypes]
        
        if len(similarities) == 0:
            print("no allocated prototypes exist.")
            break

        star = common.max_index(similarities)
        winner = allocated_prototypes[star]
        prediction = winner.label

        common.add_prediction(prediction_stats, prediction, label)

    correct_predictions = common.count_predictions(prediction_stats, filter_label=None)["correct"]
    num_predictions = common.count_predictions(prediction_stats, filter_label=None)["total"]
    print(f"total accuracy: {correct_predictions/num_predictions}")

    return prediction_stats
