import torch


def federate(clients, t_goodness, t_similarity, device, do_resolve_conflicts=False):
    stats = {"num_merges": 0, "num_conflicts_resolved": 0}
    accumulated_prototypes = accumulate_prototypes(clients)
    filtered_prototypes = filter_prototypes(accumulated_prototypes, t_goodness)

    federated_prototypes = filtered_prototypes
    while True:

        if len(federated_prototypes) <= 1:
            break
        
        # calculate similarity matrix
        S = calculate_similarity_matrix(federated_prototypes, device)
        # federated_learning.print_matrix(S)
        i, j = find_most_similar_compatible(S, federated_prototypes)
        if i is None or j is None:
            #print("no more prototypes compatible for merge")
            break

        if S[i][j] > t_similarity:
            # try merge
            federated_prototypes = merge_prototypes(federated_prototypes, i, j)
            stats["num_merges"]+=1
        else:
            break
    
    if do_resolve_conflicts:
        federated_prototypes, num_resolved_conflicts = resolve_conflicts(federated_prototypes, t_similarity, device)
        stats["num_conflicts_resolved"] = num_resolved_conflicts

    return federated_prototypes, stats

def resolve_conflicts(prototypes, t_similarity, device):
    num_resolved_conflicts = 0
    while True:
        # calculate similarity matrix
        S = calculate_similarity_matrix(prototypes, device)
        # federated_learning.print_matrix(S)
        i, j = find_most_similar(S)

        if S[i][j] is None:
            # if S is 1x1 matrix
            break
        elif S[i][j] > t_similarity:
            # conflicting prototypes, very similar but
            # should have conflicting labels
            assert prototypes[i].label != prototypes[j].label
            prototypes = pick_one_prototype(prototypes, i, j)
            num_resolved_conflicts+=1

        else:
            break 

    return prototypes, num_resolved_conflicts  


def accumulate_prototypes(clients):
    prototypes = []
    for client in clients:
        prototypes.extend(client["prototypes"])
    return prototypes

def filter_prototypes(prototypes, thresh_goodness=None):
    filtered = []
    for prototype in prototypes:
        if prototype.allocated and (thresh_goodness is None or prototype.goodness > thresh_goodness):
            filtered.append(prototype)
    return filtered

def calculate_similarity_matrix(prototypes, device):
    num_prototypes = len(prototypes)
    # S is upper triangle matrix because similarity is commutative and similarity to oneself is 1
    S = []
    for i in range(num_prototypes):
        s = [None] * num_prototypes
        S.append(s)

    for i in range(num_prototypes):
        for j in range(i+1, num_prototypes):
            S[i][j] = torch.dot(prototypes[i].center.to(device), prototypes[j].center.to(device)).item()
    #print(S)
    return S

def print_matrix(m):
    for row in m:
        print(row)

def find_most_similar_compatible(s, federated_prototypes):

    while True:
        i, j = find_most_similar(s)
        if s[i][j] == -1.0:
            # no compatible prototypes
            return None, None

        if prototypes_compatible(federated_prototypes[i], federated_prototypes[j]):
            return i, j
        else:
            # ignore prototype pair
            s[i][j] = -1.0

def prototypes_compatible(p1, p2):
    # check if 2 prototypes are mergeable by some rules
    if p1.label < 0 or p2.label < 0 or p1.label == p2.label:
        return True
    return False

def find_most_similar(s):
    num_prototypes = len(s)
    max_value = None
    max_index = (0, 0)
    for i in range(num_prototypes):
        for j in range(i+1, num_prototypes):
            value = s[i][j]
            if max_value is None or max_value < value:
                max_value = value
                max_index = (i, j)
    
    return max_index

def merge_prototypes(prototypes, i, j):
    merged_prototype = {
        "center": torch.mean(torch.stack([prototypes[i].center, prototypes[j].center]), 0),
        "goodness": (prototypes[i].goodness + prototypes[j].goodness)/2,
        "label": max(prototypes[i].label, prototypes[j].label),
        "allocated": True,
    }
    # override prototype i
    prototypes[i].load_from_dict(merged_prototype)

    # remove prototypes[j]
    del prototypes[j]

    return prototypes 

def pick_one_prototype(prototypes, i, j):
    if prototypes[i].goodness > prototypes[j].goodness:
        del prototypes[j]
    elif prototypes[j].goodness > prototypes[i].goodness:
        del prototypes[i]
    else:
        print("both conflicting prototypes have the same goodness.")
        del prototypes[i]

    return prototypes 