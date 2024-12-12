from prototypes import prototype 
import pathlib
import torch
import os

def allocate_new_protoype(prototypes, feature_vec, label=None):
    for prototype in prototypes:
        if (not prototype.allocated):
            prototype.allocate(feature_vec, label)
            return True
    return False

def update_m_closest_prototypes(m, similarities, allocated_prototypes, label, output, prototypes):
    assert label is not None # method will only be called, if label is known
    positive_match = False
    # sort the list by similarity and get permutation list
    combined = [(similarities[i].item(), allocated_prototypes[i]) for i in range(len(allocated_prototypes))]
    combined.sort(key=lambda tup: tup[0], reverse=True)
    # skip the winner prototype
    # print(f"combined (sim, label): {[(combine[0], combine[1].label) for combine in combined]}")
    for i in range(1, min(m, len(allocated_prototypes))):
        if (not combined[i][1].allocated):
            print("next m proto not allocated")
            continue
        psi = get_psi(combined[i][1].label, label)
        combined[i][1].update(output, psi)

        if psi == 1:
            positive_match = True
    
    if not positive_match:
        return allocate_new_protoype(prototypes, output, label)
    else:
        return True
    
def get_psi(prototype_label, prototype_prediction):
    if (prototype_label == prototype_prediction):
        return 1
    else:
        return -1
    
def save_prototypes(tag, prototypes, automatic=False, automatic_dir="auto"):
    base_dir = pathlib.Path(__file__).parent.resolve()
    # serialize
    prototypes_list = []
    for prototype in prototypes:
        prototypes_list.append(prototype.as_dict())
    #print(prototypes_list)
    #with open(f"{base_dir}/data/{tag}.proto.save", "w+") as f:
    dir_full = f"{base_dir}/../data/{automatic_dir if automatic else 'manual'}"
    if not os.path.isdir(dir_full):
        os.mkdir(dir_full)

    if(os.path.exists(f"{dir_full}/{tag}.proto.save")):
        print("Did not save prototypes, path already exists.")
        raise FileExistsError 
        # return False
    else:
        torch.save(prototypes_list, f"{dir_full}/{tag}.proto.save")
        #json.dump(prototypes_list, f)
        return True

def load_prototypes(tag, device, automatic=False, automatic_dir="auto"):
    base_dir = pathlib.Path(__file__).parent.resolve()
    prototypes_list = torch.load(f"{base_dir}/../data/{automatic_dir if automatic else 'manual'}/{tag}.proto.save", device)

    prototypes = []
    for elem in prototypes_list:
        proto = prototype.Prototype(0, -1, 0)
        proto.load_from_dict(elem)
        prototypes.append(proto)

    return prototypes