import torch
from torchvision import datasets, transforms
import pathlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os, io
import glob
import random
from PIL import Image

def calculate_loss(stats, best_possible_stats):
    counts_stats = count_predictions(stats)
    counts_bp_stats = count_predictions(best_possible_stats)
    loss = counts_bp_stats['correct']/counts_bp_stats['total'] - counts_stats['correct']/counts_stats['total']
    return loss

def best_possible_stats(stats_list):
    all_labels = []
    for stats in stats_list:
        for stat in stats:
            if stat["label"] not in all_labels:
                all_labels.append(stat["label"])

    best_stats = [{"label":label, "total":100, "correct":1, "incorrect":0, "default_value": True} for label in all_labels]

    for stats in stats_list:
        for stat in stats:
            for best_stat in best_stats:
                # find match
                if best_stat["label"] == stat["label"]:
                    if best_stat["default_value"] or best_stat["correct"]/best_stat["total"] < stat["correct"]/stat["total"]:
                        # stat is better, replace
                        best_stat["total"] = stat["total"]
                        best_stat["correct"] = stat["correct"]
                        best_stat["incorrect"] = stat["incorrect"]
                        best_stat["default_value"] = False

    return best_stats

def get_device():
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def get_tags(automatic_dir="auto"):
    base_dir = pathlib.Path(__file__).parent.resolve()
    files = glob.glob(f"{base_dir}/../data/{automatic_dir}/*.proto.save")
    tags = [file.split("/")[-1][0:-11] for file in files]
    return tags   

def load_dataset(batch_size, dataset_tag, num_classes, uniform_dist_samples, max_classes_coeff, uniform_dist_labels, classes=[], train=True):
    if dataset_tag == "mnist":
        if classes == []:
            classes = [x for x in range(num_classes)]

        if uniform_dist_labels:
            random.shuffle(classes)
            num_labels = round(random.uniform(1-0.5, round(num_classes*max_classes_coeff)+0.499))
            labels = classes[0:num_labels]
        else:
            labels = classes

        return load_mnist_by_labels(batch_size, labels, train, uniform_dist_samples)

def map_tag_to_num_classes(tag):
    if tag == "mnist":
        return 10
    else:
        print("mapping from tag to net not defined")
        raise NotImplementedError

def load_mnist_by_labels(batch_size, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], train=True, uniform_dist_samples=False):
    base_dir = pathlib.Path(__file__).parent.resolve()
    args = {'batch_size': batch_size, 'shuffle': True}
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST(f'{base_dir}/../feature_extractor/data', train=train, transform=transform)
    indices_filter_labels = [idx for idx, target in enumerate(dataset.targets) if target in labels]
    dataset_filtered_labels = torch.utils.data.Subset(dataset, indices_filter_labels)

    dataset_filtered = dataset_filtered_labels
    if uniform_dist_samples:
        percentages_filter_samples = {}
        for l in labels:
            percentages_filter_samples.update({str(l): random.uniform(0.5, 1)})
            
        indices_filter_labels_samples = [idx for idx, target in enumerate(dataset.targets) if idx in dataset_filtered_labels.indices and roll_dice(percentages_filter_samples[str(target.item())])]
        dataset_filtered_labels_samples = torch.utils.data.Subset(dataset, indices_filter_labels_samples)
        dataset_filtered = dataset_filtered_labels_samples

    data_loader = torch.utils.data.DataLoader(dataset_filtered, **args)

    print(f"dataset loaded for labels: {str(labels)} with {len(data_loader.dataset)} samples. Train={train}")
    return data_loader

def roll_dice(t):
    num = random.uniform(0, 1)

    if num < t:
        return True
    else:
        return False

def check_dataset_compatability(meta, participant_indices):
    comp_dataset = meta[participant_indices[0]]["dataset_tag"]
    comp_fe = meta[participant_indices[0]]["feature_extractor_tag"]
    for p in participant_indices:
        if meta[p]["dataset_tag"] != comp_dataset:
            return False
        if meta[p]["feature_extractor_tag"] != comp_fe:
            return False
    return True

def max_index(num_array):
    max_val = None
    max_index = None
    for i in range(len(num_array)):
        elem = num_array[i]
        if max_val is None or max_val < elem:
            max_val = elem
            max_index = i
    return max_index

def save_metadata(tag, goodness, tau, m, num_prototypes, labels, parents, dataset_tag, feature_extractor_tag, automatic=False, num_merges=0, num_conflicts_resolved=0, automatic_dir="auto"):
    base_dir = pathlib.Path(__file__).parent.resolve()
    d = {
        "goodness": goodness,
        "tau": tau,
        "m": m,
        "num_prototypes": num_prototypes,
        "parents": parents,
        "labels": labels,
        "dataset_tag": dataset_tag,
        "feature_extractor_tag": feature_extractor_tag,
        "num_merges": num_merges,
        "num_conflicts_resolved": num_conflicts_resolved
    }
    dir_full = f"{base_dir}/../data/{automatic_dir if automatic else 'manual'}"
    if(os.path.exists(f"{dir_full}/{tag}.meta.json")):
        print("Did not save meta, path already exists.")
    else:
        with open(f"{dir_full}/{tag}.meta.json", "w+") as f:
            json.dump(d, f)

def save_metadata_full(meta, tag, automatic=False, automatic_dir="auto"):
    base_dir = pathlib.Path(__file__).parent.resolve()
    dir_full = f"{base_dir}/../data/{automatic_dir if automatic else 'manual'}"


    with open(f"{dir_full}/{tag}.meta.json", "w") as f:
        json.dump(meta, f)


def load_metadata(tag, automatic=False, automatic_dir="auto"):
    base_dir = pathlib.Path(__file__).parent.resolve()
    dir_full = f"{base_dir}/../data/{automatic_dir if automatic else 'manual'}"
    with open(f"{dir_full}/{tag}.meta.json", "r+") as f:
        d = json.load(f)
        return d

def count_predictions(prediction_stats, filter_label=None):
    counts = {"total": 0, "correct": 0, "incorrect": 0}
    for stat in prediction_stats:
        if filter_label is None or filter_label == stat["label"]:
            counts["total"] += stat["total"]
            counts["correct"] += stat["correct"]
            counts["incorrect"] += stat["incorrect"]

    return counts

def add_prediction(prediction_stats, prediction, label):
    for stat in prediction_stats:
        if stat["label"] == label:
            stat["total"] += 1
            if prediction == label:
                stat["correct"] += 1
            else:
                stat["incorrect"] += 1
            break

def visualize_proto_and_out(writer, outputs, labels, prototypes):
    vectors = [prototype.center.to(get_device()) for prototype in prototypes]
    vectors.extend(outputs)
    vectors_tensor = torch.stack(vectors)

    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    output_meta = [classes[lab] for lab in labels]
    meta = [f"proto({prototype.label})" for prototype in prototypes]
    meta.extend(output_meta)

    writer.add_embedding(vectors_tensor, metadata=meta, tag="proto_output")


def visualize_proto(writer, prototypes, tag):
    vectors = [prototype.center.to(get_device()) for prototype in prototypes if prototype.allocated]
    vectors_tensor = torch.stack(vectors)
    meta = [f"proto({prototype.label}), g:{prototype.goodness}" for prototype in prototypes if prototype.allocated]

    writer.add_embedding(vectors_tensor, metadata=meta, tag=tag)

def visualize_loss_by_federation_participants(writer, loss_infos):
    
    participants = [loss_info["num_fed_participants"] for loss_info in loss_infos]
    min_part = min(participants)
    max_part = max(participants)
    spread = max_part - min_part
    num_containers = 12
    containers = []
    if spread < num_containers:
        containers = [(i, i+1) for i in range(min_part, max_part+1)]
    else:
        step = round(spread/num_containers)
        i = min_part + step
        prev_i = min_part
        while i < max_part:
            containers.append((prev_i, i+1))
            prev_i = i+1
            i+=step
        containers.append((prev_i, i+1))
        # print(containers)
    labels = []
    avgs = []

    fig, ax = plt.subplots(layout='constrained')

    for mi, ma in containers:
        losses = [loss_info["loss"] for loss_info in loss_infos if loss_info["num_fed_participants"] >= mi and loss_info["num_unlabeled_prototypes_parents"] < ma]
        losses_np = np.array(losses)
        losses_avg = np.average(losses_np)
        avgs.append(losses_avg)
        labels.append(f"{mi}\n-\n{ma-1}")
    
    bars = ax.bar(labels, [round(num, 3) for num in avgs])
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('average loss')
    ax.set_xlabel("number of participants for federating")
    ax.set_title(f"Loss per number of federation participants")

    writer.add_figure(f"Loss per federation participants", fig)
    fig.clf()
    
def visualize_loss_by_num_prototypes(writer, loss_infos):
    
    participants = [loss_info["num_prototypes"] for loss_info in loss_infos]
    min_part = min(participants)
    max_part = max(participants)
    spread = max_part - min_part
    num_containers = 12
    containers = []
    if spread < num_containers:
        containers = [(i, i+1) for i in range(min_part, max_part+1)]
    else:
        step = round(spread/num_containers)
        i = min_part + step
        prev_i = min_part
        while i < max_part:
            containers.append((prev_i, i+1))
            prev_i = i+1
            i+=step
        containers.append((prev_i, i+1))
        # print(containers)
    labels = []
    avgs = []

    fig, ax = plt.subplots(layout='constrained')

    for mi, ma in containers:
        losses = [loss_info["loss"] for loss_info in loss_infos if loss_info["num_prototypes"] >= mi and loss_info["num_prototypes"] < ma]
        losses_np = np.array(losses)
        losses_avg = np.average(losses_np)
        avgs.append(losses_avg)
        labels.append(f"{mi}\n-\n{ma-1}")
    
    bars = ax.bar(labels, [round(num, 3) for num in avgs])
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('average loss')
    ax.set_xlabel("number of prototypes after federating")
    ax.set_title(f"Loss per number of prototypes")

    writer.add_figure(f"Loss per number of prototypes", fig)
    fig.clf()

def visualize_loss_by_num_unlabeled_prototypes_parents(writer, loss_infos):
    
    participants = [loss_info["num_unlabeled_prototypes_parents"] for loss_info in loss_infos]
    min_part = min(participants)
    max_part = max(participants)
    spread = max_part - min_part
    num_containers = 12
    containers = []
    if spread < num_containers:
        containers = [(i, i+1) for i in range(min_part, max_part+1)]
    else:
        step = round(spread/num_containers)
        i = min_part + step
        prev_i = min_part
        while i < max_part:
            containers.append((prev_i, i+1))
            prev_i = i+1
            i+=step
        containers.append((prev_i, i+1))
        # print(containers)
    labels = []
    avgs = []

    fig, ax = plt.subplots(layout='constrained')

    for mi, ma in containers:
        losses = [loss_info["loss"] for loss_info in loss_infos if loss_info["num_unlabeled_prototypes_parents"] >= mi and loss_info["num_unlabeled_prototypes_parents"] < ma]
        losses_np = np.array(losses)
        losses_avg = np.average(losses_np)
        avgs.append(losses_avg)
        labels.append(f"{mi}\n-\n{ma-1}")
    
    bars = ax.bar(labels, [round(num, 3) for num in avgs])
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('average loss')
    ax.set_xlabel("number of unlabeled prototypes")
    ax.set_title(f"Loss per number of unlabeled prototypes among parents")

    writer.add_figure(f"Loss per number of unlabeled prototypes among parents", fig)
    fig.clf()

def visualize_loss_by_num_merges(writer, loss_infos):
    
    participants = [loss_info["num_merges"] for loss_info in loss_infos]
    min_part = min(participants)
    max_part = max(participants)
    spread = max_part - min_part
    num_containers = 12
    containers = []
    if spread < num_containers:
        containers = [(i, i+1) for i in range(min_part, max_part+1)]
    else:
        step = round(spread/num_containers)
        i = min_part + step
        prev_i = min_part
        while i < max_part:
            containers.append((prev_i, i+1))
            prev_i = i+1
            i+=step
        containers.append((prev_i, i+1))
        # print(containers)
    labels = []
    avgs = []

    fig, ax = plt.subplots(layout='constrained')

    for mi, ma in containers:
        losses = [loss_info["loss"] for loss_info in loss_infos if loss_info["num_merges"] >= mi and loss_info["num_merges"] < ma]
        losses_np = np.array(losses)
        losses_avg = np.average(losses_np)
        avgs.append(losses_avg)
        labels.append(f"{mi}\n-\n{ma-1}")
    
    bars = ax.bar(labels, [round(num, 3) for num in avgs])
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('average loss')
    ax.set_xlabel("number of merges")
    ax.set_title(f"Loss per number of merges while federating")

    writer.add_figure(f"Loss per number of merges while federating", fig)
    fig.clf()

def visualize_loss_by_num_conflicts_resolved(writer, loss_infos):
    
    participants = [loss_info["num_conflicts_resolved"] for loss_info in loss_infos]
    min_part = min(participants)
    max_part = max(participants)
    spread = max_part - min_part
    num_containers = 12
    containers = []
    if spread < num_containers:
        containers = [(i, i+1) for i in range(min_part, max_part+1)]
    else:
        step = round(spread/num_containers)
        i = min_part + step
        prev_i = min_part
        while i < max_part:
            containers.append((prev_i, i+1))
            prev_i = i+1
            i+=step
        containers.append((prev_i, i+1))
        # print(containers)
    labels = []
    avgs = []

    fig, ax = plt.subplots(layout='constrained')

    for mi, ma in containers:
        losses = [loss_info["loss"] for loss_info in loss_infos if loss_info["num_conflicts_resolved"] >= mi and loss_info["num_unlabeled_prototypes_parents"] < ma]
        losses_np = np.array(losses)
        losses_avg = np.average(losses_np)
        avgs.append(losses_avg)
        labels.append(f"{mi}\n-\n{ma-1}")
    
    bars = ax.bar(labels, [round(num, 3) for num in avgs])
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('average loss')
    ax.set_xlabel("number of resolved conflicts")
    ax.set_title(f"Loss per number of resolved conflicts while federating")

    writer.add_figure(f"Loss per number of resolved conflicts while federating", fig)
    fig.clf()

def visualize_federation_participants_distribution(writer, loss_infos):
    
    participants = [loss_info["num_fed_participants"] for loss_info in loss_infos]
    min_part = min(participants)
    max_part = max(participants)
    labels = []
    nums = []

    fig, ax = plt.subplots(layout='constrained')

    for i in range(min_part, max_part+1, 1):
        num = participants.count(i)
        nums.append(num)
        labels.append(f"{i}")
    
    bars = ax.bar(labels, [round(num, 3) for num in nums])
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('number of clients federated from x participants')
    ax.set_xlabel("number of participants for federating")
    ax.set_title(f"Federation participants distribution")

    writer.add_figure(f"Federation participants distribution", fig)
    fig.clf()

def write_params(writer, params):

    fig, ax = plt.subplots()
    # hide axes
    ax.set(xlim=(0, 10), ylim=(0, 10))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    v = [str(params[param]) for param in params.keys()]
    k = [param for param in params.keys()]
    t = ""
    for i in range(len(v)):
        t += f"{k[i]}: {v[i]}\n"

    #ax.table(cellText=t, rowLabels=r, loc='top')
    plt.text(5, 10, t, fontsize=10, style='oblique', ha='center',
         va='top', wrap=True)

    fig.tight_layout()
    writer.add_figure(f"Parameters", fig)
    fig.clf()

def visualize_loss_distribution(writer, loss_infos):
    
    losses = [loss_info["loss"] for loss_info in loss_infos]
    losses_np = np.array(losses)
    min_loss = min(losses)
    max_loss = max(losses)
    containers = 10
    labels = []
    nums = []
    steps = (max_loss-min_loss)/containers
    eps = 0.0001

    fig, ax = plt.subplots(layout='constrained')

    for i in range(containers):
        thresh = min_loss + i*steps
        nums.append(((thresh <= losses_np) & (losses_np <= (thresh+steps+eps))).sum())
        labels.append(f"{round(thresh, 3)}\n - \n{round(thresh+steps, 3)}")
    
    bars = ax.bar(labels, nums)
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('number of clients')
    ax.set_xlabel("average loss per client")
    ax.set_title(f"Loss distribution of federated clients \n(avg: {round(np.average(losses_np), 4)})")

    writer.add_figure(f"Loss distribution", fig)
    fig.clf()

def visualize_goodness_distribution(writer, goodnesses, tag):
    
    losses = goodnesses
    losses_np = np.array(losses)
    min_loss = min(losses)
    max_loss = max(losses)
    containers = 50
    labels = []
    nums = []
    steps = (max_loss-min_loss)/containers
    eps = 0.0001

    fig, ax = plt.subplots(layout='constrained')

    for i in range(containers):
        thresh = min_loss + i*steps
        nums.append(((thresh <= losses_np) & (losses_np <= (thresh+steps+eps))).sum())
        labels.append(f"{int(thresh)}\n - \n{int(thresh+steps)}")
    
    bars = ax.bar(labels, nums)
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('number of prototypes')
    ax.set_xlabel("goodness")
    ax.set_title(f"Goodness distribution of {tag} \n(avg: {round(np.average(losses_np), 4)})")

    writer.add_figure(f"Goodness distribution of {tag}", fig, close=True)
    fig.clear()
    ax.clear()
    del fig, ax


def visualize_num_prototypes_parents_distribution(writer, loss_infos):

    losses = []
    for client in loss_infos:
        losses.extend(client["num_prototypes_parents"])
    
    num_parent_prototypes = losses
    num_parent_prototypes_np = np.array(num_parent_prototypes)
    min_part = min(num_parent_prototypes)
    max_part = max(num_parent_prototypes)
    spread = max_part - min_part
    num_containers = 12
    containers = []
    if spread < num_containers:
        containers = [(i, i+1) for i in range(min_part, max_part+1)]
    else:
        step = round(spread/num_containers)
        i = min_part + step
        prev_i = min_part
        while i < max_part:
            containers.append((prev_i, i+1))
            prev_i = i+1
            i+=step
        containers.append((prev_i, i+1))
        # print(containers)
    labels = []
    nums = []

    fig, ax = plt.subplots(layout='constrained')

    for mi, ma in containers:
        nums.append(((mi <= num_parent_prototypes_np) & (num_parent_prototypes_np < (ma))).sum())
        labels.append(f"{mi}\n-\n{ma-1}")
    
    bars = ax.bar(labels, nums)
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel("number of parent clients")
    ax.set_xlabel("number of prototypes before federating")
    ax.set_title(f"Prototype-count distribution of parent clients \n(avg: {round(np.average(num_parent_prototypes_np), 4)})")

    writer.add_figure(f"Prototype-count distribution parents", fig)
    fig.clf()

def visualize_num_prototypes_distribution(writer, loss_infos):
    
    losses = [loss_info["num_prototypes"] for loss_info in loss_infos]
    num_parent_prototypes = losses
    num_parent_prototypes_np = np.array(num_parent_prototypes)
    min_part = min(num_parent_prototypes)
    max_part = max(num_parent_prototypes)
    spread = max_part - min_part
    num_containers = 12
    containers = []
    if spread < num_containers:
        containers = [(i, i+1) for i in range(min_part, max_part+1)]
    else:
        step = round(spread/num_containers)
        i = min_part + step
        prev_i = min_part
        while i < max_part:
            containers.append((prev_i, i+1))
            prev_i = i+1
            i+=step
        containers.append((prev_i, i+1))
        # print(containers)
    labels = []
    nums = []

    fig, ax = plt.subplots(layout='constrained')

    for mi, ma in containers:
        nums.append(((mi <= num_parent_prototypes_np) & (num_parent_prototypes_np < (ma))).sum())
        labels.append(f"{mi}\n-\n{ma-1}")
    
    bars = ax.bar(labels, nums)
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('number of federated clients')
    ax.set_xlabel("number of prototypes after federating")
    ax.set_title(f"Prototype-count distribution of federated clients \n(avg: {round(np.average(num_parent_prototypes_np), 4)})")

    writer.add_figure(f"Prototype-count distribution", fig)
    fig.clf()

def visualize_num_unlabeled_prototypes_parents_distribution(writer, loss_infos):
    
    losses = [loss_info["num_unlabeled_prototypes_parents"] for loss_info in loss_infos]
    num_parent_prototypes = losses
    num_parent_prototypes_np = np.array(num_parent_prototypes)
    min_part = min(num_parent_prototypes)
    max_part = max(num_parent_prototypes)
    spread = max_part - min_part
    num_containers = 12
    containers = []
    if spread < num_containers:
        containers = [(i, i+1) for i in range(min_part, max_part+1)]
    else:
        step = round(spread/num_containers)
        i = min_part + step
        prev_i = min_part
        while i < max_part:
            containers.append((prev_i, i+1))
            prev_i = i+1
            i+=step
        containers.append((prev_i, i+1))
        # print(containers)
    labels = []
    nums = []

    fig, ax = plt.subplots(layout='constrained')

    for mi, ma in containers:
        nums.append(((mi <= num_parent_prototypes_np) & (num_parent_prototypes_np < (ma))).sum())
        labels.append(f"{mi}\n-\n{ma-1}")
    
    bars = ax.bar(labels, nums)
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('number of federated clients')
    ax.set_xlabel("total number of unlabeled prototypes from parents")
    ax.set_title(f"Distribution of number of unlabeled parent prototypes\n(avg: {round(np.average(num_parent_prototypes_np), 4)})")

    writer.add_figure(f"Unlabeled parent prototypes distribution", fig)
    fig.clf()

def visualize_num_conflicts_resolved_distribution(writer, loss_infos):
    
    losses = [loss_info["num_conflicts_resolved"] for loss_info in loss_infos]
    num_parent_prototypes = losses
    num_parent_prototypes_np = np.array(num_parent_prototypes)
    min_part = min(num_parent_prototypes)
    max_part = max(num_parent_prototypes)
    spread = max_part - min_part
    num_containers = 12
    containers = []
    if spread < num_containers:
        containers = [(i, i+1) for i in range(min_part, max_part+1)]
    else:
        step = round(spread/num_containers)
        i = min_part + step
        prev_i = min_part
        while i < max_part:
            containers.append((prev_i, i+1))
            prev_i = i+1
            i+=step
        containers.append((prev_i, i+1))
        # print(containers)
    labels = []
    nums = []

    fig, ax = plt.subplots(layout='constrained')

    for mi, ma in containers:
        nums.append(((mi <= num_parent_prototypes_np) & (num_parent_prototypes_np < (ma))).sum())
        labels.append(f"{mi}\n-\n{ma-1}")
    
    
    bars = ax.bar(labels, nums)
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('number of federated clients')
    ax.set_xlabel("number of resolved conflicts in federating")
    ax.set_title(f"Distribution of number of resolved conflicts in federating\n(avg: {round(np.average(num_parent_prototypes_np), 4)})")

    writer.add_figure(f"Resolved conflicts distribution", fig)
    fig.clf()

def visualize_num_merges_distribution(writer, loss_infos):
    
    losses = [loss_info["num_merges"] for loss_info in loss_infos]
    num_parent_prototypes = losses
    num_parent_prototypes_np = np.array(num_parent_prototypes)
    min_part = min(num_parent_prototypes)
    max_part = max(num_parent_prototypes)
    spread = max_part - min_part
    num_containers = 12
    containers = []
    if spread < num_containers:
        containers = [(i, i+1) for i in range(min_part, max_part+1)]
    else:
        step = round(spread/num_containers)
        i = min_part + step
        prev_i = min_part
        while i < max_part:
            containers.append((prev_i, i+1))
            prev_i = i+1
            i+=step
        containers.append((prev_i, i+1))
        # print(containers)
    labels = []
    nums = []

    fig, ax = plt.subplots(layout='constrained')

    for mi, ma in containers:
        nums.append(((mi <= num_parent_prototypes_np) & (num_parent_prototypes_np < (ma))).sum())
        labels.append(f"{mi}\n-\n{ma-1}")
    
    bars = ax.bar(labels, nums)
    ax.bar_label(bars, padding=3, fontsize=6)
    ax.set_ylabel('number of federated clients')
    ax.set_xlabel("total number of merges while federating")
    ax.set_title(f"Distribution of number of merges while federating\n(avg: {round(np.average(num_parent_prototypes_np), 4)})")

    writer.add_figure(f"Merge-count distribution", fig)
    fig.clf()

def visualize_accuracy(writer, prediction_stats, client_name="client"):

    classes = [f'class {stat["label"]} \n acc=\n{round(stat["correct"]/stat["total"], 3)}' for stat in prediction_stats]
    x = np.arange(len(classes))  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for bar in ["total", "correct", "incorrect"]:
        offset = width * multiplier
        rects = ax.bar(x + offset, [stat[bar] for stat in prediction_stats], width, label=bar)
        ax.bar_label(rects, padding=3, fontsize=6)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    counts = count_predictions(prediction_stats)
    ax.set_ylabel('number of predictions')
    ax.set_title(f"Test Accuracy by class (total accuracy={round(counts['correct']/counts['total'], 3)})")
    ax.set_xticks(x + width, classes)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1500)

    writer.add_figure(f"accuracy by class - {client_name}", fig, close=True)
    ax.clear()
    fig.clf()