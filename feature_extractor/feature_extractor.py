import argparse
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from feature_extractor.net100 import Net100
from feature_extractor.net10 import Net10
from feature_extractor.net20 import Net20

class FeatureExtractor():
    def __init__(self, net, tag, device) -> None:
        base_dir = pathlib.Path(__file__).parent.resolve()
        self.device = device

        self.dim_featurespace = net.OUTPUT_DIM
        self.model = net.to(device)
        state_dict = torch.load(f"{base_dir}/mnist/{tag}.pt", weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def visualize_features(self, test_loader, writer, device):
        # take one batch of test data, visualize output as image in tensorboard
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        input_grid = torchvision.utils.make_grid(images, pad_value=0.4)
        writer.add_image('input batch', input_grid)

        self.model.eval()
        images, labels = images.to(device), labels.to(device)
        output = self.model(images)
        output_grid = torchvision.utils.make_grid(output.reshape((-1, 1, 10, 10)), pad_value=0.4)
        writer.add_image('output batch', output_grid)

        padded_feature = torch.nn.functional.pad(output.reshape((-1, 1, 10, 10)), (9, 9, 9, 9), value=0.4)
        concat = torch.cat((images, padded_feature), 0)
        overview_grid = torchvision.utils.make_grid(concat, pad_value=0.4, nrow=len(images))
        writer.add_image('overview', overview_grid)

    def extract_features_dataset(self, data_loader, name_str, visualize=False):

        base_dir = pathlib.Path(__file__).parent.resolve()
        device = self.device
        # accumulate model outputs
        inputs = []
        outputs = []
        labels = []
        for data, target in data_loader:
                    data, target = data.to(device), target.to(device)
                    output = self.model(data)
                    inputs.append(data)
                    outputs.append(output.reshape((output.size()[1]))) # reshape into 1-dim vector
                    labels.append(target)

        if visualize:
            # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
            writer = SummaryWriter(f'{base_dir}/runs/{name_str}')
            self.visualize_features(data_loader, writer, device)
            writer.close()
            
        return inputs, outputs, labels

    def extract_features(self, data):
        data = data.to(self.device)
        with torch.no_grad():
            output = self.model(data)
            output = output.reshape(output.size()[1]) # make output 1-dim
            return output
        
def map_tag_to_net(tag):
     if tag == "mnist_10dim":
          return Net10()
     elif tag == "mnist_20dim":
          return Net20()
     else:
          print("mapping from tag to net not defined")
          raise NotImplementedError

def main():
    base_dir = pathlib.Path(__file__).parent.resolve()
    use_cuda = torch.cuda.is_available()
    device = None

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 4, 'shuffle': True}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST(f'{base_dir}/data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST(f'{base_dir}/data', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    writer = SummaryWriter(f'{base_dir}/runs')
    model = Net100().to(device)
    fe = FeatureExtractor(Net100(), "mnist_full_10epo")
    fe.visualize_features(model, test_loader, writer, device)
    writer.close()



if __name__ == '__main__':
    #main()
    pass