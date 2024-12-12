import torch

class Prototype:
    def __init__(self, dim, id, init_goodness) -> None:
        assert id != 0 #+-0 are same
        self.center = torch.zeros(dim)
        self.goodness = init_goodness
        self.learning_rate = 1.0
        self.label = -1 * id
        self.allocated = False

    def load_from_dict(self, dict):
        self.center = dict["center"]
        self.goodness = dict["goodness"]
        self.learning_rate = 1 / dict["goodness"]
        self.label = dict["label"]
        self.allocated = dict["allocated"]

    def allocate(self, feature_vec, label=None):
        self.center = feature_vec
        self.goodness += 1
        self.learning_rate = 1 / self.goodness
        self.center = torch.nn.functional.normalize(self.center, dim=0)

        if label is not None:
            self.label = label

        self.allocated = True
        print(f"allocated new prototype to label {self.label}")

    def update(self, feature_vec, psi):
        feature_vec = torch.nn.functional.normalize(feature_vec, dim=0) #TODO is this right?
        self.center += psi * self.learning_rate * feature_vec
        self.goodness = max(1, self.goodness + psi) #goodness at least 1
        self.learning_rate = 1 / self.goodness
        self.center = torch.nn.functional.normalize(self.center, dim=0)
    
    def assign_label(self, label):
        if self.label < 0:
            self.label = label

    def as_dict(self):
        dict = {
            "center": self.center,
            "goodness": self.goodness,
            "label": self.label,
            "allocated": self.allocated,
        }
        return dict