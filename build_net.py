from torch import nn
import torchvision.models as models
import models as customized_models

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
                                 if not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names


def make_model(args):
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained = True,progress=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, args.num_classes)
    )
    print("dropout id 0.5")
    return model
def make_model_by_name(args,num_classes):
    print("=> creating model '{}'".format(args))
    model = models.__dict__[args](progress=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(2048, num_classes)
    )
    return model

if __name__=='__main__':
    all_model = sorted(name for name in models.__dict__ if not name.startswith("__"))
    print(all_model)
    # print(model_names)
    # print(customized_models_names)