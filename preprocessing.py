# import pyarrow as pa
from datasets import Dataset # arrow dataset class
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms, models
from transformers import ConvNextV2Config, ConvNextV2Model, ConvNextImageProcessor
from PIL.Image import Image
import pandas as pd

def select_classes(dataset: Dataset,
                 which_classes: list[int],
                ):
    """accepts a Dataset object and returns a subset with only a fraction
    of the classes present.
    if which_classes is as int, simply takes classes from the front.
    if which_classes is a list of ints, use the class indices corresponding to those ints."""

    if isinstance(which_classes, int):
        which_classes = list(range(which_classes))
    else:
        assert isinstance(which_classes[0], int)

    original_num_classes = dataset.features['label'].num_classes

    class_labels = np.asarray(dataset.select_columns('label').data).squeeze()
    chosen_class_idxs = [l for l,label in enumerate(class_labels) if label in which_classes]

    data_subset = dataset.select(chosen_class_idxs)

    # re-map the labels so they are continuous:
    data_subset.features['label'] = data_subset.features['label']
    
    return data_subset

def get_class_weights(dataset: Dataset):
    """accepts a Dataset object, and returns a probability distribution
    for a weighted loss function with respect to the dataset's class imbalance"""
    labels = dataset['label']
    counts = pd.Series(labels).value_counts()

    # for some heterogeneous subsets, it's possible that not all the classes are represented
    # so we have to fill in the missing ones:
    actual_num_classes = len(dataset.features['label'].names)
    
    class_weights = [counts[c] / len(labels) if c in counts else 0 for c in range(actual_num_classes)]
    return torch.tensor(class_weights, dtype=torch.float)

def take_subset(dataset: Dataset,
                fraction: float, # percentage of data to take randomly, from 0-1
                return_rest: bool=False, # whether or not to return the remaining subset in a tuple
               ):

    if fraction == 1:
        # no need to take subset
        return dataset
    
    total_rows = len(dataset)
    num_to_take = int(total_rows * fraction)
    print(f'taking {fraction:.0%} ({num_to_take}/{total_rows}) of {dataset.info.dataset_name}({dataset.split._name})')
    random_idxs = np.random.choice(total_rows, num_to_take, replace=False)
    subset = dataset.select(random_idxs)

    if not return_rest:
        return subset
    else:
        rest_idxs = np.asarray([i for i in range(total_rows) if i not in random_idxs])
        rest_subset = dataset.select(rest_idxs)
        return (subset, rest_subset)


def partition_data(dataset: Dataset,
                   num_partitions: int,
                       # the number of client machines which will 
                       #     contribute federated learning updates.
                   heterogeneity: str='strong',
                       # one of 'none', 'weak', or 'strong' (see below).
                  ):
    """accepts a Dataset object, and divides it into num_partitions
    subsets, each representing a fraction of the data held privately 
    by some client.
    
    arg 'heterogeneity' is one of: 'none', 'weak', or 'strong':
        - 'none' is perfect homogeneity: all partitions 
            contain all class labels equally.
        - 'weak' randomly skews the distribution of class 
            labels across partitions.
        - 'strong' has random skew and additionally creates 
            partitions that lack some class labels entirely.
    
    returns (num_partitions) new Dataset objects in a list."""
    
    num_classes = dataset.features['label'].num_classes
    total_samples = dataset.num_rows
    
    # handle strong heterogeneity, and work out
    # which classes are dropped in which partitions:
    if heterogeneity == 'strong':
        
        if num_classes >= num_partitions:
            # if at least as many classes as partitions,
            # drop 1/K of classes from each of the K partitions
            fractions_to_drop = np.linspace(0, 1, num_partitions+1)
            idxs = (fractions_to_drop * num_classes).astype(int)
            
            # randomise which to drop so the ordering isn't sequential:
            shuffled_classes = list(np.random.permutation(range(num_classes)))
            classes_to_drop = [shuffled_classes[idxs[i] : idxs[i+1]]
                               for i in range(num_partitions)]
        
        elif num_partitions > num_classes:
            # or if more partitions than classes,
            # each partition will just drop 1 class.
            
            # first, each class is dropped at least once:
            class_to_drop = list(range(num_classes))
            
            # then sample from classes uniformly for each remaining partition:
            for i in range(num_partitions - num_classes):
                class_to_drop.append(np.random.choice(num_classes))
            # but format as list of lists for each partition, 
            # to match the classes >= partitions case above
            classes_to_drop = [[c] for c in class_to_drop]
        
        # finally, make the list that describes which classes each partition
        # DOES have access to, i.e. which ones are not dropped
        partition_classes = [[c for c in range(num_classes) if c not in classes_to_drop[p]] for p in range(num_partitions)]
        # and get the reverse mapping: which class is represented in which partitions
        class_partitions = [[p for p in range(num_partitions) if c in partition_classes[p]] for c in range (num_classes)] 
    
    else:
        # all partitions contain all classes:
        partition_classes = [list(range(num_classes)) for p in range(num_partitions)]
        class_partitions = [list(range(num_partitions)) for c in range(num_classes)]
        
    # now, handle 'weak' homogeneity (which is also included in strong homogeneity).
    # i.e. skewed distribution of classes between partitions which contain that class.
    
    # we'll populate this dict that enumerates the datapoint indices for each partition:
    partition_data_idxs = {p: [] for p in range(num_partitions)}

    samples_allocated = 0

    # slice out the labels column only, to identify the class distribution:
    # targets = np.asarray(dataset.select_columns('label').data).squeeze()
    targets = np.asarray([row['label'] for row in (dataset.select_columns('label')).to_list()])
    
    for c in range(num_classes):
        # get the indices in the dataset where this class occurs:
        class_example_idxs = np.where(targets == c)[0]
        num_class_datapoints = len(class_example_idxs)
        # print(f'{num_class_datapoints} samples of this class to allocate')
        # get the partitions that include this class:
        relevant_partitions = class_partitions[c]
        num_relevant = len(relevant_partitions)
        # print(f'  Where {num_relevant} partitions are relevant: {relevant_partitions}')
        
        # under perfect homoegeneity, we stratify the split such that each
        # relevant partition contains an equal fraction of a class's examples,
        # represented as equal spacing along the [0,1] number line:
        equal_distribution = np.linspace(0,1,num_relevant+1)
        
        # but if heterogeneity is at least weak, we perturb this distribution
        # to skew the distribution of classes across partitions:
        if heterogeneity in ('weak', 'strong'):
            # (I tried sampling uniformly across the number line for this,
            # but it leads to too many cases where boundaries are clustered together
            # resulting in almost no samples of a class, so instead this is an
            # iterative process with some guaranteed minimum number of datapoints per
            # class, equal to 20% of the expected equal allocation:
            tol = equal_distribution[1] * 0.2
            skewed_distribution = [equal_distribution[0]]
            for r in range(1,num_relevant):
                # sample uniformly from between the previous 'skewed' boundary
                # and the next 'equal' one:
                l_bound = skewed_distribution[-1] + tol
                u_bound = equal_distribution[r+1] - tol

                bound = np.random.uniform(l_bound, u_bound)
                skewed_distribution.append(bound)
            # as first bound is 0, final bound is 1:
            skewed_distribution.append(1)
            
            # use these perturbed points along the number line as a discrete PD:
            class_distribution = np.asarray(skewed_distribution)
        else:
            # non-heterogeneous, stratified split:
            class_distribution = equal_distribution
        
        # now translate those distributions to integer indices into the dataset 
        boundaries = (class_distribution * num_class_datapoints).astype(int)
        shuffled_class_example_idxs = np.random.permutation(class_example_idxs)
        data_idxs_per_rpartition = [shuffled_class_example_idxs[boundaries[r] : boundaries[r+1]] 
                                   for r in range(num_relevant)]
        # read 'rpartition' as 'relevant partition', i.e. the partitions that this class exists in
        num_samples_per_rpartition = [len(idxs) for idxs in data_idxs_per_rpartition]
        
        samples_allocated += sum(num_samples_per_rpartition)
        frac_allocated = samples_allocated / total_samples
        
        for r, idxs in enumerate(data_idxs_per_rpartition):
            # and, since we may be working with only a subset of all partitions
            # (because not all are guaranteed to contain this class)
            # we have to allocate them back into the appropriate partitions:
            p = relevant_partitions[r]
            partition_data_idxs[p].extend(data_idxs_per_rpartition[r])
        # this process is repeated for each class.
        
    # finally, create Dataset objects from the original Dataset 
    # which correspond to this partitioning scheme we've created

    partition_datasets = []
    for (p, idxs) in tqdm(partition_data_idxs.items(), desc='Creating partitions', ncols=80):
        partition_datasets.append(dataset.select(idxs))
    
    # verify that all samples are accounted for:
    total_partitioned_samples = sum([len(part) for part in partition_datasets])
    assert total_partitioned_samples == len(dataset), f"Missing samples during partitioning process - only {total_partitioned_samples}/{len(dataset)} accounted for ({(total_partitioned_samples / len(dataset)):.1%})"

    return partition_datasets


# util functions for the dataloader:
convnext_processor = ConvNextImageProcessor(image_mean=[0.5], image_std=[0.5])
def convnext_process_images(images: list[Image], device: str):
    return torch.tensor(np.stack(
        [convnext_processor(np.asarray([item]))['pixel_values'][0] for item in images]
        ), device=device)
def convnext_collate(loader_batch, device: str):
    images, labels = [b['image'] for b in loader_batch], [b['label'] for b in loader_batch]
    image_tensor = convnext_process_images(images, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
    return image_tensor, label_tensor

# for torchvision models:
tv_processor = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5], std=[0.5])
])
def tv_process_images(images: list[Image], device: str):
    processed_images = []
    for item in images:
        # cast to PIL image:
        if not isinstance(item, Image):
            item = Image(item)
        processed_images.append(tv_processor(item))
    return torch.stack(processed_images).to(device)
def tv_collate(batch: dict, device: str):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    
    image_tensor = tv_process_images(images, device=device)
    label_tensor = torch.tensor(labels, device=device)
    
    return image_tensor, label_tensor



def get_backbone(model_name, device):
    """initialises a vision model with a single input channel.
    model_name supports 'convnext', 'resnet', 'efficientnet' or 'squeezenet'.

    note that efficientnet trains surprisingly slowly in pytorch (I think due to depthwise convs)
    squeezenet is the smallest and fastest"""

    
    if model_name == 'convnext':
        configuration = ConvNextV2Config(num_channels=1)#, num_stages=2, depths=[3,3], hidden_sizes=[96,192])
        model = ConvNextV2Model(configuration).to(DEVICE)
        # add a global avg pooling and flatten step to the final layer:
        class pool_and_flatten(nn.Module):
            # small module to pool and flatten the last feature layer
            def forward(self, x):
                x = F.adaptive_avg_pool2d(x,(1,1))
                # x = x.flatten(start_dim=1)
                return x
        final_layer = model.encoder.stages[-1].layers[-1]                
        model.encoder.stages[-1].layers[-1] = nn.Sequential(final_layer, pool_and_flatten())
        model.out_features = model.config.hidden_sizes[-1]
        processor, collate_fn = convnext_process_images, convnext_collate
    elif model_name == 'resnet18':
        model = models.resnet18().to(device)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False, device=device)
        model.out_features = model.fc.out_features
        processor, collate_fn = tv_process_images, tv_collate
    elif model_name == 'resnet18_pretrained':
        # as above, but initialised with imagenet weights:
        try:
            model = models.resnet18(weights='IMAGENET1K_V1').to(device)
        except:
            model = models.resnet18(pretrained=True).to(device)
        # we still have to replace the first layer to account for one-channel input, but we copy over the
        # weights for one of the RGB channels as a heuristic:
        conv1_blue_weight = model.conv1.weight[:,2:3,:,:].cpu().detach() # the kernels for the blue channel

        # replace existing 3-channel input kernels with single-channel:
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False, device=device)
        # then copy over those blue weights:
        with torch.no_grad():
            model.conv1.weight.copy_(conv1_blue_weight)
        # replace the batchnorm too just in case:
        model.bn1 = torch.nn.BatchNorm2d(64, device=device)
        model.out_features = model.fc.out_features            
        processor, collate_fn = tv_process_images, tv_collate
        
    elif model_name == 'resnet50':
        model = models.resnet50().to(device)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False, device=device)
        model.out_features = model.fc.out_features
        processor, collate_fn = tv_process_images, tv_collate        
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0().to(device)
        model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False, device=device)
        model.out_features = list(model.modules())[-1].out_features
        processor, collate_fn = tv_process_images, tv_collate
    elif model_name == 'squeezenet':
        model = models.squeezenet1_1().to(device)
        model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, device=device)
        # the number of channels in the final global average pool layer:
        model.out_features = model.classifier[1].out_channels
        processor, collate_fn = tv_process_images, tv_collate
    else:
        raise ValueError(f"backbone model type ''{model_name}' not implemented")

    # processor and collate fn require the device arg too:
    d_processor = lambda x: processor(x, device=device)
    d_collate_fn = lambda x: collate_fn(x, device=device)
    return model, d_processor, d_collate_fn

def num_params(model, as_string=True):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    if as_string:
        return f'{total_params:,}'
    else:
        return total_params