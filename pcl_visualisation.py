import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps #, rcParams
from math import ceil

from datasets import Dataset


def plot_label_distribution(partitions: list[Dataset],
                            show_expected=True):
    """plots the distributionn of class labels within a collection
    of partitioned client Datasets, as a clustered bar chart."""
    
    num_partitions = len(partitions)
    total_samples = sum([len(part) for part in partitions])
    
    # assume all datasets share the same class mapping:
    class_names = partitions[0].features['label'].names
    num_classes = len(class_names)
    # num_classes = 3
    central_offset = num_classes/8 # originally 3.5 for 10 classes???
    
    x_loc = np.arange(num_partitions) # centres of each partition/cluster
    cluster_width = num_classes * 1.5 # how many bars worth of space to leave for each cluster
    bar_width = 1/cluster_width       # (given that 1 bar corresponds to a class in a partition)
                                      # here we leave 1 unit of whitespace for every 2 of data.
    
    
    
    fig, ax = plt.subplots(layout='constrained')
    
    samples_per_class_per_part = []
    
    cmap = colormaps['Blues']
    
    for p, partition in enumerate(partitions):
        labels = np.asarray(partition['label']).squeeze()
        label_idxs, label_counts = np.unique(labels, return_counts=True)
        # account for 0 labels in some datasets:
        full_counts = {i: 0 for i in range(num_classes)}
        for idx, count in zip(label_idxs, label_counts):
            full_counts[idx] = count
        assert list(label_idxs) == sorted(label_idxs)
        samples_per_class_per_part.append(full_counts)
    
    if show_expected:
        # plot the expected bar heights given homogeneous stratification:
        # total_samples = sum([sum([v for v in part.values()]) for part in samples_per_class_per_part])
        exp_count = (total_samples / num_partitions) / num_classes
        ax.axhline(exp_count, linestyle=':', c=cmap(0.3), label="'Expected' equal distribution", zorder=1)
    
    # part_labels = np.arange(num_partitions)
    
    side_width = (bar_width * num_classes/2)
    # i.e. how far to the left and right each set of bars extends from central point
    
    for c, name in enumerate(class_names[:num_classes]):
        # plot the cluster on figure axis:
        # offset = bar_width * c - (central_offset / cluster_width)
        offset = bar_width * (c+0.5) - side_width
        
        counts_per_part = [part[c] for part in samples_per_class_per_part]
        # pick a color by drawing from the colormap between range 128-255
        color = cmap((c / (num_classes*2)) + 0.5)
        rects = ax.bar(x_loc + offset, counts_per_part, bar_width, fc=color, zorder=2, label=None)
        # ax.bar_label(rects, padding=3)
    
    # indicate missing classes:
    miss_xs = []
    for p in range(num_partitions):
        for c in range(num_classes):
            if samples_per_class_per_part[p][c] == 0:
                bar_loc = x_loc[p] + (bar_width * (c+0.5) - side_width)
                miss_xs.append(bar_loc)
    if len(miss_xs) > 0:
        marker_height = total_samples / num_partitions / num_classes / 50
        plt.plot(miss_xs, [marker_height]*len(miss_xs), 'r^', fillstyle='none', label='Missing classes')
    
    ax.set_xlabel('Client')
    ax.set_ylabel('# of samples in class')
    ax.set_title('Class distribution by client')
    
    
    # minor ticks are the centres of each cluster:
    ax.set_xticks(x_loc, np.arange(num_partitions), minor=True)
    # and major ticks are the boundary between them:
    ax.set_xticks(x_loc + 0.5, [], minor=False)
    # show dividing lines between each cluster as well:
    [ax.axvline(xtick, c=cmap(0.2), linestyle='--') for xtick in (x_loc + 0.5)]
    
    ax.tick_params(axis='x', which='major', direction='out')
    ax.tick_params(axis='x', which='minor', length=0)
    
    
    ax.legend()#loc='upper right', ncols=1)
    
    plt.show()

    # return samples_per_class_per_part

def display_examples(dataset, batch_size=6, num_rows=None, size=100):
    """load a batch from the dataset
    and display the images and corresponding class labels."""
    class_names = dataset.features['label'].names
    num_classes = len(class_names)
    if num_rows is None:
        # automatically set up to 3 examples per row:
        num_rows = ceil(batch_size / 3)
    
    ### take a batch evenly-spaced across the dataset for class coverage:
    # batch_idxs = np.linspace(0, dataset.num_rows-1, batch_size).astype(int)
    
    # take a random batch:
    batch_idxs = np.random.choice(dataset.num_rows, batch_size)

    batch = dataset[batch_idxs]
    images = batch['image']
    labels = batch['label']
    classes = [class_names[l] for l in labels]

    fig, axes = plt.subplots(num_rows, ceil(batch_size / num_rows), squeeze=False, figsize=(size//10, 4*num_rows))
    all_axes = []
    for ax_row in axes:
        all_axes.extend(ax_row)
    
    for b, ax in enumerate(all_axes):
        if b < batch_size:
            ax.imshow(images[b], cmap='gray')
            ax.set_title(f'{labels[b]}: {classes[b]}')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
        else:
            ax.axis('off')
    plt.show()

def display_predictions(dataset, model, batch_size=6, num_rows=None, size=100):
    """as display_examples, but shows the predictions of a model on top as well"""
    
    class_names = dataset.features['label'].names
    num_classes = len(class_names)
    if num_rows is None:
        # automatically set up to 3 examples per row:
        num_rows = ceil(batch_size / 3)
    
    batch_idxs = np.random.choice(dataset.num_rows, batch_size)

    batch = dataset[batch_idxs]
    images = batch['image']
    labels = batch['label']
    classes = [class_names[l] for l in labels]

    ### call the model on this batch:
    inputs = model.processor(images)
    pred_logits = model(inputs)
    pred_classes = pred_logits.argmax(axis=1)
    # output first prediction:    
    pred_class_names = [class_names[cls] for cls in pred_classes]

    fig, axes = plt.subplots(num_rows, ceil(batch_size / num_rows), squeeze=False, figsize=(size//10, 4*num_rows))
    all_axes = []
    for ax_row in axes:
        all_axes.extend(ax_row)
    
    for b, ax in enumerate(all_axes):
        if b < batch_size:
            ax.imshow(images[b], cmap='gray')
            pred_label, actual_label = pred_classes[b].item(), labels[b]
            pred_name, actual_name = pred_class_names[b], classes[b]
            if pred_label == actual_label:
                # correct prediction
                ax.set_title(f' pred: {pred_name}\nlabel: {actual_name}', color='green', fontsize=8)
            else:
                # incorrect
                ax.set_title(f' pred: {pred_name} \nlabel: {actual_name}', color='red', fontsize=8)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

