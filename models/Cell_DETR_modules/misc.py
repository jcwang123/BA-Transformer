from typing import Tuple, List, Union

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.color import gray2rgb
from torchvision.utils import save_image


def to_one_hot(input: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Converts a given tensor to a one hot encoded tensor
    :param input: (torch.Tensor) Class number tensor
    :param num_classes: (int) Number of classes
    :return: (torch.Tensor) One hot tensor
    """
    one_hot = torch.zeros([input.shape[0], num_classes], dtype=torch.float)
    one_hot.scatter_(1, input.view(-1, 1).long(), 1)
    return one_hot


def normalize(input: torch.Tensor) -> torch.Tensor:
    """
    Normalize a given tensor to var=1.0 and mean=0.0
    :param input: (torch.Tensor) Input tensor
    :return: (torch.Tensor) Normalized output tensor
    """
    return (input - input.mean()) / input.std()


def normalize_0_1(input: torch.Tensor) -> torch.Tensor:
    """
    Normalize a given tensor to a range of [0, 1]
    :param input: (Torch tensor) Input tensor
    :param inplace: (bool) If true normalization is performed inplace
    :return: (Torch tensor) Normalized output tensor
    """
    # Perform normalization not inplace
    return (input - input.min()) / (input.max() - input.min())


class Logger(object):
    """
    Class to log different metrics
    """

    def __init__(self) -> None:
        self.metrics = dict()
        self.hyperparameter = dict()

    def log(self, metric_name: str, value: float) -> None:
        """
        Method writes a given metric value into a dict including list for every metric
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            self.metrics[metric_name] = [value]

    def save_metrics(self, path: str) -> None:
        """
        Static method to save dict of metrics
        :param metrics: (Dict[str, List[float]]) Dict including metrics
        :param path: (str) Path to save metrics
        :param add_time_to_file_name: (bool) True if time has to be added to filename of every metric
        """
        # Save dict of hyperparameter as json file
        with open(os.path.join(path, 'hyperparameter.txt'), 'w') as json_file:
            json.dump(self.hyperparameter, json_file)
        # Iterate items in metrics dict
        for metric_name, values in self.metrics.items():
            # Convert list of values to torch tensor to use build in save method from torch
            values = torch.tensor(values)
            # Save values
            torch.save(values, os.path.join(path, '{}.pt'.format(metric_name)))

    def get_average_metric_for_epoch(self, metric_name: str, epoch: int, epoch_name: str = 'epoch') -> float:
        """
        Method calculates the average of a metric for a given epoch
        :param metric_name: (str) Name of the metric
        :param epoch: (int) Epoch to average over
        :param epoch_name: (str) Name of epoch metric
        :return: (float) Average metric
        """
        # Convert lists to np.array
        metric = np.array(self.metrics[metric_name])
        epochs = np.array(self.metrics[epoch_name])
        # Calc mean
        metric_average = np.mean(metric[np.argwhere(epochs == epoch)])
        return float(metric_average)


def bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes: torch.Tensor) -> torch.Tensor:
    """
    This function converts a given bounding bix of the format
    [batch size, instances, 4 (x center, y center, width, height)] to [batch size, instances, 4 (x0, y0, x1, y1)].
    :param bounding_boxes: Bounding box of shape [batch size, instances, 4 (x center, y center, width, height)]
    :return: Converted bounding box of shape [batch size, instances, 4 (x0, y0, x1, y1)]
    """
    x_center, y_center, width, height = bounding_boxes.unbind(dim=-1)
    bounding_box_converted = [(x_center - 0.5 * width),
                              (y_center - 0.5 * height),
                              (x_center + 0.5 * width),
                              (y_center + 0.5 * height)]
    return torch.stack(tensors=bounding_box_converted, dim=-1)


def bounding_box_x0y0x1y1_to_xcycwh(bounding_boxes: torch.Tensor) -> torch.Tensor:
    """
    This function converts a given bounding bix of the format
    [batch size, instances, 4 (x0, y0, x1, y1)] to [batch size, instances, 4 (x center, y center, width, height)].
    :param bounding_boxes: Bounding box of shape [batch size, instances, 4 (x0, y0, x1, y1)]
    :return: Converted bounding box of shape [batch size, instances, 4 (x center, y center, width, height)]
    """
    x_0, y_0, x_1, y_1 = bounding_boxes.unbind(dim=-1)
    bounding_box_converted = [((x_0 + x_1) / 2),
                              ((y_0 + y_1) / 2),
                              (x_1 - x_0),
                              (y_1 - y_0)]
    return torch.stack(tensors=bounding_box_converted, dim=-1)


def relative_bounding_box_to_absolute(bounding_boxes: torch.Tensor, height: int, width: int,
                                      xcycwh: bool = False) -> torch.Tensor:
    """
    This function converts a relative bounding box to an absolute one for a given image shape. Inplace operation!
    :param bounding_boxes: (torch.Tensor) Bounding box with the format [batch size, instances, 4 (x0, y0, x1, y1)]
    :param height: (int) Height of the image
    :param width: (int) Width of the image
    :param xcycwh: (bool) True if the xcycwh format is given
    :return: (torch.Tensor) Absolute bounding box in the format [batch size, instances, 4 (x0, y0, x1, y1)]
    """
    # Case if xcycwh format is given
    if xcycwh:
        bounding_boxes = bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes)
    # Apply height and width
    bounding_boxes[..., [0, 2]] = bounding_boxes[..., [0, 2]] * width
    bounding_boxes[..., [1, 3]] = bounding_boxes[..., [1, 3]] * height
    # Return bounding box in the original format
    if xcycwh:
        return bounding_box_x0y0x1y1_to_xcycwh(bounding_boxes).long()
    return bounding_boxes.long()


def absolute_bounding_box_to_relative(bounding_boxes: torch.Tensor, height: int, width: int,
                                      xcycwh: bool = False) -> torch.Tensor:
    """
    This function converts an absolute bounding box to a relative one for a given image shape. Inplace operation!
    :param bounding_boxes: (torch.Tensor) Bounding box with the format [batch size, instances, 4 (x0, y0, x1, y1)]
    :param height: (int) Height of the image
    :param width: (int) Width of the image
    :param xcycwh: (bool) True if the xcycwh format is given
    :return: (torch.Tensor) Absolute bounding box in the format [batch size, instances, 4 (x0, y0, x1, y1)]
    """
    # Case if xcycwh format is given
    if xcycwh:
        bounding_boxes = bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes)
    # Apply height and width
    bounding_boxes[..., [0, 2]] = bounding_boxes[..., [0, 2]] / width
    bounding_boxes[..., [1, 3]] = bounding_boxes[..., [1, 3]] / height
    # Return bounding box in the original format
    if xcycwh:
        return bounding_box_x0y0x1y1_to_xcycwh(bounding_boxes)
    return bounding_boxes


def plot_instance_segmentation_overlay_instances_bb_classes(image: torch.Tensor, instances: torch.Tensor,
                                                            bounding_boxes: torch.Tensor,
                                                            class_labels: torch.Tensor, save: bool = False,
                                                            show: bool = False,
                                                            file_path: str = "", alpha: float = 0.3,
                                                            show_class_label: bool = True,
                                                            colors_traps: Tuple[Tuple[float, float, float], ...] = (
                                                                    (0.05, 0.05, 0.05),
                                                                    (0.25, 0.25, 0.25)),
                                                            cell_classes: Tuple[int, ...] = (2, 3),
                                                            colors_cells: Tuple[Tuple[float, float, float], ...] = (
                                                                    (1.0, 1.0, 0.0),
                                                                    (0.5, 1.0, 0.0),
                                                                    (0.0, 0.625, 1.0),
                                                                    (1.0, 0.0, 0.0),
                                                                    (0.125, 1.0, 0.0),
                                                                    (1.0, 0.375, 0.0),
                                                                    (1.0, 0.0, 0.375),
                                                                    (1.0, 0.0, 0.75),
                                                                    (0.5, 0.0, 1.0),
                                                                    (1.0, 0.75, 0.0),
                                                                    (0.125, 0.0, 1.0),
                                                                    (0.0, 1.0, 0.625),
                                                                    (0.0, 1.0, 0.25),
                                                                    (0.0, 0.25, 1.0),
                                                                    (0.875, 0.0, 1.0),
                                                                    (0.875, 1.0,
                                                                     0.0))) -> None:
    """
    Function produces an instance segmentation plot
    :param image: (torch.Tensor) Input image of shape (3, height, width) or (1, height, width)
    :param instances: (torch.Tensor) Instances masks of shape (instances, height, width)
    :param bounding_boxes: (torch.Tensor) Bounding boxes of shape (instances, 4 (x1, y1, x2, y2))
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be stored
    :param show: (bool) If true plt.show() will be called
    :param file_path: (str) Path and name where image will be stored
    :param show_class_label: (bool) If true class label will be shown in plot
    :param alpha: (float) Transparency factor of the instances
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_traps: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each trap instances.
    :param cell_classes: (Tuple[int, ...]) Tuple of cell classes
    """
    # Normalize image to [0, 255]
    image = normalize_0_1(image)
    # Convert data to numpy
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    instances = instances.detach().cpu().numpy()
    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Convert grayscale image to rgb
    if image.shape[-1] == 1:
        image = gray2rgb(image=image[:, :, 0])
    # Init counters
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Add instances to image
    for index, instance in enumerate(instances):
        # Case of cell instances
        if bool(class_labels[index] >= min(cell_classes)):
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(instance == 1,
                                          image[:, :, c] * (1 - alpha) + alpha *
                                          colors_cells[min(counter_cell_instance, len(colors_cells) - 1)][c],
                                          image[:, :, c])
            counter_cell_instance += 1
        # Case of trap class
        else:
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(instance == 1,
                                          image[:, :, c] * (1 - alpha) + alpha *
                                          colors_traps[min(counter_trap_instance, len(colors_traps) - 1)][c],
                                          image[:, :, c])
            counter_trap_instance += 1
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
    # Plot image and instances
    ax.imshow(image)
    # Init counters
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Plot bounding_boxes and classes
    for index, bounding_box in enumerate(bounding_boxes):
        # Case if cell is present
        if bool(class_labels[index] >= min(cell_classes)):
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_cells[min(counter_cell_instance, len(colors_cells) - 1)],
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                        'Cell', horizontalalignment='right', verticalalignment='bottom',
                        color="white", size=15)
            # Increment counter
            counter_cell_instance += 1
        # Cas if trap is present
        else:
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_traps[min(counter_trap_instance, len(colors_traps) - 1)],
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                        'Trap', horizontalalignment='right', verticalalignment='bottom',
                        color="white", size=15)
            # Increment counter
            counter_trap_instance += 1
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=image.shape[1] * 4 / 3.845, transparent=True, bbox_inches='tight', pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches='tight', pad_inches=0)
    # Close figure
    plt.close()


def plot_instance_segmentation_overlay_instances(image: torch.Tensor, instances: torch.Tensor,
                                                 class_labels: torch.Tensor, save: bool = False, show: bool = False,
                                                 file_path: str = "",
                                                 alpha: float = 0.5,
                                                 colors_cells: Tuple[Tuple[float, float, float], ...] = (
                                                         (1., 0., 0.89019608),
                                                         (1., 0.5, 0.90980392),
                                                         (0.7, 0., 0.70980392),
                                                         (0.7, 0.5, 0.73333333),
                                                         (0.5, 0., 0.53333333),
                                                         (0.5, 0.2, 0.55294118),
                                                         (0.3, 0., 0.45),
                                                         (0.3, 0.2, 0.45)),
                                                 colors_traps: Tuple[Tuple[float, float, float], ...] = (
                                                         (0.05, 0.05, 0.05),
                                                         (0.25, 0.25, 0.25)),
                                                 cell_classes: Tuple[int, ...] = (2, 3)) -> None:
    """
    Function produces an instance segmentation plot
    :param image: (torch.Tensor) Input image of shape (3, height, width) or (1, height, width)
    :param instances: (torch.Tensor) Instances masks of shape (instances, height, width)
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be stored
    :param show: (bool) If true plt.show() will be called
    :param file_path: (str) Path and name where image will be stored
    :param alpha: (float) Transparency factor
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_traps: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each trap instances.
    :param cell_classes: (Tuple[int, ...]) Tuple of cell classes
    """
    # Normalize image to [0, 1]
    image = normalize_0_1(image)
    # Convert data to numpy
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    instances = instances.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Convert grayscale image to rgb
    if image.shape[-1] == 1:
        image = gray2rgb(image=image[:, :, 0])
    # Init counters
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Add instances to image
    for index, instance in enumerate(instances):
        # Case of cell instances
        if bool(class_labels[index] >= min(cell_classes)):
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(instance == 1,
                                          image[:, :, c] * (1 - alpha) + alpha *
                                          colors_cells[min(counter_cell_instance, len(colors_cells) - 1)][c],
                                          image[:, :, c])
            counter_cell_instance += 1
        # Case of trap class
        else:
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(instance == 1,
                                          image[:, :, c] * (1 - alpha) + alpha *
                                          colors_traps[min(counter_trap_instance, len(colors_traps) - 1)][c],
                                          image[:, :, c])
            counter_trap_instance += 1

    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
    # Plot image and instances
    ax.imshow(image)
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=image.shape[1] * 4 / 3.845, transparent=True, bbox_inches='tight', pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches='tight', pad_inches=0)
    # Close figure
    plt.close()


def plot_instance_segmentation_labels(instances: torch.Tensor, bounding_boxes: torch.Tensor,
                                      class_labels: torch.Tensor, save: bool = False, show: bool = False,
                                      file_path: str = "",
                                      colors_cells: Tuple[Tuple[float, float, float], ...] = ((1., 0., 0.89019608),
                                                                                              (1., 0.5, 0.90980392),
                                                                                              (0.7, 0., 0.70980392),
                                                                                              (0.7, 0.5, 0.73333333),
                                                                                              (0.5, 0., 0.53333333),
                                                                                              (0.5, 0.2, 0.55294118),
                                                                                              (0.3, 0., 0.45),
                                                                                              (0.3, 0.2, 0.45)),
                                      colors_traps: Tuple[Tuple[float, float, float], ...] = (
                                              (0.3, 0.3, 0.3),
                                              (0.5, 0.5, 0.5)),
                                      cell_classes: Tuple[int, ...] = (2, 3), white_background: bool = False,
                                      show_class_label: bool = True) -> None:
    """
    Function plots given instance segmentation labels including the pixel-wise segmentation maps, bounding boxes,
    and class labels
    :param instances: (torch.Tensor) Pixel-wise instance segmentation map
    :param bounding_boxes: (torch.Tensor) Bounding boxes of shape (instances, 4 (x1, y1, x2, y2))
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be saved (matplotlib is used)
    :param show: (bool) If true matplotlib plot of the image will be shown
    :param file_path: (str) Path and name where image will be stored
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_traps: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each trap instances.
    :param white_background: (bool) If true a white background is utilized
    :param show_class_label: (bool) If true class name will be shown in the left bottom corner of each bounding box
    """
    # Convert data to numpy
    instances = instances.detach().cpu().numpy()
    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Init map to visualize instances
    instances_map = np.zeros((instances.shape[1], instances.shape[2], 3), dtype=np.float)
    # Init counters to track the number of cells and traps for different colours
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Instances to instances map
    for instance, class_label in zip(instances, class_labels):
        # Case if cell is present
        if bool(class_label >= min(cell_classes)):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_cells[min(counter_cell_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                             * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_cell_instance += 1
        # Cas if trap is present
        else:
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_traps[min(counter_trap_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                             * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_trap_instance += 1
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * instances_map.shape[0] / instances_map.shape[1])
    # Make background white if specified
    if white_background:
        for h in range(instances_map.shape[0]):
            for w in range(instances_map.shape[1]):
                if np.alltrue(instances_map[h, w, :] == np.array([0.0, 0.0, 0.0])):
                    instances_map[h, w, :] = np.array([1.0, 1.0, 1.0])
    # Plot image and instances
    ax.imshow(instances_map)
    # Init counters to track the number of cells and traps for different colours
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Plot bounding_boxes and classes
    for index, bounding_box in enumerate(bounding_boxes):
        # Case if cell is present
        if bool(class_labels[index] >= min(cell_classes)):
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_cells[min(counter_cell_instance, len(colors_cells) - 1)],
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                        'Cell', horizontalalignment='right', verticalalignment='bottom',
                        color="black" if white_background else "white", size=15)
            # Increment counter
            counter_cell_instance += 1
        # Cas if trap is present
        else:
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_traps[min(counter_trap_instance, len(colors_traps) - 1)],
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                        'Trap', horizontalalignment='right', verticalalignment='bottom',
                        color="black" if white_background else "white", size=15)
            # Increment counter
            counter_trap_instance += 1
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=instances_map.shape[1] * 4 / 3.845, transparent=True, bbox_inches='tight',
                    pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches='tight', pad_inches=0)
    # Close figure
    plt.close()


def plot_instance_segmentation_map_label(instances: torch.Tensor, class_labels: torch.Tensor, save: bool = False,
                                         show: bool = False, file_path: str = "",
                                         colors_cells: Tuple[Tuple[float, float, float], ...] = ((1., 0., 0.89019608),
                                                                                                 (1., 0.5, 0.90980392),
                                                                                                 (0.7, 0., 0.70980392),
                                                                                                 (0.7, 0.5, 0.73333333),
                                                                                                 (0.5, 0., 0.53333333),
                                                                                                 (0.5, 0.2, 0.55294118),
                                                                                                 (0.3, 0., 0.45),
                                                                                                 (0.3, 0.2, 0.45)),
                                         colors_traps: Tuple[Tuple[float, float, float], ...] = (
                                                 (0.3, 0.3, 0.3),
                                                 (0.5, 0.5, 0.5)),
                                         cell_classes: Tuple[int, ...] = (2, 3),
                                         white_background: bool = False) -> None:
    """
    Function plots given instance segmentation labels including the pixel-wise segmentation maps, bounding boxes,
    and class labels
    :param instances: (torch.Tensor) Pixel-wise instance segmentation map
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be saved (matplotlib is used)
    :param show: (bool) If true matplotlib plot of the image will be shown
    :param file_path: (str) File path to save the image
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_traps: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each trap instances.
    :param white_background: (bool) If true a white background is utilized
    """
    # Convert data to numpy
    instances = instances.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Init map to visualize instances
    instances_map = np.zeros((instances.shape[1], instances.shape[2], 3), dtype=np.float)
    # Init counters to track the number of cells and traps for different colours
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Instances to instances map
    for instance, class_label in zip(instances, class_labels):
        # Case if cell is present
        if bool(class_label >= min(cell_classes)):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_cells[min(counter_cell_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                             * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_cell_instance += 1
        # Cas if trap is present
        else:
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_traps[min(counter_trap_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                             * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_trap_instance += 1
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * instances_map.shape[0] / instances_map.shape[1])
    # Make background white if specified
    if white_background:
        for h in range(instances_map.shape[0]):
            for w in range(instances_map.shape[1]):
                if np.alltrue(instances_map[h, w, :] == np.array([0.0, 0.0, 0.0])):
                    instances_map[h, w, :] = np.array([1.0, 1.0, 1.0])
    # Plot image and instances
    ax.imshow(instances_map)
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=instances_map.shape[1] * 4 / 3.845, transparent=True, bbox_inches='tight',
                    pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches='tight', pad_inches=0)
    # Close figure
    plt.close()


def plot_image(image: torch.Tensor, save: bool = False, show: bool = False, file_path: str = "") -> None:
    """
    This function plots and saves an images
    :param image: (torch.Tensor) Image as a torch tensor
    :param save: (bool) If true image will be saved (torchvision save_image function is utilized)
    :param show: (bool) If true matplotlib plot of the image will be shown
    :param file_path: (str) File path to save the image
    """
    # Make sure image tensor is not on GPU an is not attached to graph
    image = image.cpu().detach()
    # Normalize image to [0, 255]
    image = normalize_0_1(image)
    # Save image if utilized
    if save:
        # Add batch dim to image if needed
        image_ = image.unsqueeze(dim=0) if image.ndim == 3 else image
        save_image(image_, file_path, nrow=1, padding=0, normalize=False)
    # Show matplotlib plot if utilized
    if show:
        # Change oder of dims to match matplotlib format and convert to numpy
        image = image.permute(1, 2, 0).numpy()
        # Init figure
        fig, ax = plt.subplots()
        # Set size
        fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
        # Plot image and instances
        ax.imshow(image[:, :, 0], cmap="gray")
        # Axis off
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show(bbox_inches='tight', pad_inches=0)
        # Close figure
        plt.close()


def plot_instance_segmentation_overlay_bb_classes(image: torch.Tensor, bounding_boxes: torch.Tensor,
                                                  class_labels: torch.Tensor, save: bool = False, show: bool = False,
                                                  file_path: str = "",
                                                  show_class_label: bool = True,
                                                  colors_cells: Tuple[Tuple[float, float, float], ...] =
                                                  (1., 0., 0.89019608),
                                                  colors_traps: Tuple[Tuple[float, float, float], ...] =
                                                  (0.0, 0.0, 0.0),
                                                  cell_classes: Tuple[int, ...] = (2, 3)) -> None:
    """
    Function produces an instance segmentation plot
    :param image: (torch.Tensor) Input image of shape (3, height, width) or (1, height, width)
    :param bounding_boxes: (torch.Tensor) Bounding boxes of shape (instances, 4 (x1, y1, x2, y2))
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be stored
    :param show: (bool) If true plt.show() will be called
    :param file_path: (str) Path and name where image will be stored
    :param show_class_label: (bool) If true class label is show in plot
    :param alpha: (float) Transparency factor
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_traps: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each trap instances.
    :param cell_classes: (Tuple[int, ...]) Tuple of cell classes
    """
    # Normalize image to [0, 255]
    image = normalize_0_1(image)
    # Convert data to numpy
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Convert grayscale image to rgb
    if image.shape[-1] == 1:
        image = gray2rgb(image=image[:, :, 0])
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
    # Plot image and instances
    ax.imshow(image)
    # Plot bounding_boxes and classes
    for index, bounding_box in enumerate(bounding_boxes):
        # Case if cell is present
        if bool(class_labels[index] >= min(cell_classes)):
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_cells,
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0])) - 2,
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1])) - 2,
                        'Cell', horizontalalignment='right', verticalalignment='bottom', color="white", size=15)
        # Cas if trap is present
        else:
            rectangle = patches.Rectangle((float(bounding_box[0]), float(bounding_box[1])),
                                          float(bounding_box[2]) - float(bounding_box[0]),
                                          float(bounding_box[3]) - float(bounding_box[1]),
                                          linewidth=3,
                                          edgecolor=colors_traps,
                                          facecolor='none', ls='dashed')
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0])) - 2,
                        float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1])) - 2,
                        'Trap', horizontalalignment='right', verticalalignment='bottom', color="white", size=15)
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=image.shape[1] * 4 / 3.845, transparent=True, bbox_inches='tight', pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches='tight', pad_inches=0)
    # Close figure
    plt.close()


def plot_instance_segmentation_instances(instances: torch.Tensor, class_labels: torch.Tensor, save: bool = False,
                                         show: bool = False, file_path: str = "",
                                         colors_cells: Tuple[Tuple[float, float, float], ...] = ((1., 0., 0.89019608),
                                                                                                 (1., 0.5, 0.90980392),
                                                                                                 (0.7, 0., 0.70980392),
                                                                                                 (0.7, 0.5, 0.73333333),
                                                                                                 (0.5, 0., 0.53333333),
                                                                                                 (0.5, 0.2, 0.55294118),
                                                                                                 (0.3, 0., 0.45),
                                                                                                 (0.3, 0.2, 0.45)),
                                         colors_traps: Tuple[Tuple[float, float, float], ...] = (
                                                 (0.3, 0.3, 0.3),
                                                 (0.5, 0.5, 0.5)),
                                         cell_classes: Tuple[int, ...] = (2, 3),
                                         white_background: bool = False) -> None:
    """
    Function plots given instance segmentation labels including the pixel-wise segmentation maps, bounding boxes,
    and class labels
    :param instances: (torch.Tensor) Pixel-wise instance segmentation map
    :param bounding_boxes: (torch.Tensor) Bounding boxes of shape (instances, 4 (x1, y1, x2, y2))
    :param class_labels: (torch.Tensor) Class labels of each instance (instances, )
    :param save: (bool) If true image will be saved (matplotlib is used)
    :param show: (bool) If true matplotlib plot of the image will be shown
    :param file_path: (str) File path to save the image
    :param colors_cells: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each cell instances.
    :param colors_traps: (Tuple[Tuple[float, float, float], ...]) Tuple of RGB colors to visualize each trap instances.
    :param white_background: (bool) If true a white background is utilized
    """
    # Convert data to numpy
    instances = instances.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Init counters to track the number of cells and traps for different colours
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Instances to instances map
    for index, data in enumerate(zip(instances, class_labels)):
        # Unzip data
        instance, class_label = data
        # Case if cell is present
        if bool(class_label >= min(cell_classes)):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instance = np.array(colors_cells[min(counter_cell_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                       * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_cell_instance += 1
        # Cas if trap is present
        else:
            # Add pixels of current instance, in the corresponding colour, to instances map
            instance = np.array(colors_traps[min(counter_trap_instance, len(colors_cells) - 1)]).reshape(1, 1, 3) \
                       * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_trap_instance += 1
        # Init figure
        fig, ax = plt.subplots()
        # Set size
        fig.set_size_inches(5, 5 * instance.shape[0] / instance.shape[1])
        # Make background white if specified
        if white_background:
            for h in range(instance.shape[0]):
                for w in range(instance.shape[1]):
                    if np.alltrue(instance[h, w, :] == np.array([0.0, 0.0, 0.0])):
                        instance[h, w, :] = np.array([1.0, 1.0, 1.0])
        # Plot image and instances
        ax.imshow(instance)
        # Axis off
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Save figure if utilized
        if save:
            plt.savefig(file_path.replace(".", "_{}.".format(index)), dpi=instance.shape[1] * 4 / 3.845,
                        transparent=True, bbox_inches='tight', pad_inches=0)
        # Show figure if utilized
        if show:
            plt.show(bbox_inches='tight', pad_inches=0)
        # Close figure
        plt.close()


def giou(bounding_box_1: torch.Tensor, bounding_box_2: torch.Tensor,
         return_iou: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Function computes the general IoU for two given bounding boxes
    :param bounding_box_1: (torch.Tensor) Bounding box prediction of shape (batch size, instances, 4)
    :param bounding_box_2: (torch.Tensor) Bounding box labels of shape (batch size, instances, 4)
    :param return_iou: (bool) If true the normal IoU is also returned
    :return: (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) GIoU loss value for each sample and iou optimal
    """
    # Get areas of bounding boxes
    prediction_area = (bounding_box_1[..., 2] - bounding_box_1[..., 0]) * (
            bounding_box_1[..., 3] - bounding_box_1[..., 1])
    label_area = (bounding_box_2[..., 2] - bounding_box_2[..., 0]) * (bounding_box_2[..., 3] - bounding_box_2[..., 1])
    # Calc anchors
    left_top_anchors = torch.max(bounding_box_1[..., None, :2], bounding_box_2[..., :2])
    right_bottom_anchors = torch.min(bounding_box_1[..., None, 2:], bounding_box_2[..., 2:])
    # Calc width and height and clamp if needed
    width_height = (right_bottom_anchors - left_top_anchors).clamp(min=0.0)
    # Calc intersection
    intersection = width_height[..., 0] * width_height[..., 1]
    # Calc union
    union = prediction_area + label_area - intersection
    # Calc IoU
    iou = (intersection / union)
    # Calc anchors for smallest convex hull
    left_top_anchors_convex_hull = torch.min(bounding_box_1[..., :2], bounding_box_2[..., :2])
    right_bottom_anchors_convex_hull = torch.max(bounding_box_1[..., 2:], bounding_box_2[..., 2:])
    # Calc width and height and clamp if needed
    width_height_convex_hull = (right_bottom_anchors_convex_hull - left_top_anchors_convex_hull).clamp(min=0.0)
    # Calc area of convex hull
    area_convex_hull = width_height_convex_hull[..., 0] * width_height_convex_hull[..., 1]
    # Calc gIoU
    giou = (iou - ((area_convex_hull - union) / area_convex_hull))
    # Return also the iou if needed
    if return_iou:
        return giou, iou
    return giou


def giou_for_matching(bounding_box_1: torch.Tensor, bounding_box_2: torch.Tensor) -> torch.Tensor:
    """
    Function computes the general IoU for two given bounding boxes
    :param bounding_box_1: (torch.Tensor) Bounding box prediction of shape (batch size, instances, 4)
    :param bounding_box_2: (torch.Tensor) Bounding box labels of shape (batch size, instances, 4)
    :return: (torch.Tensor) GIoU matrix for matching
    """
    # Get areas of bounding boxes
    bounding_box_1_area = (bounding_box_1[:, 2] - bounding_box_1[:, 0]) * (bounding_box_1[:, 3] - bounding_box_1[:, 1])
    bounding_box_2_area = (bounding_box_2[:, 2] - bounding_box_2[:, 0]) * (bounding_box_2[:, 3] - bounding_box_2[:, 1])
    # Calc anchors
    left_top_anchors = torch.max(bounding_box_1[:, None, :2], bounding_box_2[:, :2])
    right_bottom_anchors = torch.min(bounding_box_1[:, None, 2:], bounding_box_2[:, 2:])
    # Calc width and height and clamp if needed
    width_height = (right_bottom_anchors - left_top_anchors).clamp(min=0.0)
    # Calc intersection
    intersection = width_height[:, :, 0] * width_height[:, :, 1]
    # Calc union
    union = bounding_box_1_area[:, None] + bounding_box_2_area - intersection
    # Calc IoU
    iou = (intersection / union)
    # Calc anchors for smallest convex hull
    left_top_anchors_convex_hull = torch.min(bounding_box_1[:, None, :2], bounding_box_2[..., :2])
    right_bottom_anchors_convex_hull = torch.max(bounding_box_1[:, None, 2:], bounding_box_2[..., 2:])
    # Calc width and height and clamp if needed
    width_height_convex_hull = (right_bottom_anchors_convex_hull - left_top_anchors_convex_hull).clamp(min=0.0)
    # Calc area of convex hull
    area_convex_hull = width_height_convex_hull[:, :, 0] * width_height_convex_hull[:, :, 1]
    # Calc gIoU
    giou = (iou - ((area_convex_hull - union) / area_convex_hull))
    return giou


def iterable_to_device(data: List[torch.Tensor], device: str = "cuda") -> List[torch.Tensor]:
    """
    Function maps data to a given device.
    :param data: (List[torch.Tensor]) List of torch tensors
    :param device: (str) Device to be used
    :return: (List[torch.Tensor]) Input data mapped to the given device
    """
    # Iterate over all tensors
    for index in range(len(data)):
        # Map tensors to device
        data[index] = data[index].to(device)
    return data
