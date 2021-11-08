from typing import Union, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np
import setproctitle

from detr import CellDETR
import misc
import validation_metric


class ModelWrapper(object):
    """
    This class implements a wrapper for Cell.DETR, optimizer, datasets, and loss functions. This class implements also
    the training, validation and test method.
    """

    def __init__(self,
                 detr: Union[nn.DataParallel, CellDETR],
                 detr_optimizer: torch.optim.Optimizer,
                 detr_segmentation_optimizer: torch.optim.Optimizer,
                 training_dataset: DataLoader,
                 validation_dataset: DataLoader,
                 test_dataset: DataLoader,
                 loss_function: nn.Module,
                 learning_rate_schedule: torch.optim.lr_scheduler.MultiStepLR = None,
                 device: str = "cuda",
                 save_data_path: str = "saved_data",
                 use_telegram: bool = True) -> None:
        """
        Constructor method
        :param detr: (Union[nn.DataParallel, DETR]) DETR model
        :param detr_optimizer: (torch.optim.Optimizer) DETR model optimizer
        :param detr_segmentation_optimizer: (torch.optim.Optimizer) DETR segmentation head optimizer
        :param training_dataset: (DataLoader) Training dataset
        :param validation_dataset: (DataLoader) Validation dataset
        :param test_dataset: (DataLoader) Test dataset
        :param loss_function: (nn.Module) Loss function
        :param learning_rate_schedule: (torch.optim.lr_scheduler.MultiStepLR) Learning rate schedule
        :param device: (str) Device to be utilized
        :param save_data_path: (str) Path to store log data
        :param use_telegram: (bool) If true telegram_send is used
        """
        # Save parameters
        self.detr = detr
        self.detr_optimizer = detr_optimizer
        self.detr_segmentation_optimizer = detr_segmentation_optimizer
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.loss_function = loss_function
        self.learning_rate_schedule = learning_rate_schedule
        self.device = device
        self.save_data_path = save_data_path
        self.use_telegram = use_telegram
        # Init logger
        self.logger = misc.Logger()
        # Make directories to save logs, plots and models during training
        time_and_date = str(datetime.now())
        save_data_path = os.path.join(save_data_path, time_and_date)
        os.makedirs(save_data_path, exist_ok=True)
        self.path_save_models = os.path.join(save_data_path, "models")
        os.makedirs(self.path_save_models, exist_ok=True)
        self.path_save_plots = os.path.join(save_data_path, "plots")
        os.makedirs(self.path_save_plots, exist_ok=True)
        self.path_save_metrics = os.path.join(save_data_path, "metrics")
        os.makedirs(self.path_save_metrics, exist_ok=True)
        # Init variable to store best mIoU
        self.best_miou = 0.0

    def train(self, epochs: int = 20, validate_after_n_epochs: int = 5, save_model_after_n_epochs: int = 10,
              optimize_only_segmentation_head_after_epoch: int = 150) -> None:
        """
        Training method
        :param epochs: (int) Number of epochs to perform
        :param validate_after_n_epochs: (int) Number of epochs after the validation is performed
        :param save_model_after_n_epochs: (int) Number epochs after the current models is saved
        :param optimize_only_segmentation_head_after_epoch: (int) Number of epochs after only the seg. head is trained
        """
        # Model into train mode
        self.detr.train()
        # Model to device
        self.detr.to(self.device)
        # Init progress bar
        self.progress_bar = tqdm(total=epochs * len(self.training_dataset.dataset))
        # Main trainings loop
        for epoch in range(epochs):
            for input, instance_labels, bounding_box_labels, class_labels in self.training_dataset:
                # Update progress bar
                self.progress_bar.update(n=input.shape[0])
                # Data to device
                input = input.to(self.device)
                instance_labels = misc.iterable_to_device(instance_labels, device=self.device)
                bounding_box_labels = misc.iterable_to_device(bounding_box_labels, device=self.device)
                class_labels = misc.iterable_to_device(class_labels, device=self.device)
                # Reset gradients
                self.detr.zero_grad()
                # Get prediction
                class_predictions, bounding_box_predictions, instance_predictions = self.detr(input)
                # Calc loss
                loss_classification, loss_bounding_box, loss_segmentation = self.loss_function(class_predictions,
                                                                                               bounding_box_predictions,
                                                                                               instance_predictions,
                                                                                               class_labels,
                                                                                               bounding_box_labels,
                                                                                               instance_labels)
                # Case if the whole network is optimized
                if epoch < optimize_only_segmentation_head_after_epoch:
                    # Perform backward pass to compute the gradients
                    (loss_classification + loss_bounding_box + loss_segmentation).backward()
                    # Optimize detr
                    self.detr_optimizer.step()
                else:
                    # Perform backward pass to compute the gradients
                    loss_segmentation.backward()
                    # Optimize detr
                    self.detr_segmentation_optimizer.step()
                # Show losses in progress bar
                self.progress_bar.set_description(
                    "Epoch {}/{} Best val. mIoU={:.4f} Loss C.={:.4f} Loss BB.={:.4f} Loss Seg.={:.4f}".format(
                        epoch + 1, epochs, self.best_miou, loss_classification.item(), loss_bounding_box.item(),
                        loss_segmentation.item()))
                # Log losses
                self.logger.log(metric_name="loss_classification", value=loss_classification.item())
                self.logger.log(metric_name="loss_bounding_box", value=loss_bounding_box.item())
                self.logger.log(metric_name="loss_segmentation", value=loss_segmentation.item())
            # Learning rate schedule step
            if self.learning_rate_schedule is not None:
                self.learning_rate_schedule.step()
            # Validate
            if (epoch + 1) % validate_after_n_epochs == 0:
                self.validate(epoch=epoch, train=True)
            # Save model
            if (epoch + 1) % save_model_after_n_epochs == 0:
                torch.save(
                    self.detr.module.state_dict() if isinstance(self.detr, nn.DataParallel) else self.detr.state_dict(),
                    os.path.join(self.path_save_models, "detr_{}.pt".format(epoch)))
        # Final validation
        self.validate(epoch=epoch, number_of_plots=30)
        # Close progress bar
        self.progress_bar.close()
        # Load best model
        self.detr.state_dict(torch.load(os.path.join(self.path_save_models, "detr_best_model.pt")))

    @torch.no_grad()
    def validate(self, validation_metrics_classification: Tuple[nn.Module, ...] = (
            validation_metric.ClassificationAccuracy(),),
                 validation_metrics_bounding_box: Tuple[nn.Module, ...] = (
                         nn.L1Loss(), nn.MSELoss(), validation_metric.BoundingBoxIoU(),
                         validation_metric.BoundingBoxGIoU()),
                 validation_metrics_segmentation: Tuple[nn.Module, ...] = (
                         validation_metric.Accuracy(), validation_metric.Precision(), validation_metric.Recall(),
                         validation_metric.F1(), validation_metric.IoU(), validation_metric.MIoU(),
                         validation_metric.Dice(), validation_metric.CellIoU(),
                         validation_metric.MeanAveragePrecision(), validation_metric.InstancesAccuracy()),
                 epoch: int = -1, number_of_plots: int = 5, train: bool = False) -> None:
        """
        Validation method
        :param validation_metrics_classification: (Tuple[nn.Module, ...]) Validation modules for classification
        :param validation_metrics_bounding_box: (Tuple[nn.Module, ...]) Validation modules for bounding boxes
        :param validation_metrics_segmentation: (Tuple[nn.Module, ...]) Validation modules for segmentation
        :param epoch: (int) Current epoch
        :param number_of_plots: (int) Number of validation plot to be produced
        :param train: (bool) Train flag if set best model is saved based on val iou
        """
        # DETR to device
        self.detr.to(self.device)
        # DETR into eval mode
        self.detr.eval()
        # Init dicts to store metrics
        metrics_classification = dict()
        metrics_bounding_box = dict()
        metrics_segmentation = dict()
        # Init indexes of elements to be plotted
        plot_indexes = np.random.choice(np.arange(0, len(self.validation_dataset)), number_of_plots, replace=False)
        # Main loop over the validation set
        for index, batch in enumerate(self.validation_dataset):
            # Get data from batch
            input, instance_labels, bounding_box_labels, class_labels = batch
            # Data to device
            input = input.to(self.device)
            instance_labels = misc.iterable_to_device(instance_labels, device=self.device)
            bounding_box_labels = misc.iterable_to_device(bounding_box_labels, device=self.device)
            class_labels = misc.iterable_to_device(class_labels, device=self.device)
            # Get prediction
            class_predictions, bounding_box_predictions, instance_predictions = self.detr(input)
            # Perform matching
            matching_indexes = self.loss_function.matcher(class_predictions, bounding_box_predictions,
                                                          class_labels, bounding_box_labels)
            # Apply permutation to labels and predictions
            class_predictions, class_labels = self.loss_function.apply_permutation(prediction=class_predictions,
                                                                                   label=class_labels,
                                                                                   indexes=matching_indexes)
            bounding_box_predictions, bounding_box_labels = self.loss_function.apply_permutation(
                prediction=bounding_box_predictions,
                label=bounding_box_labels,
                indexes=matching_indexes)
            instance_predictions, instance_labels = self.loss_function.apply_permutation(
                prediction=instance_predictions,
                label=instance_labels,
                indexes=matching_indexes)
            for batch_index in range(len(class_labels)):
                # Calc validation metrics for classification
                for validation_metric_classification in validation_metrics_classification:
                    # Calc metric
                    metric = validation_metric_classification(
                        class_predictions[batch_index, :class_labels[batch_index].shape[0]].argmax(dim=-1),
                        class_labels[batch_index].argmax(dim=-1)).item()
                    # Save metric and name of metric
                    if validation_metric_classification.__class__.__name__ in metrics_classification.keys():
                        metrics_classification[validation_metric_classification.__class__.__name__].append(metric)
                    else:
                        metrics_classification[validation_metric_classification.__class__.__name__] = [metric]
                # Calc validation metrics for bounding boxes
                for validation_metric_bounding_box in validation_metrics_bounding_box:
                    # Calc metric
                    metric = validation_metric_bounding_box(
                        bounding_box_predictions[batch_index, :bounding_box_labels[batch_index].shape[0]],
                        bounding_box_labels[batch_index]).item()
                    # Save metric and name of metric
                    if validation_metric_bounding_box.__class__.__name__ in metrics_bounding_box.keys():
                        metrics_bounding_box[validation_metric_bounding_box.__class__.__name__].append(metric)
                    else:
                        metrics_bounding_box[validation_metric_bounding_box.__class__.__name__] = [metric]
                # Calc validation metrics for bounding boxes
                for validation_metric_segmentation in validation_metrics_segmentation:
                    # Calc metric
                    metric = validation_metric_segmentation(
                        instance_predictions[batch_index, :instance_labels[batch_index].shape[0]],
                        instance_labels[batch_index], class_label=class_labels[batch_index].argmax(dim=-1)).item()
                    # Save metric and name of metric
                    if validation_metric_segmentation.__class__.__name__ in metrics_segmentation.keys():
                        metrics_segmentation[validation_metric_segmentation.__class__.__name__].append(metric)
                    else:
                        metrics_segmentation[validation_metric_segmentation.__class__.__name__] = [metric]
            if index in plot_indexes:
                # Plot
                object_classes = class_predictions[0].argmax(dim=-1).cpu().detach()
                # Case the no objects are detected
                if object_classes.shape[0] > 0:
                    object_indexes = torch.from_numpy(np.argwhere(object_classes.numpy() > 0)[:, 0])
                    bounding_box_predictions = misc.relative_bounding_box_to_absolute(
                        misc.bounding_box_xcycwh_to_x0y0x1y1(
                            bounding_box_predictions[0, object_indexes].cpu().clone().detach()), height=input.shape[-2],
                        width=input.shape[-1])
                    misc.plot_instance_segmentation_overlay_instances_bb_classes(image=input[0],
                                                                                 instances=(instance_predictions[0][
                                                                                                object_indexes] > 0.5).float(),
                                                                                 bounding_boxes=bounding_box_predictions,
                                                                                 class_labels=object_classes[
                                                                                     object_indexes],
                                                                                 show=False, save=True,
                                                                                 file_path=os.path.join(
                                                                                     self.path_save_plots,
                                                                                     "validation_plot_is_bb_c_{}_{}.png".format(
                                                                                         epoch + 1, index)))
        # Average metrics and save them in logs
        for metric_name in metrics_classification:
            self.logger.log(metric_name=metric_name + "_classification_val",
                            value=float(np.mean(metrics_classification[metric_name])))
        for metric_name in metrics_bounding_box:
            self.logger.log(metric_name=metric_name + "_bounding_box_val",
                            value=float(np.mean(metrics_bounding_box[metric_name])))
        for metric_name in metrics_segmentation:
            metric_values = np.array(metrics_segmentation[metric_name])
            # Save best mIoU model if training is utilized
            if train and "MIoU" in metric_name and float(np.mean(metrics_segmentation[metric_name])) > self.best_miou:
                # Save current mIoU
                self.best_miou = float(np.mean(metric_values[~np.isnan(metric_values)]))
                # Show best MIoU as process name
                setproctitle.setproctitle("Cell-DETR best MIoU={:.4f}".format(self.best_miou))
                # Save model
                torch.save(
                    self.detr.module.state_dict() if isinstance(self.detr, nn.DataParallel) else self.detr.state_dict(),
                    os.path.join(self.path_save_models, "detr_best_model.pt"))
            self.logger.log(metric_name=metric_name + "_segmentation_val",
                            value=float(np.mean(metric_values[~np.isnan(metric_values)])))
        # Save metrics
        self.logger.save_metrics(path=self.path_save_metrics)

    @torch.no_grad()
    def test(self, test_metrics_classification: Tuple[nn.Module, ...] = (validation_metric.ClassificationAccuracy(),),
             test_metrics_bounding_box: Tuple[nn.Module, ...] = (
                     nn.L1Loss(), nn.MSELoss(), validation_metric.BoundingBoxIoU(),
                     validation_metric.BoundingBoxGIoU()),
             test_metrics_segmentation: Tuple[nn.Module, ...] = (
                     validation_metric.Accuracy(), validation_metric.Precision(), validation_metric.Recall(),
                     validation_metric.F1(), validation_metric.IoU(), validation_metric.MIoU(),
                     validation_metric.Dice(), validation_metric.CellIoU(),
                     validation_metric.MeanAveragePrecision(), validation_metric.InstancesAccuracy())) -> None:
        """
        Test method
        :param test_metrics_classification: (Tuple[nn.Module, ...]) Test modules for classification
        :param test_metrics_bounding_box: (Tuple[nn.Module, ...]) Test modules for bounding boxes
        :param test_metrics_segmentation: (Tuple[nn.Module, ...]) Test modules for segmentation
        """
        # DETR to device
        self.detr.to(self.device)
        # DETR into eval mode
        self.detr.eval()
        # Init dicts to store metrics
        metrics_classification = dict()
        metrics_bounding_box = dict()
        metrics_segmentation = dict()
        # Main loop over the test set
        for index, batch in enumerate(self.test_dataset):
            # Get data from batch
            input, instance_labels, bounding_box_labels, class_labels = batch
            # Data to device
            input = input.to(self.device)
            instance_labels = misc.iterable_to_device(instance_labels, device=self.device)
            bounding_box_labels = misc.iterable_to_device(bounding_box_labels, device=self.device)
            class_labels = misc.iterable_to_device(class_labels, device=self.device)
            # Get prediction
            class_predictions, bounding_box_predictions, instance_predictions = self.detr(input)
            # Perform matching
            matching_indexes = self.loss_function.matcher(class_predictions, bounding_box_predictions,
                                                          class_labels, bounding_box_labels)
            # Apply permutation to labels and predictions
            class_predictions, class_labels = self.loss_function.apply_permutation(prediction=class_predictions,
                                                                                   label=class_labels,
                                                                                   indexes=matching_indexes)
            bounding_box_predictions, bounding_box_labels = self.loss_function.apply_permutation(
                prediction=bounding_box_predictions,
                label=bounding_box_labels,
                indexes=matching_indexes)
            instance_predictions, instance_labels = self.loss_function.apply_permutation(
                prediction=instance_predictions,
                label=instance_labels,
                indexes=matching_indexes)
            for batch_index in range(len(class_labels)):
                # Calc test metrics for classification
                for test_metric_classification in test_metrics_classification:
                    # Calc metric
                    metric = test_metric_classification(
                        class_predictions[batch_index, :class_labels[batch_index].shape[0]].argmax(dim=-1),
                        class_labels[batch_index].argmax(dim=-1)).item()
                    # Save metric and name of metric
                    if test_metric_classification.__class__.__name__ in metrics_classification.keys():
                        metrics_classification[test_metric_classification.__class__.__name__].append(metric)
                    else:
                        metrics_classification[test_metric_classification.__class__.__name__] = [metric]
                # Calc test metrics for bounding boxes
                for test_metric_bounding_box in test_metrics_bounding_box:
                    # Calc metric
                    metric = test_metric_bounding_box(
                        misc.bounding_box_xcycwh_to_x0y0x1y1(
                            bounding_box_predictions[batch_index, :bounding_box_labels[batch_index].shape[0]]),
                        misc.bounding_box_xcycwh_to_x0y0x1y1(bounding_box_labels[batch_index])).item()
                    # Save metric and name of metric
                    if test_metric_bounding_box.__class__.__name__ in metrics_bounding_box.keys():
                        metrics_bounding_box[test_metric_bounding_box.__class__.__name__].append(metric)
                    else:
                        metrics_bounding_box[test_metric_bounding_box.__class__.__name__] = [metric]
                # Calc test metrics for bounding boxes
                for test_metric_segmentation in test_metrics_segmentation:
                    # Calc metric
                    metric = test_metric_segmentation(
                        instance_predictions[batch_index, :instance_labels[batch_index].shape[0]],
                        instance_labels[batch_index], class_label=class_labels[batch_index].argmax(dim=-1)).item()
                    # Save metric and name of metric
                    if test_metric_segmentation.__class__.__name__ in metrics_segmentation.keys():
                        metrics_segmentation[test_metric_segmentation.__class__.__name__].append(metric)
                    else:
                        metrics_segmentation[test_metric_segmentation.__class__.__name__] = [metric]
            # Plot
            object_classes = class_predictions[0].argmax(dim=-1).cpu().detach()
            # Case the no objects are detected
            if object_classes.shape[0] > 0:
                object_indexes = torch.from_numpy(np.argwhere(object_classes.numpy() > 0)[:, 0])
                bounding_box_predictions = misc.relative_bounding_box_to_absolute(
                    misc.bounding_box_xcycwh_to_x0y0x1y1(
                        bounding_box_predictions[0, object_indexes].cpu().clone().detach()), height=input.shape[-2],
                    width=input.shape[-1])
                misc.plot_instance_segmentation_overlay_instances_bb_classes(image=input[0],
                                                                             instances=(instance_predictions[0][
                                                                                            object_indexes] > 0.5).float(),
                                                                             bounding_boxes=bounding_box_predictions,
                                                                             class_labels=object_classes[
                                                                                 object_indexes],
                                                                             show=False, save=True,
                                                                             file_path=os.path.join(
                                                                                 self.path_save_plots,
                                                                                 "test_plot_{}_is_bb_c.png".format(
                                                                                     index)))
                misc.plot_instance_segmentation_overlay_instances_bb_classes(image=input[0],
                                                                             instances=(instance_predictions[0][
                                                                                            object_indexes] > 0.5).float(),
                                                                             bounding_boxes=bounding_box_predictions,
                                                                             class_labels=object_classes[
                                                                                 object_indexes],
                                                                             show=False, save=True,
                                                                             show_class_label=False,
                                                                             file_path=os.path.join(
                                                                                 self.path_save_plots,
                                                                                 "test_plot_{}_is_bb.png".format(
                                                                                     index)))
                misc.plot_instance_segmentation_overlay_instances(image=input[0],
                                                                  instances=(instance_predictions[0][
                                                                                 object_indexes] > 0.5).float(),
                                                                  class_labels=object_classes[object_indexes],
                                                                  show=False, save=True,
                                                                  file_path=os.path.join(
                                                                      self.path_save_plots,
                                                                      "test_plot_{}_is.png".format(index)))
                misc.plot_instance_segmentation_overlay_bb_classes(image=input[0],
                                                                   bounding_boxes=bounding_box_predictions,
                                                                   class_labels=object_classes[
                                                                       object_indexes],
                                                                   show=False, save=True,
                                                                   file_path=os.path.join(
                                                                       self.path_save_plots,
                                                                       "test_plot_{}_bb_c.png".format(
                                                                           index)))
                misc.plot_instance_segmentation_labels(
                    instances=(instance_predictions[0][object_indexes] > 0.5).float(),
                    bounding_boxes=bounding_box_predictions,
                    class_labels=object_classes[object_indexes], show=False, save=True,
                    file_path=os.path.join(self.path_save_plots, "test_plot_{}_bb_no_overlay_.png".format(index)),
                    show_class_label=False, white_background=True)
                misc.plot_instance_segmentation_map_label(
                    instances=(instance_predictions[0][object_indexes] > 0.5).float(),
                    class_labels=object_classes[object_indexes], show=False, save=True,
                    file_path=os.path.join(self.path_save_plots, "test_plot_{}_no_overlay.png".format(index)),
                    white_background=True)
        # Average metrics and save them in logs
        for metric_name in metrics_classification:
            print(metric_name + "_classification_test=", float(np.mean(metrics_classification[metric_name])))
            self.logger.log(metric_name=metric_name + "_classification_test",
                            value=float(np.mean(metrics_classification[metric_name])))
        for metric_name in metrics_bounding_box:
            print(metric_name + "_bounding_box_test=", float(np.mean(metrics_bounding_box[metric_name])))
            self.logger.log(metric_name=metric_name + "_bounding_box_test",
                            value=float(np.mean(metrics_bounding_box[metric_name])))
        for metric_name in metrics_segmentation:
            metric_values = np.array(metrics_segmentation[metric_name])
            print(metric_name + "_segmentation_test=", float(np.mean(metric_values[~np.isnan(metric_values)])))
            self.logger.log(metric_name=metric_name + "_segmentation_test",
                            value=float(np.mean(metric_values[~np.isnan(metric_values)])))
        # Save metrics
        self.logger.save_metrics(path=self.path_save_metrics)
