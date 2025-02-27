from typing import Dict, Optional, Sequence, Union

import pytorch_metric_learning
import torch
from pyannote.audio.tasks import SpeakerDiarization, SpeakerEmbedding
from pyannote.audio.utils.permutation import permutate
from pyannote.database import Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric


class EmbeddingKnowledgeDistillationTask(SpeakerEmbedding):
    def __init__(
        self,
        protocol: Protocol,
        min_duration: Optional[float] = None,
        duration: float = 2.0,
        num_classes_per_batch: int = 32,
        num_chunks_per_class: int = 1,
        margin: float = 28.6,
        scale: float = 64.0,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        augmentation: Optional[BaseWaveformTransform] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        teacher: SpeakerEmbedding = None,
        kd_regularization: float = 1.0,
    ):
        super().__init__(
            protocol,
            min_duration,
            duration,
            num_classes_per_batch,
            num_chunks_per_class,
            margin,
            scale,
            num_workers,
            pin_memory,
            augmentation,
            metric,
        )

        self.teacher = teacher
        self.kd_regularization = kd_regularization

    def training_step(self, batch, batch_idx: int):
        X, y = batch["X"], batch["y"]
        prediction = self.model(X)
        loss = self.model.loss_func(prediction, y)

        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None

        # Add the distillation loss
        with torch.no_grad():
            teacher_output = self.teacher(X)

            # Calculate KL divergence between student and teacher predictions
            distillation_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(prediction, dim=1),
                torch.nn.functional.softmax(teacher_output, dim=1),
                reduction="batchmean",
            )

        loss += self.kd_regularization * distillation_loss

        self.model.log(
            "loss/train",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return {"loss": loss}


class SegmentationKnowledgeDistillationTask(SpeakerDiarization):
    def __init__(
        self,
        protocol,
        cache=None,
        duration=2,
        max_speakers_per_chunk=None,
        max_speakers_per_frame=None,
        weigh_by_cardinality=False,
        warm_up=0,
        balance=None,
        weight=None,
        batch_size=32,
        num_workers=None,
        pin_memory=False,
        augmentation=None,
        vad_loss=None,
        metric=None,
        max_num_speakers=None,
        loss=None,
        teacher=None,
        kd_regularization: float = 1.0,
    ):
        super().__init__(
            protocol,
            cache,
            duration,
            max_speakers_per_chunk,
            max_speakers_per_frame,
            weigh_by_cardinality,
            warm_up,
            balance,
            weight,
            batch_size,
            num_workers,
            pin_memory,
            augmentation,
            vad_loss,
            metric,
            max_num_speakers,
            loss,
        )

        self.teacher = teacher
        self.kd_regularization = kd_regularization

    def training_step(self, batch, batch_idx):
        # target
        target = batch["y"]
        # (batch_size, num_frames, num_speakers)

        waveform = batch["X"]
        # (batch_size, num_channels, num_samples)

        # drop samples that contain too many speakers
        num_speakers: torch.Tensor = torch.sum(torch.any(target, dim=1), dim=1)
        keep: torch.Tensor = num_speakers <= self.max_speakers_per_chunk
        target = target[keep]
        waveform = waveform[keep]

        # corner case
        if not keep.any():
            return None

        # forward pass
        prediction = self.model(waveform)
        batch_size, num_frames, _ = prediction.shape
        # (batch_size, num_frames, num_classes)

        # frames weight
        weight_key = getattr(self, "weight", None)
        weight = batch.get(
            weight_key,
            torch.ones(batch_size, num_frames, 1, device=self.model.device),
        )
        # (batch_size, num_frames, 1)

        # warm-up
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        weight[:, :warm_up_left] = 0.0
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        weight[:, num_frames - warm_up_right :] = 0.0

        if self.specifications.powerset:
            multilabel = self.model.powerset.to_multilabel(prediction)
            permutated_target, _ = permutate(multilabel, target)
            permutated_target_powerset = self.model.powerset.to_powerset(
                permutated_target.float()
            )
            seg_loss = self.segmentation_loss(
                prediction, permutated_target_powerset, weight=weight
            )

        else:
            permutated_prediction, _ = permutate(target, prediction)
            seg_loss = self.segmentation_loss(
                permutated_prediction, target, weight=weight
            )

        self.model.log(
            "loss/train/segmentation",
            seg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.vad_loss is None:
            vad_loss = 0.0

        else:
            # TODO: vad_loss probably does not make sense in powerset mode
            # because first class (empty set of labels) does exactly this...
            if self.specifications.powerset:
                vad_loss = self.voice_activity_detection_loss(
                    prediction, permutated_target_powerset, weight=weight
                )

            else:
                vad_loss = self.voice_activity_detection_loss(
                    permutated_prediction, target, weight=weight
                )

            self.model.log(
                "loss/train/vad",
                vad_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        loss = seg_loss + vad_loss

        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None

        # Knowledge Distillation - compute teacher loss
        with torch.no_grad():
            # Compute KL divergence between student and teacher predictions
            teacher_prediction = self.teacher(waveform)
            teacher_prediction = teacher_prediction[keep]
            teacher_prediction = teacher_prediction[:, :num_frames, :]
            loss_kd = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(prediction, dim=2),
                torch.nn.functional.softmax(teacher_prediction, dim=2),
                reduction="batchmean",
            )
            loss += self.kd_regularization * loss_kd

        self.model.log(
            "loss/train",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return loss
