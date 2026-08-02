from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Client:
    client_id: str
    hidden_modality_id: int
    samples: torch.Tensor
    labels: torch.Tensor
    encoder_type: str
    pred_cluster: Optional[int] = None
    input_shape: Optional[list[int]] = None
    sequence_lengths: Optional[torch.Tensor] = None

    @classmethod
    def from_payload(cls, payload: dict, pred_cluster: Optional[int] = None):
        return cls(
            client_id=str(payload["client_id"]),
            hidden_modality_id=int(payload["hidden_modality_id"]),
            samples=payload["samples"],
            labels=payload["labels"],
            encoder_type=str(payload.get("encoder_type", "time_series")),
            pred_cluster=None if pred_cluster is None else int(pred_cluster),
            input_shape=[int(v) for v in payload.get("input_shape", list(payload["samples"].shape[1:]))],
            sequence_lengths=payload.get("sequence_lengths"),
        )

    def training_view(self):
        return {
            "client_id": self.client_id,
            "samples": self.samples,
            "labels": self.labels,
            "encoder_type": self.encoder_type,
            "pred_cluster": self.pred_cluster,
            "input_shape": self.input_shape,
            "sequence_lengths": self.sequence_lengths,
        }

    def to_payload(self):
        return {
            "client_id": self.client_id,
            "hidden_modality_id": int(self.hidden_modality_id),
            "samples": self.samples,
            "labels": self.labels,
            "encoder_type": self.encoder_type,
            "pred_cluster": self.pred_cluster,
            "input_shape": [int(v) for v in (self.input_shape or list(self.samples.shape[1:]))],
            "sequence_lengths": self.sequence_lengths,
        }
