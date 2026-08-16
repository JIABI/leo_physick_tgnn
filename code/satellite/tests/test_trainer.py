import torch
from torch.utils.data import DataLoader

from leo_pg.data.collate import collate_episode
from leo_pg.train.trainer import Trainer


class _TwoStepModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward_episode(self, episode, device):
        predictions = [self.value.reshape(1, 1) for _ in episode["steps"]]
        targets = [step["y"].to(device) for step in episode["steps"]]
        return {"preds": predictions, "ys": targets}


def test_one_step_training_supervises_every_timestep():
    episode = {
        "steps": [
            {"y": torch.tensor([[1.0]]), "meta": {"K_users": 0}},
            {"y": torch.tensor([[0.0]]), "meta": {"K_users": 0}},
        ]
    }
    loader = DataLoader([episode], batch_size=1, collate_fn=collate_episode)
    model = _TwoStepModel()
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        lr=0.1,
        weight_decay=0.0,
        clip_grad_norm=1.0,
    )
    trainer.train_one_step(loader, epochs=1)
    assert model.value.item() > 0.0
