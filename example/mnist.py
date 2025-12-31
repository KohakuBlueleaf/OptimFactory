import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms as trns

from anyschedule import AnySchedule
from tqdm.auto import tqdm

from optimfactory import (
    mup_param_group,
    mup_init,
    mup_patch_output,
    muon_param_group_split,
    compdp_param_group,
    ComboOptimizer,
    ComboLRScheduler,
)


EPOCH = 20
MODEL_CH = 16
NORM_TYPE = "instance"
BATCH_SIZE = 256
WORKERS = 4
EMA_DECAY = 0.999

BASE_DIM = 256
BASE_BS = 64
BASE_DS = 1_000_000

BASE_LR = 1e-3
BASE_EPS = 1e-6
BASE_WD = 0.1
BASE_BETA1 = 0.9
BASE_BETA2 = 0.99
USE_MUON = True
USE_MUP = False
USE_COMDP = True


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
elif torch.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float32
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.bfloat16
TRANSFORMS = trns.Compose(
    [
        trns.ToTensor(),
        trns.Normalize((0.5,), (0.5,)),
    ]
)
TRAIN_TRANSFORMS = trns.Compose(
    [
        trns.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        trns.ToTensor(),
        trns.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 3.3)),
        trns.Normalize((0.5,), (0.5,)),
    ]
)


class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def Norm2D(ch, type):
    match type:
        case ("group", groups):
            if groups >= ch:
                groups = ch
            return nn.GroupNorm(groups, ch)
        case "instance":
            return nn.GroupNorm(ch, ch)
        case "spatial-layer":
            return nn.GroupNorm(1, ch)
        case "2d-layer":
            return nn.Sequential(
                Permute((0, 2, 3, 1)),
                nn.LayerNorm(ch),
                Permute((0, 3, 1, 2)),
            )


def block(in_ch, out_ch, mid_ch):
    return nn.Sequential(
        Norm2D(in_ch, NORM_TYPE),
        nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch),
        Permute((0, 2, 3, 1)),
        nn.Linear(in_ch, mid_ch),
        nn.Mish(),
        nn.Linear(mid_ch, out_ch),
        Permute((0, 3, 1, 2)),
    )


def mlp(in_ch, out_ch, mid_ch):
    return nn.Sequential(
        nn.LayerNorm(in_ch),
        nn.Linear(in_ch, mid_ch),
        nn.Mish(),
        nn.Linear(mid_ch, out_ch),
    )


class Net(nn.Module):
    def __init__(self, num_classes=10, base_dim=32, use_mup=True):
        super(Net, self).__init__()
        self.in_proj = nn.Conv2d(1, base_dim, 3, 1, 1)
        self.block1 = block(base_dim, base_dim, base_dim * 4)  # 28x28
        self.down_proj1 = nn.Conv2d(base_dim * 4, base_dim * 2, 1)
        self.block2 = block(base_dim * 2, base_dim * 2, base_dim * 8)  # 14x14
        self.down_proj2 = nn.Conv2d(base_dim * 8, base_dim * 4, 1)
        self.block3 = block(base_dim * 4, base_dim * 4, base_dim * 16)  # 7x7

        self.mlp1 = mlp(base_dim * 4, base_dim * 4, base_dim * 16)
        self.mlp2 = mlp(base_dim * 4, base_dim * 4, base_dim * 16)
        self.out_proj = nn.Linear(base_dim * 4, num_classes)

        if use_mup:
            mup_init(self.parameters())
            mup_patch_output(self.out_proj)

    def forward(self, x):
        h = self.in_proj(x)
        h = h + self.block1(h)
        h = F.pixel_unshuffle(h, 2)
        h = self.down_proj1(h)
        h = h + self.block2(h)
        h = F.pixel_unshuffle(h, 2)
        h = self.down_proj2(h)
        h = h + self.block3(h)
        h = F.adaptive_avg_pool2d(h, 1)
        h = h.view(h.size(0), -1)
        h = h + self.mlp1(h)
        h = h + self.mlp2(h)
        h = self.out_proj(h)
        return h


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer,
    lr_scheduler=None,
    init_ema_loss: float = 0.0,
    init_decay: float = 0.0,
    ema_decay: float = 0.995,
):
    model.train()
    ema_loss = init_ema_loss
    for batch_idx, (data, target) in (
        pbar := tqdm(enumerate(train_loader), total=len(train_loader))
    ):
        data, target = data.to(device), target.to(device)
        with torch.autocast(device.type, DTYPE):
            output = model(data)
            loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if ema_loss == 0:
            ema_loss = loss.item()
        else:
            ema_decay = min(batch_idx / (1 + batch_idx), ema_decay)
            ema_decay = max(ema_decay, init_decay)
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * loss.item()

        pbar.set_postfix(
            {
                "loss": ema_loss,
            }
        )
    return ema_loss


def test(model: nn.Module, device: torch.device, test_loader: DataLoader, epoch: int):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.autocast(device.type, DTYPE):
                output = model(data)
                test_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"[Epoch {epoch}] Test set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({100. * correct / len(test_loader.dataset):.0f}%)"
    )


def main():
    device = torch.device(DEVICE)

    train_loader = DataLoader(
        MNIST(
            "./data",
            train=True,
            download=True,
            transform=TRAIN_TRANSFORMS,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=True,
        persistent_workers=WORKERS > 0,
    )
    test_loader = DataLoader(
        MNIST(
            "./data",
            train=False,
            transform=TRANSFORMS,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        persistent_workers=WORKERS > 0,
    )

    datasize = len(train_loader.dataset) * EPOCH / BASE_DS
    batchsize = BATCH_SIZE / BASE_BS

    model = Net(10, MODEL_CH, use_mup=USE_MUP or USE_COMDP).to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")
    if USE_COMDP:
        param_groups = compdp_param_group(
            model,
            BASE_LR,
            BASE_WD,
            BASE_EPS,
            BASE_BETA1,
            BASE_BETA2,
            BASE_DIM,
            1,
            1,
            batchsize,
            datasize,
            input_module=model.in_proj,
            output_module=model.out_proj,
        )
    elif USE_MUP:
        param_groups = mup_param_group(
            model.parameters(),
            base_lr=BASE_LR,
            base_dim=BASE_DIM,
            weight_decay=BASE_WD,
            weight_decay_scale=True,
            input_module=model.in_proj,
        )
    else:
        param_groups = model.parameters()
    lr_scheduler_config = {
        "lr": {
            "mode": "cosine",
            "end": len(train_loader) * EPOCH + 1,
            "warmup": 100,
            "min_value": 0.01,
        }
    }

    if USE_MUON:
        muon_group, adam_group = muon_param_group_split(param_groups, dim_threshold=16)
        optimizer = ComboOptimizer(
            [
                optim.Muon(
                    muon_group,
                    lr=BASE_LR,
                    weight_decay=BASE_WD,
                    adjust_lr_fn="match_rms_adamw",  # This provide moonlight version scaling
                ),
                optim.AdamW(
                    adam_group,
                    lr=BASE_LR,
                    betas=(BASE_BETA1, BASE_BETA2),
                    weight_decay=BASE_WD,
                ),
            ]
        )
        lr_scheduler = ComboLRScheduler(
            [
                AnySchedule(optimizer.optimizers[0], config=lr_scheduler_config),
                AnySchedule(optimizer.optimizers[1], config=lr_scheduler_config),
            ]
        )
    else:
        optimizer = optim.AdamW(
            param_groups,
            lr=BASE_LR,
            betas=(BASE_BETA1, BASE_BETA2),
            weight_decay=BASE_WD,
        )
        lr_scheduler = AnySchedule(optimizer, config=lr_scheduler_config)

    ema_loss = 0
    for epoch in range(EPOCH):
        ema_loss = train(
            model,
            device,
            train_loader,
            optimizer,
            lr_scheduler,
            init_ema_loss=ema_loss,
            init_decay=EMA_DECAY if epoch else 0,
            ema_decay=EMA_DECAY,
        )
        test(model, device, test_loader, epoch)


if __name__ == "__main__":
    main()
