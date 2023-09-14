from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pix2tex.dataset.dataset import Im2LatexDataset
import os
import argparse
import logging
import yaml

import torch
from munch import Munch
from tqdm.auto import tqdm
import wandb
import torch.nn as nn
from pix2tex.eval import evaluate
from pix2tex.models import get_model
from pix2tex.utils import in_model_path, parse_args, seed_everything, get_optimizer, get_scheduler, gpu_memory_check
import pprint

#* ===== for EVAL parts =====
from torchtext.data import metrics

from pix2tex.eval import detokenize, alternatives, token2str, post_process, distance
import numpy as np

#* ^^^^^ for EVAL parts ^^^^^

import lightning.pytorch as pl


class LatexOCRPL(pl.LightningModule):
    def __init__(self, pt_model, microbatch, optimizer, scheduler, args, tokenizer):
        super().__init__()
        self.pt_model = pt_model
        self.microbatch = microbatch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.automatic_optimization = False

        self.args = args
        self.tokenizer = tokenizer
        self.batchsize = self.args.batchsize

    def forward(self, *args, **kwargs):
        return self.pt_model(*args, **kwargs)

    #* ===== TRAIN =====

    def on_train_epoch_start(self):
        return

    def on_train_epoch_end(self):
        self.lr_schedulers().step()  # * need due to no auto opt
        return

    def on_train_start(self):
        print()
        print("fit start from epoch", self.current_epoch)
        print("stop right after epoch", self.args.middlestop)
        assert self.args.middlestop < 0 or self.current_epoch < self.args.middlestop

    def on_train_end(self):
        return

    def training_step(self, batch, batch_idx):
        total_losses = []

        for dl_idx in range(len(batch)):
            sub_batch = batch[dl_idx]
            seq, im = sub_batch
            opt = self.optimizers()

            opt.zero_grad()
            total_loss = 0
            for j in range(0, len(im), self.microbatch):
                tgt_seq, tgt_mask = seq['input_ids'][j:j +
                                                     self.microbatch], seq['attention_mask'][j:j+self.microbatch].bool()
                loss = self(im[j:j+self.microbatch], tgt_seq=tgt_seq, mask=tgt_mask) * \
                    self.microbatch/self.batchsize
                self.manual_backward(loss)
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.pt_model.parameters(), 1)
            opt.step()

            self.log(f"train_loss_{dl_idx}", total_loss, on_epoch=True)
            total_losses.append(total_loss)
        self.log(f"train_loss_avg", sum(total_losses) /
                 len(total_losses), on_epoch=True)

    #* ===== VALID =====

    def on_validation_epoch_start(self):
        vlen = len(self.trainer.val_dataloaders)
        self.bleus, self.edit_dists, self.token_acc = [[] for _ in range(
            vlen)], [[] for _ in range(vlen)], [[] for _ in range(vlen)]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        seq, im = batch
        # only if wrapped into torch dataloader
        # seq = {k: v.squeeze(0) for k,v in seq.items()}
        # im = im.squeeze(0)

        dec = self.pt_model.generate(
            im, temperature=self.args.get('temperature', .2))
        pred = detokenize(dec, self.tokenizer)
        truth = detokenize(seq['input_ids'], self.tokenizer)
        self.bleus[dataloader_idx].append(
            metrics.bleu_score(pred, [alternatives(x) for x in truth]))

        for predi, truthi in zip(token2str(dec, self.tokenizer), token2str(seq['input_ids'], self.tokenizer)):
            ts = post_process(truthi)
            if len(ts) > 0:
                self.edit_dists[dataloader_idx].append(
                    distance(post_process(predi), ts)/len(ts))
                    
        tgt_seq = seq['input_ids'][:, 1:]
        shape_diff = dec.shape[1]-tgt_seq.shape[1]
        if shape_diff < 0:
            dec = torch.nn.functional.pad(
                dec, (0, -shape_diff), "constant", args.pad_token)
        elif shape_diff > 0:
            tgt_seq = torch.nn.functional.pad(
                tgt_seq, (0, shape_diff), "constant", args.pad_token)

        mask = torch.logical_or(
            tgt_seq != args.pad_token, dec != args.pad_token)
        tok_acc = (dec == tgt_seq)[mask].float().mean().item()
        self.token_acc[dataloader_idx].append(tok_acc)

    def on_validation_epoch_end(self):
        metric_dict = dict()
        for i in range(len(self.trainer.val_dataloaders)):
            metric_dict[f"val_bleu_{i}"] = np.mean(self.bleus[i])
            metric_dict[f"val_ed_{i}"] = np.mean(self.edit_dists[i])
            metric_dict[f"val_acc_{i}"] = np.mean(self.token_acc[i])

        self.log_dict(metric_dict)
        print()
        print("Epoch", self.current_epoch, "validation scores:", metric_dict)

    #* ===== TEST (not now) =====

    #* ===== OTHERS =====

    def configure_optimizers(self):
        return {"optimizer": self.optimizer,  "lr_scheduler": self.scheduler}


class MiddleStop(Callback):
    """stop right after ep stop_at, 0-index, if stop_at is negative then nothing happens"""

    def __init__(self, stop_at):
        super().__init__()
        self.stop_at = stop_at
        self.verbose = False

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._run_early_stopping_check(trainer, pl_module)

    def _run_early_stopping_check(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        should_stop = (self.stop_at >= 0 and pl_module.current_epoch >= self.stop_at)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch


class SaveLast(Callback):
    def __init__(self):
        super().__init__()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        save_path = os.path.join(
            trainer.logger.log_dir, "checkpoints", f"ep{pl_module.current_epoch-1}.pth")
        save_path = os.path.abspath(save_path)
        torch.save(pl_module.pt_model.state_dict(), save_path)
        print("last model saved in .pth format")


# class IterDatasetWrapper(torch.utils.data.IterableDataset):
#     def __init__(self, ds):
#         self.ds = ds

#     def __iter__(self):
#         iter(self.ds)
#         return self.ds

#     def __next__(self):
#         return next(self.ds)

#     def __len__(self):
#         return len(self.ds)


def get_pl_trainer(device, name, epochs, val_batches, middlestop_ep, out_path=None):
    # set up device in pt-lightning
    from functools import partial
    if device.find("cuda:") != -1:
        _get_fn = partial(pl.Trainer, accelerator="gpu", devices=[
                          int(x) for x in device.split("cuda:")[1].split(",")])
    else:
        _get_fn = partial(pl.Trainer, accelerator="cpu")

    # start a new logger, i.e. new experiment version, even in continuing training
    if out_path is None:
        out_path = "./lightning_logs"
    print("out path =", os.path.abspath(out_path))
    logger = TensorBoardLogger(out_path, name=name, version=args.version)

    # log freq of tboard, not very important
    log_freq = args.log_every

    return _get_fn(limit_train_batches=None, max_epochs=epochs,
                   logger=logger, log_every_n_steps=log_freq,
                   check_val_every_n_epoch=1,
                   callbacks=[MiddleStop(middlestop_ep),
                              ModelCheckpoint(
                                  dirpath=None, save_top_k=-1, monitor="train_loss_avg"),
                              TQDMProgressBar(),
                              SaveLast()
                              ],
                   limit_val_batches=val_batches,
                   precision="16-mixed"
                   )

#* ===== MAIN LOOP =====


def train(args):

    dl_trains = [Im2LatexDataset().load(x) for x in args.data]
    dl_train_combined = dl_trains[0]
    for i in range(1, len(dl_trains)):
        dl_train_combined.combine(dl_trains[i])
    dl_train_combined.update(**args, test=False)
    dl_train_combined._get_ext_size()

    valdataloaders = [Im2LatexDataset().load(x) for x in args.valdata]
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize,
                   keep_smaller_batches=True, test=True)
    for v in valdataloaders:
        v.update(**valargs)
        v._get_ext_size()
    # valdataloaders = [torch.utils.data.DataLoader(IterDatasetWrapper(v), batch_size=1,
        # sampler = torch.utils.data.sampler.SequentialSampler(IterDatasetWrapper(v))) for v in valdataloaders]

    device = args.device
    model = get_model(args)
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_memory_check(model, args)

    # the name would be handle in pt-lightning trainer
    # out_path = os.path.join(args.output_path, args.name)
    out_path = args.output_path
    os.makedirs(out_path, exist_ok=True)

    # load pth model, would be overridden if pl_chpt is supplied
    if args.load_chkpt is not None:
        model.load_state_dict(torch.load(args.load_chkpt, map_location=device))

    opt = get_optimizer(args.optimizer)(
        model.parameters(), args.lr, betas=args.betas)
    scheduler = get_scheduler(args.scheduler)(
        opt, step_size=args.lr_step, gamma=args.gamma)

    microbatch = args.get('micro_batchsize', -1)
    if microbatch == -1:
        microbatch = args.batchsize

    pl.seed_everything(args.seed, workers=True)

    pl_model = LatexOCRPL(model, microbatch=microbatch,
                          optimizer=opt, scheduler=scheduler,
                          args=args, tokenizer=dl_train_combined.tokenizer)
    pl_trainer = get_pl_trainer(device=args.device,
                                name=args.name, epochs=args.epochs,
                                val_batches=args.valbatches,  # now a fixed number
                                middlestop_ep=args.middlestop,
                                out_path=out_path
                                )

    pl_trainer.fit(model=pl_model, train_dataloaders=[dl_train_combined],
                   val_dataloaders=valdataloaders, ckpt_path=args.load_pl_chpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', default=None,
                        help='path to yaml config file', type=str)

    parsed_args = parser.parse_args()
    assert parsed_args.config is not None
    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # pprint.pprint(parsed_args)

    args = parse_args(Munch(params), **vars(parsed_args))
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.WARNING)
    seed_everything(args.seed)

    args.wandb = False
    train(args)
