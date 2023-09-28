# LaTeX-OCR colab reproduction with pytorch lightning

contains a training notebook and 3 training files: new annotated config, new trainer, rewritten dataset

training notebook contains the whole workflow to reproduce the model in colab

pytorch lightning to faciliate training:

- checkpointing with model and optimzer, scheduler, logs etc. (so that training can be smoothly done in several sessions)

- multiple dataloaders allowed for both training/validation set

- enable 16-bit mixed precision training easily

other modifications:

- using a callback to stop training at the middle

- saving .ckpt file at the end of each training epoch, and single .pth file at end of training session

- validation on every n steps, with fixed number/ratio of batches

- local tensorboard logging rather than wandb

final performance (30 epochs, step LR schedule with 0.8x learning rate every 10 epochs):

BLEU (↑): 0.835, ED (↓): 0.143, ACC (↑): 0.521, not as good as provided models for now


