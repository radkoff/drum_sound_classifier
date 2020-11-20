import shutil
import logging
import os
from pathlib import Path

from torch.optim import SGD
import torch.nn.functional as F
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

def set_handler(handler):
    logger.addHandler(handler)

# To resume training of an existing model
# Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
def _load_checkpoint(model, optimizer, experiment_name):
    filename = f'model_latest_{experiment_name}.pt'
    logger.info(f'Loading previous model state {filename}')
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        start_best_metric = checkpoint[('best_accuracy')]
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        raise ValueError('No checkpoint found')

    return model, optimizer, start_epoch, start_best_metric


def run_classification(model, train_loader, val_loader, epochs, early_stopping, lr, momentum, log_interval, experiment_name, continueing=False):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    writer = SummaryWriter(here/f'tensorboard/runs_{experiment_name}')
    data_loader_iter = iter(train_loader)
    x, y = next(data_loader_iter)
    writer.add_graph(model, x)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    start_best_accuracy = 0.0
    if continueing:
        model, optimizer, start_epoch, start_best_accuracy = _load_checkpoint(model, optimizer, experiment_name)
        model.train()   # In case the model was saved after a test loop where model.eval() was called

    evaluator = create_supervised_evaluator(model, device=device,
                                            metrics={'accuracy': Accuracy(), 'nll': Loss(F.nll_loss)})
    evaluator_val = create_supervised_evaluator(model, device=device,
                                                metrics={'accuracy': Accuracy(), 'nll': Loss(F.nll_loss)})
    trainer = create_supervised_trainer(model, optimizer, nn.NLLLoss(), device=device)

    desc = 'ITERATION - loss: {:.4f}'
    progress_bar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.STARTED)
    def init(engine):
        engine.state.epoch = start_epoch
        engine.state.best_accuracy = start_best_accuracy

    # One iteration = one batch
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            progress_bar.desc = desc.format(engine.state.output)
            progress_bar.update(log_interval)
            writer.add_scalar('training/loss', engine.state.output, engine.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_gradients(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    writer.add_scalar(f'{n}/gradient', p.grad.abs().mean(), engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        progress_bar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        logger.info(
            'Training Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}'
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )
        writer.add_scalar('training/avg_loss', avg_nll, engine.state.epoch)
        writer.add_scalar('training/avg_accuracy', avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results_and_save(engine):
        evaluator_val.run(val_loader)
        metrics = evaluator_val.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        logger.info(
            'Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}'.format(engine.state.epoch, avg_accuracy, avg_nll))

        progress_bar.n = progress_bar.last_print_n = 0
        writer.add_scalar('valdation/avg_loss', avg_nll, engine.state.epoch)
        writer.add_scalar('valdation/avg_accuracy', avg_accuracy, engine.state.epoch)

        # Save the model every epoch. If it's the best seen so far, save it separately
        torch.save({
            'epoch': engine.state.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': avg_accuracy,
            'best_accuracy': engine.state.best_accuracy,
            'loss': avg_nll
        }, f'model_latest_{experiment_name}.pt')
        if avg_accuracy > engine.state.best_accuracy:
            engine.state.best_accuracy = avg_accuracy
            shutil.copyfile(f'model_latest_{experiment_name}.pt', f'model_best_{experiment_name}.pt')

    # Early stopping
    handler = EarlyStopping(patience=early_stopping, score_function=(lambda engine: -evaluator_val.state.metrics['nll']), trainer=trainer)
    evaluator_val.add_event_handler(Events.COMPLETED, handler)

    trainer.run(train_loader, max_epochs=epochs)
    progress_bar.close()
    writer.close()
