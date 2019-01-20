"""Training WaveRNN Model.

usage: train.py [options] <data-root>

options:
    --checkpoint-dir=<dir>      Directory where to save model checkpoints [default: checkpoints].
    --checkpoint=<path>         Restore model from checkpoint path if given.
    --log-event-path=<path>     Path to tensorboard event log
    -h, --help                  Show this help message and exit
"""
import os

import librosa
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from docopt import docopt
from os.path import join
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import raw_collate, discrete_collate, AudiobookDataset
from distributions import *
from hparams import hparams as hp
from loss_function import nll_loss
from lrschedule import noam_learning_rate_decay, step_learning_rate_decay
from model import build_model
from utils import num_params_count

global_step = 0
global_epoch = 0
global_test_step = 0
use_cuda = torch.cuda.is_available()


def np_now(tensor):
    return tensor.detach().cpu().numpy()


def clamp(x, lo=0, hi=1):
    return max(lo, min(hi, x))


class PruneMask():
    def __init__(self, layer, prune_rnn_input):
        self.mask = []
        self.p_idx = [0]
        self.total_params = 0
        self.pruned_params = 0
        self.split_size = 0
        self.init_mask(layer, prune_rnn_input)

    def init_mask(self, layer, prune_rnn_input):
        # Determine the layer type and
        # num matrix splits if rnn
        layer_type = str(layer).split('(')[0]
        splits = {'Linear': 1, 'GRU': 3, 'LSTM': 4}

        # Organise the num and indices of layer parameters
        # Dense will have one index and rnns two (if pruning input)
        if layer_type != 'Linear':
            self.p_idx = [0, 1] if prune_rnn_input else [1]

        # Get list of parameters from layers
        params = self.get_params(layer)

        # For each param matrix in this layer, create a mask
        for W in params:
            self.mask += [torch.ones_like(W)]
            self.total_params += W.size(0) * W.size(1)

        # Need a split size for mask_from_matrix() later on
        self.split_size = self.mask[0].size(0) // splits[layer_type]

    def get_params(self, layer):
        params = []
        for idx in self.p_idx:
            params += [list(layer.parameters())[idx].data]
        return params

    def update_mask(self, layer, z):
        params = self.get_params(layer)
        for i, W in enumerate(params):
            self.mask[i] = self.mask_from_matrix(W, z)
        self.update_prune_count()

    def apply_mask(self, layer):
        params = self.get_params(layer)
        for M, W in zip(self.mask, params): W *= M

    def mask_from_matrix(self, W, z):
        # Split into gate matrices (or not)
        W_split = torch.split(W, self.split_size)

        M = []
        # Loop through splits
        for W in W_split:
            # Sort the magnitudes
            N = W.shape[0]
            W = np.transpose(W, (1, 0))
            W_abs = torch.abs(W)
            L = np.reshape(W_abs, (-1, N // hp.sparse_group, hp.sparse_group))
            S = L.sum(dim=-1)
            sorted_abs, _ = torch.sort(S.view(-1))

            # Pick k (num weights to zero)
            k = int(W.size(0) * W.size(1) // hp.sparse_group * z)
            threshold = sorted_abs[k]
            mask = (S >= threshold).float()
            mask = np.repeat(mask, hp.sparse_group, axis=1)
            mask = np.transpose(mask, (1, 0))
            # Create the mask
            M += [mask]

        return torch.cat(M)

    def update_prune_count(self):
        self.pruned_params = 0
        for M in self.mask:
            self.pruned_params += int(np_now((M - 1).sum() * -1))


class Pruner(object):
    def __init__(self, layers, start_prune, prune_steps, target_sparsity,
                 prune_rnn_input=True):
        self.z = 0  # Objects sparsity @ time t
        self.t_0 = start_prune
        self.S = prune_steps
        self.Z = target_sparsity
        self.num_pruned = 0
        self.total_params = 0
        self.masks = []
        self.layers = layers
        for layer in layers:
            self.masks += [PruneMask(layer, prune_rnn_input)]
        self.count_total_params()

    def update_sparsity(self, t):
        z = self.Z * (1 - (1 - (t - self.t_0) / self.S) ** 3)
        self.z = clamp(z, 0, self.Z)
        return

    def prune(self, step):
        self.update_sparsity(step)
        for (l, m) in zip(self.layers, self.masks):
            m.update_mask(l, self.z)
            m.apply_mask(l)
        return self.count_num_pruned()

    def restart(self, layers, step):
        # In case training is stopped
        self.update_sparsity(step)
        for (l, m) in zip(layers, self.masks):
            m.update_mask(l, self.z)

    def count_num_pruned(self):
        self.num_pruned = 0
        for m in self.masks:
            self.num_pruned += m.pruned_params
        return self.num_pruned

    def count_total_params(self):
        for m in self.masks:
            self.total_params += m.total_params


def save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(step))
    optimizer_state = optimizer.state_dict()
    global global_test_step
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch
    global global_test_step

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    global_test_step = checkpoint.get("global_test_step", 0)

    return model


def test_save_checkpoint():
    checkpoint_path = "checkpoints/"
    device = torch.device("cuda" if use_cuda else "cpu")
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    global global_step, global_epoch, global_test_step
    save_checkpoint(device, model, optimizer, global_step, checkpoint_path, global_epoch)

    model = load_checkpoint(checkpoint_path + "checkpoint_step000000000.pth", model, optimizer, False)


def evaluate_model(model, data_loader, checkpoint_dir, limit_eval_to=5):
    """evaluate model and save generated wav and plot

    """
    test_path = data_loader.dataset.test_path
    test_files = os.listdir(test_path)
    counter = 0
    output_dir = os.path.join(checkpoint_dir, 'eval')
    for f in test_files:
        if f[-7:] == "mel.npy":
            mel = np.load(os.path.join(test_path, f))
            wav = model.generate(mel, batched=True)
            # save wav
            wav_path = os.path.join(output_dir, "checkpoint_step{:09d}_wav_{}.wav".format(global_step, counter))
            librosa.output.write_wav(wav_path, wav, sr=hp.sample_rate)
            # save wav plot
            fig_path = os.path.join(output_dir, "checkpoint_step{:09d}_wav_{}.png".format(global_step, counter))
            fig = plt.plot(wav.reshape(-1))
            plt.savefig(fig_path)
            # clear fig to drawing to the same plot
            plt.clf()

            if counter == 0:
                wav = model.generate(mel, batched=False)
                # save wav
                wav_path = os.path.join(output_dir,
                                        "checkpoint_step{:09d}_wav_unbatched_{}.wav".format(global_step, counter))
                librosa.output.write_wav(wav_path, wav, sr=hp.sample_rate)
                # save wav plot
                fig_path = os.path.join(output_dir,
                                        "checkpoint_step{:09d}_wav_unbatched_{}.png".format(global_step, counter))
                fig = plt.plot(wav.reshape(-1))
                plt.savefig(fig_path)
                # clear fig to drawing to the same plot
                plt.clf()

            counter += 1

        # stop evaluation early via limit_eval_to
        if counter >= limit_eval_to:
            break


def train_loop(device, model, data_loader, optimizer, checkpoint_dir):
    """Main training loop.

    """
    # create loss and put on device
    if hp.input_type == 'raw':
        if hp.distribution == 'beta':
            criterion = beta_mle_loss
        elif hp.distribution == 'gaussian':
            criterion = gaussian_loss
    elif hp.input_type == 'mixture':
        criterion = discretized_mix_logistic_loss
    elif hp.input_type in ["bits", "mulaw"]:
        criterion = nll_loss
    else:
        raise ValueError("input_type:{} not supported".format(hp.input_type))

    # Pruner for reducing memory footprint
    layers = [model.rnn1, model.rnn2]
    pruner = Pruner(layers, hp.start_prune, hp.prune_steps, hp.sparsity_target)

    global global_step, global_epoch, global_test_step
    while global_epoch < hp.nepochs:
        running_loss = 0
        for i, (x, m, y) in enumerate(tqdm(data_loader)):
            x, m, y = x.to(device), m.to(device), y.to(device)
            y_hat = model(x, m)
            y = y.unsqueeze(-1)
            loss = criterion(y_hat, y)
            # calculate learning rate and update learning rate
            if hp.fix_learning_rate:
                current_lr = hp.fix_learning_rate
            elif hp.lr_schedule_type == 'step':
                current_lr = step_learning_rate_decay(hp.initial_learning_rate, global_step, hp.step_gamma,
                                                      hp.lr_step_interval)
            else:
                current_lr = noam_learning_rate_decay(hp.initial_learning_rate, global_step, hp.noam_warm_up_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            optimizer.zero_grad()
            loss.backward()
            # clip gradient norm
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), hp.grad_norm)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            writer.add_scalar("loss", float(loss.item()), global_step)
            writer.add_scalar("avg_loss", float(avg_loss), global_step)
            writer.add_scalar("learning_rate", float(current_lr), global_step)
            writer.add_scalar("grad_norm", float(grad_norm), global_step)

            # saving checkpoint if needed
            if global_step != 0 and global_step % hp.save_every_step == 0:
                pruner.prune(global_step)
                save_checkpoint(device, model, optimizer, global_step, checkpoint_dir, global_epoch)
            # evaluate model if needed
            if global_step != 0 and global_test_step != True and global_step % hp.evaluate_every_step == 0:
                pruner.prune(global_step)
                print("step {}, evaluating model: generating wav from mel...".format(global_step))
                evaluate_model(model, data_loader, checkpoint_dir)
                print("evaluation finished, resuming training...")

            # reset global_test_step status after evaluation
            if global_test_step is True:
                global_test_step = False
            global_step += 1

        num_pruned=pruner.prune(global_step)
        print("epoch:{}, running loss:{}, average loss:{}, current lr:{}, num_pruned:{}".format(global_epoch, running_loss, avg_loss,
                                                                                 current_lr, num_pruned))
        global_epoch += 1


def test_prune(model):
    layers = [model.rnn1, model.rnn2]
    start_prune = 0
    prune_steps = 100  # 20000
    sparsity_target = 0.9375
    pruner = Pruner(layers, start_prune, prune_steps, sparsity_target)

    for i in range(100):
        n_pruned = pruner.prune(100)
        print(f'{i}: {n_pruned}')

    return layers


if __name__ == "__main__":
    args = docopt(__doc__)
    # print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    data_root = args["<data-root>"]
    log_event_path = args["--log-event-path"]

    # make dirs, load dataloader and set up device
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, 'eval'), exist_ok=True)
    dataset = AudiobookDataset(data_root)
    if hp.input_type == 'raw':
        collate_fn = raw_collate
    elif hp.input_type == 'mixture':
        collate_fn = raw_collate
    elif hp.input_type in ['bits', 'mulaw']:
        collate_fn = discrete_collate
    else:
        raise ValueError("input_type:{} not supported".format(hp.input_type))
    data_loader = DataLoader(dataset, collate_fn=collate_fn, shuffle=True, num_workers=int(hp.num_workers),
                             batch_size=hp.batch_size)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device:{}".format(device))

    if log_event_path is None:
        log_event_path = "log/log_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        log_event_path += "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    print("Tensorboard event path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)

    # build model, create optimizer
    model = build_model().to(device)
    print("Parameter Count:")
    print("I: %.3f million" % (num_params_count(model.I)))
    print("Upsample: %.3f million" % (num_params_count(model.upsample)))
    print("rnn1: %.3f million" % (num_params_count(model.rnn1)))
    print("rnn2: %.3f million" % (num_params_count(model.rnn2)))
    print("fc1: %.3f million" % (num_params_count(model.fc1)))
    print("fc2: %.3f million" % (num_params_count(model.fc2)))
    print("fc3: %.3f million" % (num_params_count(model.fc3)))
    print(model)

    optimizer = optim.Adam(model.parameters(),
                           lr=hp.initial_learning_rate, betas=(
            hp.adam_beta1, hp.adam_beta2),
                           eps=hp.adam_eps, weight_decay=hp.weight_decay,
                           amsgrad=hp.amsgrad)

    if hp.fix_learning_rate:
        print("using fixed learning rate of :{}".format(hp.fix_learning_rate))
    elif hp.lr_schedule_type == 'step':
        print("using exponential learning rate decay")
    elif hp.lr_schedule_type == 'noam':
        print("using noam learning rate decay")

    # load checkpoint
    if checkpoint_path is None:
        print("no checkpoint specified as --checkpoint argument, creating new model...")
    else:
        model = load_checkpoint(checkpoint_path, model, optimizer, False)
        print("loading model from checkpoint:{}".format(checkpoint_path))
        # set global_test_step to True so we don't evaluate right when we load in the model
        global_test_step = True
        hp.start_prune = 0 #start pruning right away if loaded from checkpoint

    # main train loop
    try:
        train_loop(device, model, data_loader, optimizer, checkpoint_dir)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    finally:
        print("saving model....")
        save_checkpoint(device, model, optimizer, global_step, checkpoint_dir, global_epoch)


def test_eval():
    data_root = "data_dir"
    dataset = AudiobookDataset(data_root)
    if hp.input_type == 'raw':
        collate_fn = raw_collate
    elif hp.input_type == 'bits':
        collate_fn = discrete_collate
    else:
        raise ValueError("input_type:{} not supported".format(hp.input_type))
    data_loader = DataLoader(dataset, collate_fn=collate_fn, shuffle=True, num_workers=0, batch_size=hp.batch_size)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device:{}".format(device))

    # build model, create optimizer
    model = build_model().to(device)

    evaluate_model(model, data_loader)
