import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import argparse
import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
from dataset import FaceDataset
from defaults import _C as cfg


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor_1 = AverageMeter()
    accuracy_monitor_2 = AverageMeter()
    accuracy_monitor_3 = AverageMeter()
    accuracy_monitor_4 = AverageMeter()
    accuracy_monitor_5 = AverageMeter()
    accuracy_monitor_6 = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, age, gender, race, makeup, time, happiness in _tqdm:
            y = torch.stack([age, gender, race, makeup, time, happiness], dim=1)
            x = x.to(device)
            y = y.to(device)

            # compute output
            outputs = model(x)

            # calc loss
            loss = (1/6)*(criterion(outputs[:,:101], y[:,0])+criterion(outputs[:,101:103], y[:,1])+criterion(outputs[:,103:106], y[:,2])+criterion(outputs[:,106:110], y[:,3])+criterion(outputs[:,110:112], y[:,4])+criterion(outputs[:,112:], y[:,5]))
            cur_loss = loss.item()

            # calc accuracy
            _, predicted_age = outputs[:,:101].max(1)
            _, predicted_gender = outputs[:,101:103].max(1)
            _, predicted_race = outputs[:,103:106].max(1)
            _, predicted_makeup = outputs[:,106:110].max(1)
            _, predicted_time = outputs[:,110:112].max(1)
            _, predicted_happiness = outputs[:,112:].max(1)

            
            correct_num_age = predicted_age.eq(y[:,0]).sum().item()
            correct_num_gender = predicted_gender.eq(y[:,1]).sum().item()
            correct_num_race = predicted_race.eq(y[:,2]).sum().item()
            correct_num_makeup = predicted_makeup.eq(y[:,3]).sum().item()
            correct_num_time = predicted_time.eq(y[:,4]).sum().item()
            correct_num_happiness = predicted_happiness.eq(y[:,5]).sum().item()

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)

            accuracy_monitor_1.update(correct_num_age, sample_num)
            accuracy_monitor_2.update(correct_num_gender, sample_num)
            accuracy_monitor_3.update(correct_num_race, sample_num)
            accuracy_monitor_4.update(correct_num_makeup, sample_num)
            accuracy_monitor_5.update(correct_num_time, sample_num)
            accuracy_monitor_6.update(correct_num_happiness, sample_num)


            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            _tqdm.set_postfix(OrderedDict(
                  stage="train",
                  epoch=epoch,
                  loss=loss_monitor.avg,
                  acc_age=accuracy_monitor_1.avg,
                  acc_gender=accuracy_monitor_2.avg,
                  acc_race=accuracy_monitor_3.avg,
                  acc_makeup=accuracy_monitor_4.avg,
                  acc_time=accuracy_monitor_5.avg,
                  acc_happiness=accuracy_monitor_6.avg
            ))




            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc_age=accuracy_monitor_1.avg,
                              correct_age=correct_num_age,
                              acc_gender=accuracy_monitor_2.avg,
                              correct_gender=correct_num_gender,
                              acc_race=accuracy_monitor_3.avg,
                              correct_race=correct_num_race,
                              acc_makeup=accuracy_monitor_4.avg,
                              correct_makeup=correct_num_makeup,
                              acc_time=accuracy_monitor_5.avg,
                              correct_time=correct_num_time,
                              acc_happiness=accuracy_monitor_6.avg,
                              correct_happiness=correct_num_happiness,
                              sample_num=sample_num)

    return (loss_monitor.avg, accuracy_monitor_1.avg,accuracy_monitor_2.avg,accuracy_monitor_3.avg,accuracy_monitor_4.avg,accuracy_monitor_5.avg,accuracy_monitor_6.avg)


def validate(validate_loader, model, criterion, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor_1 = AverageMeter()
    accuracy_monitor_2 = AverageMeter()
    accuracy_monitor_3 = AverageMeter()
    accuracy_monitor_4 = AverageMeter()
    accuracy_monitor_5 = AverageMeter()
    accuracy_monitor_6 = AverageMeter()

    # Initialize lists for predictions and ground truths for each category
    preds_1, gt_1 = [], []  # Age
    preds_2, gt_2 = [], []  # Gender
    preds_3, gt_3 = [], []  # Race
    preds_4, gt_4 = [], []  # Makeup
    preds_5, gt_5 = [], []  # Time
    preds_6, gt_6 = [], []  # Happiness

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, age, gender, race, makeup, time, happiness) in enumerate(_tqdm):
                y = torch.stack([age, gender, race, makeup, time, happiness], dim=1)

                x = x.to(device)
                y = y.to(device)

                # compute output
                outputs = model(x)

                # Append predictions and ground truths for each category

                # Age
                softmax_age = F.softmax(outputs[:, :101], dim=-1).cpu().numpy()
                preds_1.append(softmax_age)
                gt_1.append(y[:, 0].cpu().numpy())

                # Gender
                softmax_gender = F.softmax(outputs[:, 101:103], dim=-1).cpu().numpy()
                preds_2.append(softmax_gender)
                gt_2.append(y[:, 1].cpu().numpy())

                # Race
                softmax_race = F.softmax(outputs[:, 103:106], dim=-1).cpu().numpy()
                preds_3.append(softmax_race)
                gt_3.append(y[:, 2].cpu().numpy())

                # Makeup
                softmax_makeup = F.softmax(outputs[:, 106:110], dim=-1).cpu().numpy()
                preds_4.append(softmax_makeup)
                gt_4.append(y[:, 3].cpu().numpy())

                # Time
                softmax_time = F.softmax(outputs[:, 110:112], dim=-1).cpu().numpy()
                preds_5.append(softmax_time)
                gt_5.append(y[:, 4].cpu().numpy())

                # Happiness
                softmax_happiness = F.softmax(outputs[:, 112:], dim=-1).cpu().numpy()
                preds_6.append(softmax_happiness)
                gt_6.append(y[:, 5].cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = (1/6)*(criterion(outputs[:,:101], y[:,0])+criterion(outputs[:,101:103], y[:,1])+criterion(outputs[:,103:106], y[:,2])+criterion(outputs[:,106:110], y[:,3])+criterion(outputs[:,110:112], y[:,4])+criterion(outputs[:,112:], y[:,5]))
                    cur_loss = loss.item()

                    # calc accuracy
                    _, predicted_age = outputs[:,:101].max(1)
                    _, predicted_gender = outputs[:,101:103].max(1)
                    _, predicted_race = outputs[:,103:106].max(1)
                    _, predicted_makeup = outputs[:,106:110].max(1)
                    _, predicted_time = outputs[:,110:113].max(1)
                    _, predicted_happiness = outputs[:,113:].max(1)

                    
                    correct_num_age = predicted_age.eq(y[:,0]).sum().item()
                    correct_num_gender = predicted_gender.eq(y[:,1]).sum().item()
                    correct_num_race = predicted_race.eq(y[:,2]).sum().item()
                    correct_num_makeup = predicted_makeup.eq(y[:,3]).sum().item()
                    correct_num_time = predicted_time.eq(y[:,4]).sum().item()
                    correct_num_happiness = predicted_happiness.eq(y[:,5]).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)

                    accuracy_monitor_1.update(correct_num_age, sample_num)
                    accuracy_monitor_2.update(correct_num_gender, sample_num)
                    accuracy_monitor_3.update(correct_num_race, sample_num)
                    accuracy_monitor_4.update(correct_num_makeup, sample_num)
                    accuracy_monitor_5.update(correct_num_time, sample_num)
                    accuracy_monitor_6.update(correct_num_happiness, sample_num)

                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                  acc_age=accuracy_monitor_1.avg,
                                  correct_age=correct_num_age,
                                  acc_gender=accuracy_monitor_2.avg,
                                  correct_gender=correct_num_gender,
                                  acc_race=accuracy_monitor_3.avg,
                                  correct_race=correct_num_race,
                                  acc_makeup=accuracy_monitor_4.avg,
                                  correct_makeup=correct_num_makeup,
                                  acc_time=accuracy_monitor_5.avg,
                                  correct_time=correct_num_time,
                                  acc_happiness=accuracy_monitor_6.avg,
                                  correct_happiness=correct_num_happiness,
                                  sample_num=sample_num)

    preds_1 = np.concatenate(preds_1, axis=0)
    gt_1 = np.concatenate(gt_1, axis=0)

    preds_2 = np.concatenate(preds_2, axis=0)
    gt_2 = np.concatenate(gt_2, axis=0)

    preds_3 = np.concatenate(preds_3, axis=0)
    gt_3 = np.concatenate(gt_3, axis=0)

    preds_4 = np.concatenate(preds_4, axis=0)
    gt_4 = np.concatenate(gt_4, axis=0)

    preds_5 = np.concatenate(preds_5, axis=0)
    gt_5 = np.concatenate(gt_5, axis=0)

    preds_6 = np.concatenate(preds_6, axis=0)
    gt_6 = np.concatenate(gt_6, axis=0)

    ages = np.arange(0, 101)
    genders = np.arange(0, 2)
    races = np.arange(0, 3)
    makeups = np.arange(0, 4)
    times = np.arange(0, 2)
    happinesses = np.arange(0, 4)

    ave_preds_1 = (preds_1 * ages).sum(axis=-1)  # Age, as before
    ave_preds_2 = (preds_2 * genders).sum(axis=-1)  # Gender, for consistency in handling
    ave_preds_3 = (preds_3 * races).sum(axis=-1)  # Race
    ave_preds_4 = (preds_4 * makeups).sum(axis=-1)  # Makeup
    ave_preds_5 = (preds_5 * times).sum(axis=-1)  # Time
    ave_preds_6 = (preds_6 * happinesses).sum(axis=-1)  # Happiness

    diff_1 = ave_preds_1 - gt_1
    mae_1 = np.abs(diff_1).mean()

    diff_2 = ave_preds_2 - gt_2
    mae_2 = np.abs(diff_2).mean()

    diff_3 = ave_preds_3 - gt_3
    mae_3 = np.abs(diff_3).mean()

    diff_4 = ave_preds_4 - gt_4
    mae_4 = np.abs(diff_4).mean()

    diff_5 = ave_preds_5 - gt_5
    mae_5 = np.abs(diff_5).mean()

    diff_6 = ave_preds_6 - gt_6
    mae_6 = np.abs(diff_6).mean()


    return loss_monitor.avg, accuracy_monitor_1.avg,accuracy_monitor_2.avg,accuracy_monitor_3.avg,accuracy_monitor_4.avg,accuracy_monitor_5.avg,accuracy_monitor_6.avg, mae_1,mae_2,mae_3,mae_4,mae_5,mae_6


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH, num_classes = 116)

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    

    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae_1 = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc_1,train_acc_2,train_acc_3,train_acc_4,train_acc_5,train_acc_6 = train(train_loader, model, criterion, optimizer, epoch, device)

        # validate
        val_loss, val_acc_1, val_acc_2, val_acc_3, val_acc_4, val_acc_5, val_acc_6, val_mae_1, val_mae_2, val_mae_3, val_mae_4, val_mae_5, val_mae_6 = validate(val_loader, model, criterion, epoch, device)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc_age", train_acc_1, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc_age", val_acc_1, epoch)
            val_writer.add_scalar("mae_age", val_mae_1, epoch)
            
        # Checkpointing logic
        if val_mae_1 < best_val_mae_1:
            # Print message indicating improvement
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae_1:.3f} to {val_mae_1:.3f}")
            
            # Check if using multiple GPUs
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            
            # Save the model checkpoint
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae_1)))
            )
            # Update best_val_mae_1
            best_val_mae_1 = val_mae_1
        else:
            # Print message indicating no improvement
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae_1:.3f} ({val_mae_1:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae on Age: {best_val_mae_1:.3f}")

    print("Final Accuracies on Training Set (except Age):")
    print(f"Gender: {train_acc_2:.3f}, Race: {train_acc_3:.3f}, Makeup: {train_acc_4:.3f}, Time: {train_acc_5:.3f}, Happiness: {train_acc_6:.3f}")
    print()

    print("Final Accuracies on Validation Set (except Age):")
    print(f"Gender: {val_acc_2:.3f}, Race: {val_acc_3:.3f}, Makeup: {val_acc_4:.3f}, Time: {val_acc_5:.3f}, Happiness: {val_acc_6:.3f}")
    print()


    torch.save(model, '/content/Project-age-estimation-pytorch/trained_model.pth')




if __name__ == '__main__':
    main()
