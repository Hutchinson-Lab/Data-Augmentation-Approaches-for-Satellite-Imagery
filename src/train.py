from time import time
import torch
from torchvision.transforms import v2
from src.datasets import satellite_dataloader
from src.mixing import sat_cutMix, sat_slideMix
from time import time


def train_model(model, cuda, dataloader, optimizer, epoch, criterion, num_classes, model_type, print_every=100,
                mixing_method=None, satcutmix_alpha=1.0, satslidemix_beta=1.0, sat_num_pairs=1):

    model.train()
    t0 = time()
    sum_loss = 0
    n_train, n_batches = len(dataloader.dataset), len(dataloader)
    print_sum_loss = 0
    idx = 0

    if mixing_method is not None:
        regression = model_type == 'regression'
        if mixing_method == 'CutMix':
            cutmix = v2.CutMix(num_classes=num_classes)
        elif mixing_method == 'Sat-CutMix':
            cutmix = sat_cutMix(num_classes, satcutmix_alpha, sat_num_pairs, regression)
        elif mixing_method == 'Sat-SlideMix':
            cutmix = sat_slideMix(num_classes, satslidemix_beta, sat_num_pairs, regression)

    for img, label, _ in dataloader:
        if cuda:
            img = img.cuda()
            label = label.cuda()

        if mixing_method is not None:
            label_pre = label
            img, label = cutmix(img, label_pre)

        optimizer.zero_grad()

        outputs = torch.squeeze(model(img)).to(torch.float64)
        if num_classes == 1:
            outputs = outputs.double()
            label = label.double()

        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        if (idx + 1) * dataloader.batch_size % print_every == 0:
            print_avg_loss = (sum_loss - print_sum_loss) / (
                print_every / dataloader.batch_size)
            print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(
                epoch, (idx + 1) * dataloader.batch_size, n_train,
                100 * (idx + 1) / n_batches, print_avg_loss))
            print_sum_loss = sum_loss
        idx += 1
    avg_loss = sum_loss / n_batches
    print('\nTrain Epoch {}: Loss {:0.4f}, Time {:0.3f}s'.format(epoch, avg_loss, time()-t0))
    return avg_loss


def validate_model(model, cuda, dataloader, epoch, criterion, num_classes, timeit=False):

    with torch.no_grad():
        model.eval()
        t0 = time()
        sum_loss = 0
        n_train, n_batches = len(dataloader.dataset), len(dataloader)

        for img, label, _ in dataloader:
            if cuda:
                img = img.cuda()
                label = label.cuda()

            if timeit:
                t_start = time()
                outputs = torch.squeeze(model(img)).to(torch.float64)
                t_end = time()
            else:
                outputs = torch.squeeze(model(img)).to(torch.float64)
            if num_classes == 1:
                outputs = outputs.double()
                label = label.double()

            loss = criterion(outputs, label)

            sum_loss += loss.item()
        avg_loss = sum_loss / n_batches
        print('Test Epoch {}: Loss {:0.4f}, Time {:0.3f}s'.format(epoch, avg_loss, time()-t0))
    if timeit:
        return avg_loss, t_end-t_start
    else:
        return avg_loss
