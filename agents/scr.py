import torch
import torch.nn as nn
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from torch import softmax
from torch.utils import data

from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.buffer.buffer import Buffer
from utils.buffer.carto_update import Carto_update
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter


class SupContrastReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SupContrastReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]),
                              scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                for j in range(self.mem_iters):
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))
                        combined_batch_aug = self.transform(combined_batch)
                        features = torch.cat([self.model.forward(combined_batch).unsqueeze(1),
                                              self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                        loss = self.criterion(features, combined_labels)
                        losses.update(loss, batch_y.size(0))
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()
                if self.buffer.update_method.__class__ is not Carto_update:
                    # update mem
                    self.buffer.update(batch_x, batch_y)
                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        .format(i, losses.avg(), acc_batch.avg())
                    )

            if self.buffer.update_method.__class__ is Carto_update:
                self.model = self.model.eval()
                carto_data_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=False)
                last_index = 0
                for batch_data in carto_data_loader:
                    batch_x, batch_y = batch_data
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    with torch.no_grad():
                        features = self.model.forward(batch_x).unsqueeze(1)

                    true_label_probs = softmax(features, dim=2).squeeze(1).gather(1, batch_y.unsqueeze(1)).squeeze(1)
                    for sample_idx, prob in enumerate(true_label_probs):
                        sample_data = self.carto_data[self.task_seen][last_index + sample_idx]
                        sample_data.probabilities.append(prob.item())
                        sample_data.index = last_index + sample_idx
                        sample_data.experience = self.task_seen
                        sample_data.label = batch_y[sample_idx].item()
                    last_index += len(true_label_probs)
                self.model = self.model.train()

        if self.buffer.update_method.__class__ is Carto_update:
            # update mem
            carto_data_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=False)
            x = None
            y = None
            for batch_data in carto_data_loader:
                batch_x, batch_y = batch_data
                if x is None:
                    x = batch_x
                    y = batch_y
                else:
                    x = torch.cat((x, batch_x))
                    y = torch.cat((y, batch_y))

            self.buffer.update(
                x,
                y,
                task_seen=self.task_seen,
                carto_data=self.carto_data,
                classes_seen=self.classes_seen
            )

        self.after_train()
