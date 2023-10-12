import torch
from torch import softmax
from torch.utils import data

from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.buffer.buffer import Buffer
from utils.buffer.carto_update import Carto_update
from utils.loss import SupConLoss
from utils.setup_elements import transforms_match, transforms_aug
from utils.utils import maybe_cuda


class ProxyContrastiveReplay(ContinualLearner):
    """
        Proxy-based Contrastive Replay,
        Implements the strategy defined in
        "PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning"
        https://arxiv.org/abs/2304.04408

        This strategy has been designed and tested in the
        Online Setting (OnlineCLScenario). However, it
        can also be used in non-online scenarios
        """

    def __init__(self, model, opt, params):
        super(ProxyContrastiveReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.classes_seen = set()

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        self.classes_seen = self.classes_seen.union(set(y_train))
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()
        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x_aug = torch.stack([transforms_aug[self.data](batch_x[idx].cpu())
                                           for idx in range(batch_x.size(0))])
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_x_aug = maybe_cuda(batch_x_aug, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x_combine = torch.cat((batch_x, batch_x_aug))
                batch_y_combine = torch.cat((batch_y, batch_y))
                for j in range(self.mem_iters):
                    logits, feas = self.model.pcrForward(batch_x_combine)
                    novel_loss = 0 * self.criterion(logits, batch_y_combine)
                    self.opt.zero_grad()

                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        # mem_x, mem_y = Rotation(mem_x, mem_y)
                        mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                                                 for idx in range(mem_x.size(0))])
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_x_combine = torch.cat([mem_x, mem_x_aug])
                        mem_y_combine = torch.cat([mem_y, mem_y])

                        mem_logits, mem_fea = self.model.pcrForward(mem_x_combine)

                        combined_feas = torch.cat([mem_fea, feas])
                        combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                        combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels]

                        combined_feas_norm = torch.norm(combined_feas, p=2, dim=1).unsqueeze(1).expand_as(combined_feas)
                        combined_feas_normalized = combined_feas.div(combined_feas_norm + 0.000001)

                        combined_feas_aug_norm = torch.norm(combined_feas_aug, p=2, dim=1).unsqueeze(1).expand_as(
                            combined_feas_aug)
                        combined_feas_aug_normalized = combined_feas_aug.div(combined_feas_aug_norm + 0.000001)
                        cos_features = torch.cat([combined_feas_normalized.unsqueeze(1),
                                                  combined_feas_aug_normalized.unsqueeze(1)],
                                                 dim=1)
                        PSC = SupConLoss(temperature=0.09, contrast_mode='proxy')
                        novel_loss += PSC(features=cos_features, labels=combined_labels)

                    novel_loss.backward()
                    self.opt.step()
                # update mem
                if self.buffer.update_method.__class__ is not Carto_update:
                    self.buffer.update(batch_x, batch_y)

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
