import torch
import copy
def contrastlearning(epochs,contrast_loader,classi_loader,optim_contrast,optim_classi, SToptimizer,Sptransnet,encoder,lentrain,lencontrast,GCNnet,GCNDATA,optimizer1, train_samples_gt_onehot, train_label_mask):
    contrast_criterion = ContrastiveLoss()
    criterion  = torch.nn.CrossEntropyLoss()

    GCNnet.train()
    for epoch in range(epochs):
        train_acc = 0
        epoch_contrastloss = 0
        epoch_classiloss = 0
        if epoch<40:
            for step, (x_i, x_j, label) in enumerate(contrast_loader):
                x_i = x_i.cuda().float()
                x_j = x_j.cuda().float()
                x_i = Sptransnet(x_i)
                x_j = Sptransnet(x_j)

                label = label.cuda().float()
                encoder = encoder.cuda()
                encoder.train()

                h_i, z_i = encoder(x_i,contrast = True)
                h_j, z_j = encoder(x_j,contrast = True)

                contrast_loss = contrast_criterion(h_i, h_j, label)
                epoch_contrastloss += contrast_loss.item()
                optim_contrast.zero_grad()
                SToptimizer.zero_grad()
                contrast_loss.backward(retain_graph=True)
                optim_contrast.step()
                SToptimizer.step()
        for i, (data, label) in enumerate(classi_loader):  # supervised part
            data = data.cuda().float()
            label = label.cuda().long()
            data = Sptransnet(data)
            h, z = encoder(data,contrast = True)
            classi_loss = criterion(z, label)
            epoch_classiloss += classi_loss.item()
            pred = torch.max(z, 1)[1]
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()
            optim_classi.zero_grad()
            SToptimizer.zero_grad()

            classi_loss.backward(retain_graph=True)
            optim_classi.step()
            SToptimizer.step()

        if epoch>40:
            GCN_input = Sptransnet(GCNDATA, full=True)
            output1, _ = encoder(GCN_input, full=True)
            output,loss2 = GCNnet(GCN_input, output1, merge=True)
            loss2 = torch.sum(loss2 * train_label_mask)
            del GCN_input
            optimizer1.zero_grad()
            loss1 = compute_loss(output, train_samples_gt_onehot, train_label_mask)
            loss1 = loss1+loss2
            loss1.backward(retain_graph=True)
            optimizer1.step()  # Does the update
            SToptimizer.step()
            if epoch % 10 == 0:
                print('GCN loss:%.6f' % loss1)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.25):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy