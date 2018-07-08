import torch.nn.functional as F


def bootstrapped_loss(base=F.mse_loss):
    def loss_function(input, target):
        loss = 0.
        for i in range(input.shape[-1]):
            loss += base(input[:, i], target[:, i])
        return loss
    return loss_function