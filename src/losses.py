import torch
from sklearn.mixture import GaussianMixture
from torch.nn.functional import normalize


def unsqueeze(x):
    if x.dim() == 1:
        return x.unsqueeze(1)
    else:
        return x


def similarity(x, y, kernel='linear'):
    # Compute the similarity between two sets.
    gamma = 1 / x.size(1)
    if kernel == 'linear':
        out = x.matmul(y.t())
    elif kernel == 'rbf':
        out = torch.exp(-gamma * torch.cdist(x, y))
    elif kernel == 'poly':
        c = 1
        d = 3
        out = (gamma * x.matmul(y.t()) + c) ** d
    else:
        raise ValueError(kernel)
    return out.mean()


def squared_mmd(x, y, kernel='linear'):
    out1 = similarity(x, x, kernel)
    out2 = similarity(y, y, kernel)
    out3 = similarity(x, y, kernel)
    return torch.clamp_min(out1 + out2 - 2 * out3, 0)  # Avoid underflow


def distance(x, y, metric='euclidean', kernel='linear', aggregation='mean'):
    # Compute the distance between two sets.
    if metric == 'mmd':
        out = squared_mmd(x, y, kernel)
    elif metric == 'manhattan':  # Divide by 2 to make it in [0, 1]
        out = torch.cdist(x, y, p=1) / 2
    elif metric == 'euclidean':
        out = torch.cdist(x, y, p=2) / 2
    elif metric == 'chebyshev':
        out = torch.cdist(x, y, p=torch.inf) / 2
    elif metric == 'cosine':  # This is actually not a distance metric.
        out = 1 - x.matmul(y.t())
    else:
        raise ValueError(metric)

    if aggregation == 'mean':
        return out.mean()
    elif aggregation == 'none':
        assert metric != 'mmd'
        return out
    else:
        raise ValueError(aggregation)


def estimate_gaussian(x, correction=False):
    mean = x.mean(0)
    cov = torch.cov(x.t(), correction=int(correction)) \
        .view(x.size(1), x.size(1))
    return mean, cov


def gaussian_kld(mean1, cov1, mean2, cov2):
    # Compute KLD(P1 | P2) between 2 Gaussian distributions
    inv2 = torch.inverse(cov2)
    out1 = torch.logdet(cov2) - torch.logdet(cov1)
    out2 = (mean1 - mean2).t().matmul(inv2).matmul(mean1 - mean2)
    out3 = torch.trace(inv2.matmul(cov1))
    return out1 + out2 + out3


def get_loss_names():
    return ['base', 'mmd', 'cov', 'ours-scaled', 'ours-cov', 'ours']


def get_all_losses(trn_emb, trn_labels, test_emb):
    trn_emb_n = trn_emb[trn_labels == 0]
    trn_emb_a = trn_emb[trn_labels == 1]
    metric = 'euclidean'

    out = [
        BaseLoss(metric)(trn_emb, test_emb),
        BaseLoss(metric='mmd')(trn_emb, test_emb),
        BaseCovLoss(metric)(trn_emb, test_emb),
        ScaledLoss(metric)(trn_emb_n, trn_emb_a, test_emb),
        -EmbCovLoss(metric, maximum=torch.inf)(trn_emb_n, trn_emb_a, test_emb),
        HybridLoss(metric)(trn_emb_n, trn_emb_a, test_emb),
    ]
    return [e.item() for e in out]


class BaseLoss:
    def __init__(self, metric='mmd', kernel='linear'):
        super().__init__()
        self.metric = metric
        self.kernel = kernel

    def __call__(self, x, y):
        x = unsqueeze(x)
        y = unsqueeze(y)
        return distance(x, y, self.metric, self.kernel)


class ScaledLoss:
    def __init__(self, metric='mmd', kernel='linear', epsilon=1e-8):
        super().__init__()
        self.metric = metric
        self.kernel = kernel
        self.epsilon = epsilon

    def __call__(self, x, y, z):
        x = unsqueeze(x)
        y = unsqueeze(y)
        z = unsqueeze(z)
        d1 = distance(torch.cat([x, y]), z, self.metric, self.kernel)
        d2 = distance(x, y, self.metric, self.kernel)
        return d1 / (d2 + self.epsilon)


class CovLoss:
    def __init__(self, inverse=True, log=False, normalize=True):
        super().__init__()
        self.inverse = inverse
        self.log = log
        self.normalize = normalize

    def __call__(self, scr_x, scr_y, scr_z):
        if self.log:
            scr_x = torch.log(scr_x)
            scr_y = torch.log(scr_y)
            scr_z = torch.log(scr_z)

        if self.normalize:
            scr_all = torch.cat([scr_x, scr_y, scr_z])
            scr_max = torch.max(scr_all)
            scr_min = torch.min(scr_all)
            if self.log:
                scr_z = (scr_z - scr_min) / (scr_max - scr_min)
            else:
                scr_z = scr_z / scr_max

        out = torch.cov(scr_z, correction=0)
        if self.inverse:
            out = -out
        return out


class BaseCovLoss:
    def __init__(self, metric='euclidean', epsilon=1e-8):
        super().__init__()
        self.metric = metric
        self.epsilon = epsilon

    def __call__(self, emb_x, emb_z):
        diff = ((emb_z.unsqueeze(0) - emb_x.unsqueeze(1)) ** 2).sum(2).sqrt()
        cov = torch.cov(diff.view(-1), correction=0).sqrt()
        return -cov


class EmbCovLoss:
    def __init__(self, metric='euclidean', epsilon=1e-8, sqrt=True, maximum=1 / 2):
        super().__init__()
        self.metric = metric
        self.epsilon = epsilon
        self.sqrt = sqrt
        self.maximum = maximum

    def __call__(self, emb_x, emb_y, emb_z):
        mean_x = emb_x.mean(0)
        units = normalize(emb_y - mean_x, dim=1)
        out = torch.einsum('ik,jk->ij', emb_z - mean_x, units).view(-1)
        cov = torch.cov(out, correction=0)
        if self.sqrt:
            cov = torch.sqrt(cov)
        dist = distance(emb_x, emb_y, self.metric)
        return torch.clamp_max(cov / dist, self.maximum)


class HybridLoss:
    def __init__(self, metric='mmd', kernel='linear', epsilon=1e-8, cov_style=3,
                 **cov_args):
        super().__init__()
        self.cov = EmbCovLoss(metric, **cov_args)
        self.cov_style = cov_style
        self.loss = ScaledLoss(metric, kernel, epsilon)

    def __call__(self, emb_x, emb_y, emb_z):
        cov = self.cov(emb_x, emb_y, emb_z)
        loss = self.loss(emb_x, emb_y, emb_z)
        if self.cov_style == 1:
            return loss - cov
        elif self.cov_style == 2:
            return loss ** 2 + (0.5 - cov) ** 2
        elif self.cov_style == 3:
            return loss - cov / loss
        elif self.cov_style == 4:
            return loss - cov / torch.sqrt(loss)
        elif self.cov_style == 5:
            return loss - cov / loss ** 2
        else:
            raise ValueError(self.cov_style)


class GMMLoss:
    def __init__(self, metric='euclidean', average='micro', beta=1.):
        super().__init__()
        self.metric = metric
        self.average = average
        self.beta = beta
        self.means = None
        self.covariances = None
        self.membership = None

    def distance(self, x, z, index):
        if self.metric.startswith('kld'):
            mean1, cov1 = estimate_gaussian(x)
            mean2 = self.means[index]
            cov2 = self.covariances[index]
            if self.metric == 'kld2':  # Switch P1 and P2
                mean1, cov1, mean2, cov2 = mean2, cov2, mean1, cov1
            return gaussian_kld(mean1, cov1, mean2, cov2)
        else:
            return distance(x, z, self.metric, aggregation='none') * \
                self.membership[:, index]

    def combine(self, a, b):
        if a.dim() == b.dim() == 0:
            return (a + b) / 2
        elif self.average == 'macro':
            return (a.mean() + b.mean()) / 2
        elif self.average == 'micro':
            return torch.cat([a, b]).mean()
        else:
            raise ValueError(self.average)

    def __call__(self, x, y, z):
        # Compute elementwise membership to the 2 distributions
        x = unsqueeze(x)
        y = unsqueeze(y)
        z = unsqueeze(z)

        gmm = GaussianMixture(n_components=2).fit(z)
        self.means = torch.tensor(gmm.means_, dtype=torch.float32)
        self.covariances = torch.tensor(gmm.covariances_, dtype=torch.float32)
        _, log_mem = gmm._e_step(z)
        self.membership = torch.exp(torch.tensor(log_mem, dtype=torch.float32))

        out = []
        for i in [0, 1]:
            dx = self.beta * self.distance(x, z, i)
            dy = self.distance(y, z, 1 - i)
            out.append(self.combine(dx, dy))
        return torch.min(torch.stack(out))  # Pick the optimal matching
