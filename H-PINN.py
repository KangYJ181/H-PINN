import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from sklearn.metrics import r2_score
import random


# ensure that the network training is reproducible
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# training device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Swish activation function
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


# define the network model
class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_units):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_units))
        self.layers.append(Swish())

        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
            self.layers.append(Swish())

        self.layers.append(nn.Linear(hidden_units, output_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# normalization function
def normalize(data, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        min_val = np.min(data)
        max_val = np.max(data)

    center = (max_val + min_val) / 2
    scale = (max_val - min_val) / 2
    normalized_data = (data - center) / scale

    return normalized_data, min_val, max_val, center, scale


# inverse normalization function
def denormalize(normalized_data, center, scale):
    return normalized_data * scale + center


# load the data
def load_data(data_path):
    t_data = sio.loadmat(os.path.join(data_path, 't.mat'))['t'].flatten()
    a_v_data = sio.loadmat(os.path.join(data_path, 'a_v.mat'))['a_v']
    a_p_data = sio.loadmat(os.path.join(data_path, 'a_p.mat'))['a_p']

    output_data = np.concatenate([a_v_data.T, a_p_data.T], axis=1)
    return t_data, output_data, a_v_data.shape[0], a_p_data.shape[0]


# physical loss function
def compute_phys_loss(model, inputs, A, B_permute, C, D, E, F_permute, G_inv, H, centers_v, scales_v, centers_p, scales_p, scale_t, center_t, ns_loss=True, return_ap_norm=False):

    if ns_loss:
        inputs.requires_grad_(True)

    a_v_norm = model(inputs)  # (batch_size, 13)
    batch_size = a_v_norm.shape[0]

    # inverse normalization
    a_v = a_v_norm * scales_v + centers_v
    t_real = inputs * scale_t + center_t

    # the expected residual corresponding to PPE (based on the Fourier fitting)
    expected_residuals_Poisson = torch.zeros((batch_size, 8), device=device)

    # first-order PPE residual
    expected_residuals_Poisson[:, 0] = (-0.00008445 - 0.12066866 * torch.cos(0.498666 * t_real)
                                        - 0.01142628 * torch.sin(0.498666 * t_real)).squeeze()

    expected_residuals_Poisson[:, 1] = (-0.00000251 - 0.01710957 * torch.cos(0.498666 * t_real)
                                        + 0.11826501 * torch.sin(0.498666 * t_real)).squeeze()

    expected_residuals_Poisson[:, 2] = (0.00127325 + 0.00104148 * torch.cos(0.498666 * t_real)
                                        - 0.00183626 * torch.sin(0.498666 * t_real)
                                        - 0.13006063 * torch.cos(0.997331 * t_real)
                                        + 0.06258192 * torch.sin(0.997331 * t_real)).squeeze()

    expected_residuals_Poisson[:, 3] = (0.00008972 + 0.00181127 * torch.cos(0.498666 * t_real)
                                        + 0.00091820 * torch.sin(0.498666 * t_real)
                                        - 0.06228814 * torch.cos(0.997331 * t_real)
                                        - 0.12878394 * torch.sin(0.997331 * t_real)).squeeze()

    expected_residuals_Poisson[:, 4] = (0.00001124 - 0.00580745 * torch.cos(0.498666 * t_real)
                                        - 0.00184639 * torch.sin(0.498666 * t_real)
                                        + 0.00131890 * torch.cos(0.997331 * t_real)
                                        + 0.00047863 * torch.sin(0.997331 * t_real)
                                        + 0.03309401 * torch.cos(1.495997 * t_real)
                                        - 0.01428269 * torch.sin(1.495997 * t_real)).squeeze()

    expected_residuals_Poisson[:, 5] = (-0.00000124 - 0.00204656 * torch.cos(0.498666 * t_real)
                                        - 0.00628699 * torch.sin(0.498666 * t_real)
                                        - 0.00075318 * torch.cos(0.997331 * t_real)
                                        + 0.00134350 * torch.sin(0.997331 * t_real)
                                        + 0.01307274 * torch.cos(1.495997 * t_real)
                                        + 0.03422323 * torch.sin(1.495997 * t_real)).squeeze()

    expected_residuals_Poisson[:, 6] = (0.00024504 + 0.00006876 * torch.cos(0.498666 * t_real)
                                        - 0.00006923 * torch.sin(0.498666 * t_real)
                                        + 0.01072500 * torch.cos(0.997331 * t_real)
                                        + 0.00137016 * torch.sin(0.997331 * t_real)
                                        + 0.00077331 * torch.cos(1.495997 * t_real)
                                        - 0.00067481 * torch.sin(1.495997 * t_real)
                                        - 0.03200049 * torch.cos(1.994662 * t_real)
                                        + 0.00589500 * torch.sin(1.994662 * t_real)).squeeze()

    expected_residuals_Poisson[:, 7] = (-0.00020059 + 0.00023596 * torch.cos(0.498666 * t_real)
                                        - 0.00022787 * torch.sin(0.498666 * t_real)
                                        - 0.00117868 * torch.cos(0.997331 * t_real)
                                        + 0.01047291 * torch.sin(0.997331 * t_real)
                                        + 0.00070118 * torch.cos(1.495997 * t_real)
                                        + 0.00078215 * torch.sin(1.495997 * t_real)
                                        - 0.00566128 * torch.cos(1.994662 * t_real)
                                        - 0.03168504 * torch.sin(1.994662 * t_real)).squeeze()

    # the first term of PPE
    Poisson_1 = torch.einsum('ij,bj->bi', E, a_v)

    Poisson_2 = torch.einsum('bm,mnj,bn->bj', a_v, F_permute, a_v)

    # calculate a_p
    H_expanded = H.unsqueeze(0).repeat(batch_size, 1)
    rhs = expected_residuals_Poisson - Poisson_1 - Poisson_2 - H_expanded
    a_p = torch.einsum('ij,bj->bi', G_inv, rhs)

    # normalization
    a_p_norm = (a_p - centers_p) / scales_p

    if ns_loss:
        # automatic differentiation calculates the derivative terms
        da_v_dt_norm = []
        for i in range(13):
            grad_i = torch.autograd.grad(a_v_norm[:, i], inputs, grad_outputs=torch.ones_like(a_v_norm[:, i]),
                                         create_graph=True, retain_graph=True)[0]
            da_v_dt_norm.append(grad_i)
        da_v_dt_norm = torch.cat(da_v_dt_norm, dim=1)

        # the first term of N-S
        NS_1 = (scales_v / scale_t) * da_v_dt_norm

        NS_2 = torch.einsum('ij,bj->bi', A, a_v)

        NS_3 = torch.einsum('bm,mnj,bn->bj', a_v, B_permute, a_v)

        NS_4 = torch.einsum('ij,bj->bi', C, a_p)

        D_expanded = D.unsqueeze(0).repeat(batch_size, 1)
        NS_5 = D_expanded

        # prediction Residual of N-S
        F_NS = NS_1 + NS_2 + NS_3 + NS_4 + NS_5

        # the expected residual corresponding to N-S (based on the Fourier fitting)
        expected_residuals_NS = torch.zeros_like(F_NS)

        # first-order N-S residual
        expected_residuals_NS[:, 0] = (0.00028181 + 0.36086871 * torch.cos(0.498666 * t_real)
                                       - 0.05950400 * torch.sin(0.498666 * t_real)).squeeze()

        expected_residuals_NS[:, 1] = (-0.00004367 - 0.05794308 * torch.cos(0.498666 * t_real)
                                       - 0.35512783 * torch.sin(0.498666 * t_real)).squeeze()

        expected_residuals_NS[:, 2] = (-0.00003651 - 0.08411388 * torch.cos(0.997331 * t_real)
                                       + 0.19664800 * torch.sin(0.997331 * t_real)).squeeze()

        expected_residuals_NS[:, 3] = (0.00011648 + 0.19587388 * torch.cos(0.997331 * t_real)
                                       + 0.08296880 * torch.sin(0.997331 * t_real)).squeeze()

        expected_residuals_NS[:, 4] = (-0.00000394 - 0.00670765 * torch.cos(1.495997 * t_real)
                                       + 0.09960074 * torch.sin(1.495997 * t_real)).squeeze()

        expected_residuals_NS[:, 5] = (-0.00008062 - 0.09980309 * torch.cos(1.495997 * t_real)
                                       - 0.00672767 * torch.sin(1.495997 * t_real)).squeeze()

        expected_residuals_NS[:, 6] = (0.00004526 - 0.00960718 * torch.cos(1.994662 * t_real)
                                       + 0.04640127 * torch.sin(1.994662 * t_real)).squeeze()

        expected_residuals_NS[:, 7] = (0.00002299 - 0.04657640 * torch.cos(1.994662 * t_real)
                                       - 0.00958780 * torch.sin(1.994662 * t_real)).squeeze()

        expected_residuals_NS[:, 8] = (-0.00000393 - 0.00000957 * torch.cos(0.498666 * t_real)
                                       + 0.00000069 * torch.sin(0.498666 * t_real)
                                       + 0.00000130 * torch.cos(0.997331 * t_real)
                                       - 0.00007279 * torch.sin(0.997331 * t_real)
                                       + 0.00030897 * torch.cos(1.495997 * t_real)
                                       - 0.00049687 * torch.sin(1.495997 * t_real)
                                       + 0.00005216 * torch.cos(1.994662 * t_real)
                                       - 0.00021136 * torch.sin(1.994662 * t_real)
                                       - 0.00682011 * torch.cos(2.493328 * t_real)
                                       + 0.02515497 * torch.sin(2.493328 * t_real)).squeeze()

        expected_residuals_NS[:, 9] = (0.00001963 + 0.00014436 * torch.cos(0.498666 * t_real)
                                       + 0.00001703 * torch.sin(0.498666 * t_real)
                                       + 0.00003617 * torch.cos(0.997331 * t_real)
                                       - 0.00000895 * torch.sin(0.997331 * t_real)
                                       - 0.00038747 * torch.cos(1.495997 * t_real)
                                       - 0.00032185 * torch.sin(1.495997 * t_real)
                                       - 0.00002318 * torch.cos(1.994662 * t_real)
                                       - 0.00003196 * torch.sin(1.994662 * t_real)
                                       + 0.02522066 * torch.cos(2.493328 * t_real)
                                       + 0.00683690 * torch.sin(2.493328 * t_real)).squeeze()

        expected_residuals_NS[:, 10] = (0.00001464 + 0.00008122 * torch.cos(0.997331 * t_real)
                                        - 0.00001637 * torch.sin(0.997331 * t_real)
                                        + 0.00021847 * torch.cos(1.994662 * t_real)
                                        - 0.00029310 * torch.sin(1.994662 * t_real)
                                        - 0.00747517 * torch.cos(2.991993 * t_real)
                                        + 0.01243641 * torch.sin(2.991993 * t_real)
                                        + 0.00193593 * torch.cos(3.989324 * t_real)
                                        + 0.00003826 * torch.sin(3.989324 * t_real)).squeeze()
        expected_residuals_NS[:, 11] = (-0.00002566 + 0.00004575 * torch.cos(0.997331 * t_real)
                                        - 0.00008440 * torch.sin(0.997331 * t_real)
                                        - 0.00020746 * torch.cos(1.994662 * t_real)
                                        - 0.00022431 * torch.sin(1.994662 * t_real)
                                        + 0.01242407 * torch.cos(2.991993 * t_real)
                                        + 0.00737955 * torch.sin(2.991993 * t_real)
                                        - 0.00006941 * torch.cos(3.989324 * t_real)
                                        + 0.00212673 * torch.sin(3.989324 * t_real)).squeeze()

        expected_residuals_NS[:, 12] = (0.00000125 - 0.00479164 * torch.cos(3.490659 * t_real)
                                        - 0.01780669 * torch.sin(3.490659 * t_real)).squeeze()

        # calculate the difference between the predicted residual of the network and the expected residual
        diff_NS = F_NS - expected_residuals_NS

        # physical loss
        phys_loss = torch.mean(torch.sum(diff_NS ** 2, dim=1))
    else:
        phys_loss = torch.tensor(0.0, device=device)

    if return_ap_norm:
        return phys_loss, a_p_norm
    else:
        return phys_loss, None


# training function
def train_model(model, train_loader, val_loader, epochs, lr, device, w,
                A, B_permute, C, D, E, F_permute, G_inv, H,
                centers_v, scales_v, centers_p, scales_p, scale_t, center_t,
                num_collocation, num_ext_collocation,
                start_epoch=0,
                adam_optimizer=None, lbfgs_optimizer=None,
                checkpoint_path='checkpoint.pth'
):
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    mse_losses = []
    phys_losses = []

    # initialize the optimizer
    if adam_optimizer is None:
        adam_optimizer = optim.Adam(model.parameters(), lr=lr)

    # training set domain
    full_train_inputs = train_loader.dataset.tensors[0]
    min_t = full_train_inputs.min().item()
    max_t = full_train_inputs.max().item()

    num = start_epoch + 1
    for epoch in range(start_epoch, epochs):
        i = epoch + 1
        decay = int((i - 1) / 2500)  # decaying learning rate
        current_lr = lr * (0.95 ** decay)
        for param_group in adam_optimizer.param_groups:
            param_group['lr'] = current_lr
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            adam_optimizer.zero_grad()
            a_v_norm = model(inputs)

            # data cost
            phys_loss_sup, a_p_norm = compute_phys_loss(model, inputs, A, B_permute, C, D, E, F_permute, G_inv, H, centers_v,
                                                        scales_v, centers_p, scales_p, scale_t, center_t, ns_loss=True, return_ap_norm=True)
            outputs = torch.cat((a_v_norm, a_p_norm), dim=1)
            mse_loss = criterion(outputs, targets)

            # physical cost within training domain
            collocation_inputs_in = torch.tensor(np.random.uniform(min_t, max_t, (num_collocation, 1)),
                                                 dtype=torch.float32).to(device)
            phys_loss_col_in, _ = compute_phys_loss(model, collocation_inputs_in, A, B_permute, C, D, E, F_permute, G_inv, H,
                                                    centers_v,scales_v, centers_p, scales_p, scale_t, center_t, ns_loss=True, return_ap_norm=False)

            # physical cost beyond training domain
            collocation_inputs_ext = torch.tensor(np.random.uniform(1.0, 3.0, (num_ext_collocation, 1)),
                                                  dtype=torch.float32).to(device)
            phys_loss_col_ext, _ = compute_phys_loss(model, collocation_inputs_ext, A, B_permute, C, D, E, F_permute, G_inv, H,
                                                     centers_v, scales_v, centers_p, scales_p, scale_t, center_t, ns_loss=True, return_ap_norm=False)

            phys_loss_value = phys_loss_sup + phys_loss_col_in + phys_loss_col_ext
            loss = mse_loss + w * phys_loss_value

            loss.backward()
            adam_optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_losses.append(train_loss / len(train_loader.dataset))

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    a_v_norm = model(inputs)
                    _, a_p_norm = compute_phys_loss(model, inputs, A, B_permute, C, D, E, F_permute, G_inv, H, centers_v,
                                                    scales_v, centers_p, scales_p, scale_t, center_t, ns_loss=False, return_ap_norm=True)
                    outputs = torch.cat((a_v_norm, a_p_norm), dim=1)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            val_losses.append(val_loss / len(val_loader.dataset))

        mse_losses.append(mse_loss.item())
        phys_losses.append(phys_loss_value.item())

        print(num, mse_loss.item(), (phys_loss_value).item())
        num += 1

        # checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'adam_optimizer_state_dict': adam_optimizer.state_dict(),
            'train_losses': train_losses,
        }
        torch.save(checkpoint, checkpoint_path)

    return train_losses, val_losses, mse_losses, phys_losses


# prediction function
def get_full_prediction(model, inputs, A, B_permute, C, D, E, F_permute, G_inv, H, centers_v, scales_v, centers_p, scales_p, scale_t, center_t):
    inputs = inputs.to(device)
    model.eval()
    with torch.no_grad():
        a_v_norm = model(inputs)
        _, a_p_norm = compute_phys_loss(model, inputs, A, B_permute, C, D, E, F_permute, G_inv, H, centers_v,
                                        scales_v, centers_p, scales_p, scale_t, center_t, ns_loss=False, return_ap_norm=True)
        outputs = torch.cat((a_v_norm, a_p_norm), dim=1)
    return outputs.cpu().numpy()


def main():
    data_path = r"training data"
    t_data, output_data, n_v, n_p = load_data(data_path)
    n_output = n_v + n_p

    # normalization
    t_norm, min_t, max_t, center_t, scale_t = normalize(t_data.reshape(-1, 1))
    output_norm = np.zeros_like(output_data)
    mins_o = np.zeros(n_output)
    maxs_o = np.zeros(n_output)
    centers = np.zeros(n_output)
    scales = np.zeros(n_output)
    for i in range(n_output):
        output_norm[:, i], mins_o[i], maxs_o[i], centers[i], scales[i] = normalize(output_data[:, i])

    X = torch.FloatTensor(t_norm)
    y = torch.FloatTensor(output_norm)

    # training set and test set
    n_total = len(X)
    n_test = 190
    indices = np.random.permutation(n_total)
    test_indices, train_indices = indices[:n_test], indices[n_test:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # hyperparameter setting
    epochs = 40000
    hidden_layers = 3
    hidden_units = 80
    lr = 1e-2  # initial learning rate
    w = 1e0  # physical weight
    num_collocation = 442  # collocation point within training domain
    num_ext_collocation = 442  # collocation point beyond training domain

    # load the operator matrix
    mat_path = r"operator"
    A_np = sio.loadmat(os.path.join(mat_path, 'A.mat'))['A']
    B_permute_np = sio.loadmat(os.path.join(mat_path, 'B_permute.mat'))['B_permute']
    C_np = sio.loadmat(os.path.join(mat_path, 'C.mat'))['C']
    D_np = sio.loadmat(os.path.join(mat_path, 'D.mat'))['D']
    E_np = sio.loadmat(os.path.join(mat_path, 'E.mat'))['E']
    F_permute_np = sio.loadmat(os.path.join(mat_path, 'F_permute.mat'))['F_permute']
    G_np = sio.loadmat(os.path.join(mat_path, 'G.mat'))['G']
    H_np = sio.loadmat(os.path.join(mat_path, 'H.mat'))['H']

    # convert to the Torch tensor
    A = torch.tensor(A_np, dtype=torch.float32, device=device)
    B_permute = torch.tensor(B_permute_np, dtype=torch.float32, device=device)
    C = torch.tensor(C_np, dtype=torch.float32, device=device)
    D = torch.tensor(D_np, dtype=torch.float32, device=device).squeeze()
    E = torch.tensor(E_np, dtype=torch.float32, device=device)
    F_permute = torch.tensor(F_permute_np, dtype=torch.float32, device=device)
    G = torch.tensor(G_np, dtype=torch.float32, device=device)
    G_inv = torch.inverse(G)
    H = torch.tensor(H_np, dtype=torch.float32, device=device).squeeze()

    centers_v = torch.tensor(centers[:13], dtype=torch.float32, device=device)
    scales_v = torch.tensor(scales[:13], dtype=torch.float32, device=device)
    centers_p = torch.tensor(centers[13:], dtype=torch.float32, device=device)
    scales_p = torch.tensor(scales[13:], dtype=torch.float32, device=device)
    scale_t = torch.tensor(scale_t, dtype=torch.float32, device=device)

    final_train_dataset = TensorDataset(X_train, y_train)
    final_train_loader = DataLoader(final_train_dataset, batch_size=64, shuffle=True)

    checkpoint_path = 'checkpoint.pth'

    set_seed(42)
    final_model = PINN(1, n_v, hidden_layers, hidden_units).to(device)

    # if there are checkpoints, load the model, optimizer, and epoch
    start_epoch = 0
    adam_optimizer = optim.Adam(final_model.parameters(), lr=lr)
    lbfgs_optimizer = None
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        final_model.load_state_dict(checkpoint['model_state_dict'])
        adam_optimizer.load_state_dict(checkpoint['adam_optimizer_state_dict'])
        if 'lbfgs_optimizer_state_dict' in checkpoint:
            lbfgs_optimizer = optim.LBFGS(final_model.parameters(), lr=1.0, max_iter=20, history_size=100)
            lbfgs_optimizer.load_state_dict(checkpoint['lbfgs_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    train_losses, _, mse_losses, phys_losses= train_model(
        final_model, final_train_loader, None, epochs, lr, device, w,
        A, B_permute, C, D, E, F_permute, G_inv, H,
        centers_v, scales_v, centers_p, scales_p,
        scale_t, center_t, num_collocation,
        num_ext_collocation=num_ext_collocation,
        start_epoch=start_epoch, adam_optimizer=adam_optimizer,
        lbfgs_optimizer=lbfgs_optimizer, checkpoint_path=checkpoint_path
    )

    # save the model
    torch.save(final_model.state_dict(), 'PINN_model.pth')

    import pandas as pd
    df = pd.DataFrame({
        'epoch': range(1, len(mse_losses) + 1),
        'mse_loss': mse_losses,
        'phys_loss_value': phys_losses
    })
    df.to_csv('epoch_losses.csv', index=False)

    # evaluate the model
    final_model.eval()
    y_train_pred_norm = get_full_prediction(final_model, X_train, A, B_permute, C, D, E, F_permute, G_inv, H, centers_v, scales_v, centers_p, scales_p, scale_t, center_t)
    y_test_pred_norm = get_full_prediction(final_model, X_test, A, B_permute, C, D, E, F_permute, G_inv, H, centers_v, scales_v, centers_p, scales_p, scale_t, center_t)
    all_pred_norm = get_full_prediction(final_model, X, A, B_permute, C, D, E, F_permute, G_inv, H, centers_v, scales_v, centers_p, scales_p, scale_t, center_t)

    y_train_true_norm = y_train.numpy()
    y_test_true_norm = y_test.numpy()

    def calculate_mse(true, pred):
        return np.mean((true - pred) ** 2)

    train_mse = calculate_mse(y_train_true_norm, y_train_pred_norm)
    test_mse = calculate_mse(y_test_true_norm, y_test_pred_norm)

    train_r2 = r2_score(y_train_true_norm, y_train_pred_norm)
    test_r2 = r2_score(y_test_true_norm, y_test_pred_norm)

    print(f"training set MSE: {train_mse:.15f}")
    print(f"test set MSE: {test_mse:.15f}")
    print(f"training set R2: {train_r2:.15f}")
    print(f"test set R2: {test_r2:.15f}")

    # plotting of the scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(y_train_true_norm.flatten(), y_train_pred_norm.flatten(), alpha=0.5,
                label=f'Training set ($R^2={train_r2:.4f}$)', color='blue', s=10)
    plt.scatter(y_test_true_norm.flatten(), y_test_pred_norm.flatten(), alpha=0.5,
                label=f'Test set ($R^2={test_r2:.4f}$)', color='red', s=10)
    min_val = min(np.min(y_train_true_norm), np.min(y_test_true_norm))
    max_val = max(np.max(y_train_true_norm), np.max(y_test_true_norm))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal fit')
    plt.xlabel('True Values (Normalized)')
    plt.ylabel('Predicted Values (Normalized)')
    plt.title('Predicted vs True Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_scatter.png')
    plt.show()

    # plot the curve depicting the variation of the normalized reduced-order coefficients over time
    # velocity mode coefficient
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        if i < n_v:
            ax.plot(t_norm.flatten(), output_norm[:, i], 'b-', label='True')
            ax.plot(t_norm.flatten(), all_pred_norm[:, i], 'r--', label='Predicted')
            ax.set_title(f'Velocity POD Coef. {i + 1}')
            ax.grid(True)
        else:
            ax.axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.suptitle('Velocity POD Coefficient vs. Time', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('velocity_coefficients.png')
    plt.show()

    # pressure mode coefficient
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for i, ax in enumerate(axes.flat):
        if i < n_p:
            ax.plot(t_norm.flatten(), output_norm[:, n_v + i], 'b-', label='True')
            ax.plot(t_norm.flatten(), all_pred_norm[:, n_v + i], 'r--', label='Predicted')
            ax.set_title(f'Pressure POD Coef. {i + 1}')
            ax.grid(True)
        else:
            ax.axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.suptitle('Pressure POD Coefficient vs. Time', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pressure_coefficients.png')
    plt.show()


    # plot the predicted results (extend the time axis by 2 times)
    t_norm = t_norm.flatten()

    step = t_norm[1] - t_norm[0]
    t_ext = np.arange(-1, 3 + step, step)

    def extend_true_curve(t_base, y_base, period=2, max_t=3):
        repeat_times = int(np.ceil((max_t - 1) / period))
        t_ext_all, y_ext_all = [], []
        for k in range(repeat_times + 1):
            shifted_t = t_base + period * k
            mask = (shifted_t <= max_t)
            if np.any(mask):
                t_ext_all.append(shifted_t[mask])
                y_ext_all.append(y_base[mask, :])
        return np.concatenate(t_ext_all), np.vstack(y_ext_all)

    t_true_v, y_true_v = extend_true_curve(t_norm, output_norm[:, :n_v], period=2, max_t=3)
    t_true_p, y_true_p = extend_true_curve(t_norm, output_norm[:, n_v:], period=2, max_t=3)

    X_ext = torch.FloatTensor(t_ext.reshape(-1, 1))
    y_pred_ext = get_full_prediction(final_model, X_ext, A, B_permute, C, D, E, F_permute, G_inv, H, centers_v, scales_v, centers_p, scales_p, scale_t, center_t)

    # velocity mode coefficient
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        if i < n_v:
            ax.plot(t_true_v, y_true_v[:, i], 'b-', label='True')
            ax.plot(t_ext, y_pred_ext[:, i], 'r--', label='Predicted')
            ax.set_xlim([-1, 3])
            ax.set_title(f'Velocity POD Coef. {i + 1}')
            ax.grid(True)
        else:
            ax.axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.suptitle('Velocity POD Coefficient vs. Time (Extended)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('velocity_coefficients_extended.png')
    plt.show()

    # pressure mode coefficient
    fig, axes = plt.subplots(4, 4, figsize=(16, 9))
    for i, ax in enumerate(axes.flat):
        if i < n_p:
            ax.plot(t_true_p, y_true_p[:, i], 'b-', label='True')
            ax.plot(t_ext, y_pred_ext[:, n_v + i], 'r--', label='Predicted')
            ax.set_xlim([-1, 3])
            ax.set_title(f'Pressure POD Coef. {i + 1}')
            ax.grid(True)
        else:
            ax.axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.suptitle('Pressure POD Coefficient vs. Time (Extended)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pressure_coefficients_extended.png')
    plt.show()

    t_real_ext = np.arange(0, 25.28, 0.02)

    t_norm_ext = (t_real_ext - center_t) / scale_t.item()
    X_ext = torch.FloatTensor(t_norm_ext.reshape(-1, 1))

    y_pred_norm_ext = get_full_prediction(final_model, X_ext, A, B_permute, C, D, E, F_permute, G_inv, H, centers_v, scales_v, centers_p, scales_p, scale_t, center_t)

    y_pred_ext = np.zeros_like(y_pred_norm_ext)
    for i in range(n_output):
        y_pred_ext[:, i] = denormalize(y_pred_norm_ext[:, i], centers[i], scales[i])

    velocity_coefs = y_pred_ext[:, :n_v].T  # (n_v, 632)
    pressure_coefs = y_pred_ext[:, n_v:].T  # (n_p, 632)

    empty_row = np.full((1, 1264), np.nan)

    final_data = np.vstack([velocity_coefs, empty_row, pressure_coefs])

    # save the prediciotn results
    import pandas as pd
    df = pd.DataFrame(final_data)
    df.to_csv('pod_coefficients_predictions.csv', index=False, header=False, na_rep='')


if __name__ == "__main__":
    main()