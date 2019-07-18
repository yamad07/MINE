import torch


class MINETrainer:

    def __init__(self, g, g_ma, d, mine, batch_size, z_dim, iter_num=100):
        self.g = g
        self.g_ma = g_ma
        self.d = d
        self.mine = mine
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.iter_num = iter_num
        self.ma_rate = 0.001
        self.ma_rate = 0.001

        self.d_optim = torch.optim.Adam(self.d.parameters(), lr=1e-4)
        self.g_optim = torch.optim.Adam(self.g.parameters(), lr=1e-4)
        self.mine_optim = torch.optim.Adam(self.mine.parameters(), lr=1e-4)

    def train(self, x, n_epoch):
        for epoch in range(n_epoch):
            discriminator_loss, gan_loss, mi = self._train_iter(x)
            print("EPOCH: {} D(x): {} G(x): {} MI: {}",
                  discriminator_loss, gan_loss, mi)

    def _train_iter(self, x):
        for i in range(self.iter_num):
            # Train discriminator
            batch = torch.FloatTensor(
                x[self.batch_size * i: self.batch_size * (i + 1)])
            z = torch.randn((self.batch_size, self.z_dim))
            x_tilde = self.g(z)
            d_x_tilde = self.d(x_tilde)

            d_x = self.d(x)
            loss = 0
            discriminator_loss = -torch.log(d_x).mean() - torch.log(
                1 - d_x_tilde).mean()

            self.d_optim.zero_grad()
            discriminator_loss.backward()
            self.d_optim.step()

            # Train generator
            z = torch.randn((self.batch_size, self.z_dim))
            z_bar = torch.narrow(
                torch.randn((self.batch_size, self.z_dim)),
                dim=1,
                start=0,
                length=3)

            x_tilde = self.g(z)
            d_x_tilde = self.d(x_tilde)

            d_x = self.d(x)
            gan_loss = -torch.log(d_x).mean() - torch.log(1 - d_x_tilde).mean()

            z = torch.narrow(z, dim=1, start=0, length=3)
            mi = torch.mean(self.mine(z, x_tilde) - torch.log(
                torch.mean(torch.exp(self.mine(z_bar, x_tilde)))) + 1e-8)
            loss = gan_loss - 0.01 * mi
            self.g_optim.zero_grad()
            loss.backward()
            self.g_optim.step()

            # Train mine
            z = torch.randn((self.batch_size, 10))
            z_bar = torch.narrow(torch.randn((self.batch_size, 10)),
                                 dim=1, start=0, length=3)
            x_tilde = self.g(z)
            et = torch.mean(torch.exp(self.mine(z_bar, x_tilde)))
            if self.mine.ma_et is None:
                self.mine.ma_et = et.detach().item()
            self.mine.ma_et += self.ma_rate * (
                et.detach().item() - self.mine.ma_et)
            z = torch.narrow(z, dim=1, start=0, length=3)
            mutual_information = torch.mean(self.mine(
                z, x_tilde)) - torch.log(et) * et.detach() / self.mine.ma_et
            loss = - mutual_information

            self.mine_optim.zero_grad()
            loss.backward()
            self.mine_optim.step()
        return discriminator_loss, gan_loss, mutual_information
