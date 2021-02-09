from torch import nn, optim
from torch.nn import functional as f


class VanillaCNN(nn.Module):
    def __init__(
        self,
        input_size: tuple = (1, 256, 256),
        kernel_size: int = 3,
        channel_dims: list = [32, 64],
        hidden_dims=(256, 128),
        n_classes=26,
        batchnorm=False,
    ):
        super(VanillaCNN, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.channel_dims = channel_dims
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.batchnorm = batchnorm

        self.layers = []
        prev_channel_dim = self.input_size[0]
        for channel_dim in self.channel_dims:
            self.layers.append(
                nn.Conv2d(
                    in_channels=prev_channel_dim,
                    out_channels=channel_dim,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                )
            )
            if self.batchnorm:
                self.layers.append(nn.BatchNorm2d(num_features=channel_dim))

            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.layers.append(nn.Dropout2d(p=0.5))
            prev_channel_dim = channel_dim

        self.layers.append(nn.Flatten())
        prev_hidden_dim = (
            prev_channel_dim
            * (self.input_size[1])
            // (2 ** len(self.channel_dims))
            * (self.input_size[2] // (2 ** len(self.channel_dims)))
        )
        for hidden_dim in self.hidden_dims:
            self.layers.append(
                nn.Linear(
                    in_features=prev_hidden_dim, out_features=hidden_dim, bias=True
                )
            )
            self.layers.append(nn.ReLU(inplace=True))
            prev_hidden_dim = hidden_dim

        self.layers.append(
            nn.Linear(
                in_features=prev_hidden_dim, out_features=self.n_classes, bias=True
            )
        )

        self.network = nn.Sequential()
        for layer_idx, layer in enumerate(self.layers):
            layer_name = f"{type(layer).__name__.lower()}_{layer_idx}"
            self.network.add_module(name=layer_name, module=layer)

        self.init_param()

    def init_param(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, X):
        return self.network.forward(X)
