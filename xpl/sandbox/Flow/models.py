import torch


class Flow_Network(torch.nn.Module):
    def __init__(self):
        super(Flow_Network, self).__init__()

        # These configs should be automated later but for now they are good

        self.out_channels = 6
        self.in_channels = 6

        self.net_out_channels = [64, 64, 64]
        self.kernel_size = 5

        self.encoder, self.decoder = self.compile_nets(in_channels=self.in_channels,
                                                       out_channels=self.out_channels,
                                                       net_out_channels=self.net_out_channels,
                                                       kernel_size=self.kernel_size)
        self.normalize_layer = torch.nn.BatchNorm2d(3)

        [self.add_module(f'encoder_{i}', encoder)
         for i, encoder in enumerate(self.encoder)]
        [self.add_module(f'decoder_{i}', decoder)
         for i, decoder in enumerate(self.decoder)]

        self.optim = torch.optim.Adam(params=self.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optim,
                                                             max_lr=1e-3,
                                                             steps_per_epoch=100,
                                                             epochs=100)
        self.zero_grad()

    def zero_grad(self):
        self.optim.zero_grad()

    def step(self):
        self.optim.step()
        self.scheduler.step()

    def compile_nets(self,
                     in_channels: int,
                     out_channels: int,
                     net_out_channels: list,
                     kernel_size: int):
        encoder = []
        in_c = in_channels
        for out_c in net_out_channels:
            encoder.append(torch.nn.Sequential(
                torch.nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=in_c,
                                                             out_channels=out_c,
                                                             kernel_size=kernel_size,
                                                             stride=1,
                                                             padding=kernel_size // 2)),
            ))
            in_c = out_c

        decoder = []

        for out_c in reversed([out_channels] + net_out_channels[:-1]):
            decoder.append(torch.nn.Sequential(
                torch.nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=in_c,
                                                             out_channels=out_c,
                                                             kernel_size=kernel_size,
                                                             stride=1,
                                                             padding=kernel_size // 2)),

            ))
            in_c = out_c

        return encoder, decoder

    def forward(self,
                first_frames: torch.FloatTensor,
                second_frames: torch.FloatTensor,
                ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """[summary]

        Args:
            self ([type]): [description]
            torch ([type]): [description]
            torch ([type]): [description]
            torch ([type]): [description]

        Returns:
            [type]: [description]
        """
        net_input = torch.cat(tensors=[self.normalize_layer(first_frames),
                                       self.normalize_layer(second_frames),
                                       ],
                              dim=1)

        max_pool_activations = []
        for layer in self.encoder:
            net_output = layer(net_input)
            if net_input.shape[1] == net_output.shape[1]:
                net_input = (net_input + net_output) / 2.0
            else:
                net_input = net_output
            net_input = torch.nn.functional.relu(net_input)
            net_input, activations = torch.nn.functional.max_pool2d_with_indices(input=net_input,
                                                                                 kernel_size=2,
                                                                                 stride=2)
            max_pool_activations.append(activations)

        for layer, activations in zip(self.decoder, reversed(max_pool_activations)):
            net_input = torch.nn.functional.max_unpool2d(input=net_input,
                                                         indices=activations,
                                                         kernel_size=2,
                                                         stride=2)
            net_output = layer(torch.nn.functional.relu(net_input))

            if net_input.shape[1] == net_output.shape[1]:
                net_input = (net_input + net_output) / 2.0
            else:
                net_input = net_output

        return (torch.tanh(net_input[:, 0:2, :, :]),  # Flows 1 -> 2
                torch.tanh(net_input[:, 2:4, :, :]),  # Flows 2 -> 1
                torch.nn.functional.logsigmoid(
                    1e-5+net_input[:, 4:5, :, :]),  # Occlusions 1-> 2
                torch.nn.functional.logsigmoid(1e-5+net_input[:, 5:6, :, :]))  # Occlusions 2 -> 1
