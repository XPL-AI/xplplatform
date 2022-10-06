import torch

from xpl.model.neural_net.xpl_model import XPLModel


class ImageClassifier(XPLModel):

    def init_neural_net(self
                        ) -> torch.nn.Module:
        return torch.nn.Conv2d(in_channels=self.definition['input_channels'],
                               out_channels=self.definition['output_channels'],
                               kernel_size=1)

    def forward(self,
                batch: dict
                ) -> None:
        input = batch[self.head_names[0]].to(self.device)
        output = self.neural_net(input).mean(-1).mean(-1)
        batch[self.tail_names[0]] = output
        return None
