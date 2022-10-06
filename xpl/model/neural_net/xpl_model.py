import abc
import torch

from xpl.model.neural_net.xpl_optimizer import XPLOptimizer
import logging
logger = logging.getLogger(__name__)


class XPLModel(torch.nn.Module):

    def __init__(self,
                 name: str,
                 definition: dict,
                 ) -> None:
        super().__init__()

        self.name = name
        self.definition = definition
        self.head_names = self.definition['heads']
        self.tail_names = self.definition['tails']
        self.neural_nets = self.init_neural_net()
        for neural_net_name, neural_net in self.neural_nets.items():
            self.add_module(name=neural_net_name,
                            module=neural_net)
        self.device = 'cuda' if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 5 else 'cpu'
        self.__train = True

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_name(self):
        return self.name

    def get_heads(self):
        return self.head_names

    def get_tails(self):
        return self.tail_names

    @abc.abstractmethod
    def init_neural_net(self,
                        ) -> dict[str, torch.nn.Module]:
        pass

    @abc.abstractmethod
    def forward(self,
                batch: dict
                ) -> None:
        pass

    def train(self):
        self.__train = True
        [neural_net.train() for neural_net in self.neural_nets.values()]

    def eval(self):
        self.__train = False
        [neural_net.eval() for neural_net in self.neural_nets.values()]

    def zero_grad(self,
                  ) -> None:
        if self.__train:
            self.optimizer.zero_grad()

    def step(self,
             ) -> None:
        if self.__train:
            self.optimizer.step()

    def __str__(self):
        return f'{self.head_names=}\n{self.tail_names=}\n{self.neural_nets=}\n'

    def __call__(self,
                 batch: dict
                 ) -> None:
        self.to(self.device)
        if self.__train:
            self.forward(batch)
        else:
            with torch.no_grad():
                self.forward(batch)

    def get_script_module(self
                          ) -> torch.jit.ScriptModule:
        model_scripts = {neural_net_name: torch.jit.script(neural_net.cpu())
                         for neural_net_name, neural_net in self.neural_nets.items()}

        if self.device == 'cuda':
            model_scripts |= {f'{neural_net_name}_cuda': torch.jit.script(neural_net)
                              for neural_net_name, neural_net in self.neural_nets.items()}
        return model_scripts

    def get_state_dict(self
                       ) -> dict:

        return {'definition': self.definition,
                'optimizer_state_dict': self.optimizer.state_dict()} | {
                    f'{neural_net_name}_state_dict': self.neural_nets[neural_net_name].state_dict()
                    for neural_net_name in self.neural_nets.keys()
        }

    def load_state_dict(self,
                        state_dict: dict
                        ) -> None:

        if not state_dict:
            return None
        
        definition = state_dict['definition']
        # TODO: resolve definition != self.definition

        loaded_optimizer_state_dict = state_dict['optimizer_state_dict']
        original_optimizer_state_dict = self.optimizer.state_dict()
        self.optimizer.load_state_dict(
            self.__safe_copy_state_dict(original_state_dict=original_optimizer_state_dict,
                                        loaded_state_dict=loaded_optimizer_state_dict))

        for neural_net_name in self.neural_nets.keys():
            loaded_neural_net_state_dict = state_dict[f'{neural_net_name}_state_dict']
            original_neural_net_state_dict = self.neural_nets[neural_net_name].state_dict()
            self.neural_nets[neural_net_name].load_state_dict(
                self.__safe_copy_state_dict(original_state_dict=original_neural_net_state_dict,
                                            loaded_state_dict=loaded_neural_net_state_dict))

        logger.info('')

    def __safe_copy_state_dict(self,
                               original_state_dict: dict[torch.nn.Module],
                               loaded_state_dict: dict[torch.nn.Module]
                               ):
        for k, v in loaded_state_dict.items():
            if k in original_state_dict:
                if isinstance(v, torch.Tensor):
                    if original_state_dict[k].shape == v.shape:
                        original_state_dict[k] = loaded_state_dict[k]
                        logger.info(f'Layer {k} loaded successfully')
                    else:
                        logger.warning(f'problem loading layer {k} ' +
                                       f'{loaded_state_dict[k].shape=} but {original_state_dict[k].shape=}')
                        min_shape = [min(a, b) for a, b in zip(
                            list(original_state_dict[k].shape), list(v.shape))]

                        logger.warning(f'problem loading layer {k} ' +
                                       f'{list(loaded_state_dict[k].shape)=} but ' +
                                       f'{list(original_state_dict[k].shape)=} and ' +
                                       f'{min_shape=}')

                        if len(min_shape) == 1:
                            original_state_dict[k][:min_shape[0]
                                                   ] = loaded_state_dict[k][:min_shape[0]]

                        elif len(min_shape) == 2:
                            original_state_dict[k][:min_shape[0],
                                                   :min_shape[1]] = loaded_state_dict[k][:min_shape[0],
                                                                                         :min_shape[1]]

                        elif len(min_shape) == 3:
                            original_state_dict[k][:min_shape[0],
                                                   :min_shape[1],
                                                   :min_shape[2]] = loaded_state_dict[k][:min_shape[0],
                                                                                         :min_shape[1],
                                                                                         :min_shape[2]]

                        elif len(min_shape) == 4:
                            original_state_dict[k][:min_shape[0],
                                                   :min_shape[1],
                                                   :min_shape[2],
                                                   :min_shape[3]] = loaded_state_dict[k][:min_shape[0],
                                                                                         :min_shape[1],
                                                                                         :min_shape[2],
                                                                                         :min_shape[3]]

                        else:
                            raise BaseException(
                                f'Unknown min shape [{min_shape}], ' +
                                f'loaded_dict_shape is: {loaded_state_dict[k].shape} ' +
                                f'but original_dict shape is: {original_state_dict[k].shape}')
                elif isinstance(v, dict):
                    # TODO: fix this for optimizers
                    pass
            else:
                logger.warning(f'model doesn\'t have {k} layer')

        return original_state_dict
