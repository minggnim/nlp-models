import os
import json
from typing import Union
import torch
from .utils import fullname, import_from_string


class Dropout(torch.nn.Module):
    '''
    Dropout layer
    Param dropout: Sets a dropout value for dense layer
    '''
    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(self.dropout)

    def forward(self, features: torch.Tensor):
        return self.dropout_layer(features)

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as f_out:
            json.dump({'dropout': self.dropout}, f_out)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as f_in:
            config = json.load(f_in)

        model = Dropout(**config)
        return model


class Dense(torch.nn.Module):
    '''
    One-layer feedforward neural network
    '''
    def __init__(
        self,
        input_features: int,
        output_features: int,
        bias: bool = True,
        activation_function=torch.nn.Sigmoid(),
        init_weights: Union[torch.Tensor, None] = None,
        init_bias: Union[torch.Tensor, None] = None
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.bias = bias
        self.activation_function = activation_function
        self.linear = torch.nn.Linear(input_features, output_features, bias)

        if init_weights is not None:
            self.linear.weight = torch.nn.Parameter(init_weights)
        if init_bias is not None:
            self.linear.bias = torch.nn.Parameter(init_bias)

    def forward(self, features: torch.Tensor):
        return self.activation_function(self.linear(features))

    def get_config_dict(self):
        return {
            'input_features': self.input_features,
            'output_features': self.output_features,
            'bias': self.bias,
            'activation_function': fullname(self.activation_function)
        }

    def save(self, output_path, file_name='pytorch_model.bin'):
        with open(os.path.join(output_path, 'config.json'), 'w') as f_out:
            json.dump(self.get_config_dict(), f_out)
        torch.save(self.state_dict(), os.path.join(output_path, file_name))

    def __repr__(self):
        return f'Dense({self.get_config_dict()})'

    @staticmethod
    def load(input_path, file_name='pytorch_model.bin'):
        with open(os.path.join(input_path, 'config.json')) as f_in:
            config = json.load(f_in)
        config['activation_function'] = import_from_string(config['activation_function'])()
        model = Dense(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, file_name), map_location=torch.device('cpu')))
