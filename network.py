import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor):
        x = self.shared(x)
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_probs, state_values
    
class ActorCriticNetworkCNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(ActorCriticNetworkCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        self.shared = nn.Sequential(
            nn.Linear(self.get_conv_output_size(3), 512),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(512, output_dim)
        self.critic = nn.Linear(512, 1)
    
    def get_conv_output_size(self, input_channels):
        dummy_input = torch.randn(1, input_channels, 250, 160)
        output_size = self.conv_layers(dummy_input).size()[1]
        return output_size
    
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = self.shared(x)
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_probs, state_values