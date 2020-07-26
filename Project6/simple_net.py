import torch
import torch.nn as nn


class SimpleNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention
    to understand what it means
    '''
    # super(SimpleNet, self).__init__()
    super().__init__()                      # REVERT BACK LATER!!!!!!!!!!!!!!

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################

    self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
        )

    self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 15),
        )

    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    ###########################################################################
    # Student code end
    ###########################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ###########################################################################
    # Student code begin
    ###########################################################################

    cnn = self.cnn_layers(x)
    model_output = self.fc_layers(cnn)

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output
