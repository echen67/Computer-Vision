import torch
import torch.nn as nn

from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
    to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one

    Download pretrained alexnet using pytorch's API (Hint: see the import
    statements)
    '''
    # super(MyAlexNet, self).__init__()
    super().__init__()                      # REVERT BACK LATER!!!!!!!!!!!!!!

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################

    # freezing the layers by setting requires_grad=False
    # example: self.cnn_layers[idx].weight.requires_grad = False

    # take care to turn off gradients for both weight and bias

    model = alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    children = list(model.children())
    alex_cnn = children[0]
    alex_fc = children[2][:-1]

    self.cnn_layers = alex_cnn
    self.fc_layers = nn.Sequential(
        nn.Flatten(),
        *list(alex_fc.children()),
        nn.Linear(in_features=4096, out_features=15, bias=True),
    )
    
    self.loss_criterion = nn.CrossEntropyLoss(reduction="sum")

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
    x = x.repeat(1, 3, 1, 1) # as AlexNet accepts color images

    ###########################################################################
    # Student code begin
    ###########################################################################

    cnn = self.cnn_layers(x)
    model_output = self.fc_layers(cnn)

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output
