import torch
import torch.nn as nn


class Sigmoid(nn.Module):
    """
    Sigmoid activation function
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the sigmoid activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return 1 / (1 + torch.exp(-x))

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derivative of the sigmoid activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.forward(x) * (1 - self.forward(x))


class Softmax(nn.Module):
    """
    Softmax activation function
    """

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the softmax activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derivative of the softmax activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.forward(x) * (1 - self.forward(x))


class ReLU(nn.Module):
    """
    ReLU activation function
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ReLU activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return torch.max(torch.tensor(0), x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derivative of the ReLU activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return torch.where(x > 0, torch.tensor(1), torch.tensor(0))


class Tanh(nn.Module):
    """
    Tanh activation function
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Tanh activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derivative of the Tanh activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return 1 - self.forward(x) ** 2


class LeakyReLU(nn.Module):
    """
    Leaky ReLU activation function
    """

    def __init__(self, alpha: float = 0.01):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Leaky ReLU activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return torch.max(self.alpha * x, x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derivative of the Leaky ReLU activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return torch.where(x > 0, torch.tensor(1), torch.tensor(self.alpha))


class ELU(nn.Module):
    """
    ELU activation function
    """

    def __init__(self, alpha: float = 1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ELU activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derivative of the ELU activation function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return torch.where(x > 0, torch.tensor(1), self.forward(x) + self.alpha)
