from typing import Any


class Relu():
    def __call__(self, inp):
        """
        Apply the function to the input tensor.

        Parameters:
            inp (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the function.
        """
        self.input = inp
        self.out = inp.clamp(min=0.)
        return self.out 
    
    def backward(self):
        """
        Set the gradients for the backward pass.

        Args:
            None

        Returns:
            None
        """
        self.input.g = (self.input > 0).float() * self.out.g
        
class Lin():
    def __init__(self, w, b):
        """
        Initializes a new instance of the class.

        Args:
            w (int): The value to assign to the `w` attribute.
            b (int): The value to assign to the `b` attribute.
        """
        self.w = w
        self.b = b
        
    def __call__(self, inp):
        """
        Calculates the output of the function based on the given input.

        Parameters:
            inp (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The output array.
        """
        self.input = inp
        self.out = inp@self.w + self.b
        return self.out
    
    
    def backward(self):
        """
        Compute the gradients for the backward pass of the neural network.

        This function updates the gradients for the input, weight, and bias tensors of the neural network.
        It uses the chain rule to compute the gradients by multiplying the output gradients with the
        transposed weights and input tensors.

        Parameters:
            None

        Returns:
            None
        """
        self.input.g = self.out.g @ self.w.t()
        self.w.g = self.input.t() @ self.out.g
        self.b.g = self.out.g.sum(0)
        
class Mse():
    def __call__(self, inp, targ):
        """
        Calculate the mean squared error between the input and target tensors.

        Parameters:
            inp (torch.Tensor): The input tensor.
            targ (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The mean squared error between the input and target tensors.
        """
        self.inp = inp
        self.targ = targ
        self.out = (inp.squeeze() - targ).pow(2).mean()
        return self.out
    
    def backward(self):
        """
        Compute the backward pass of the model.

        Parameters:
        - self: The instance of the class.
        
        Returns:
        - None
        """
        self.inp.g = 2. * (self.inp.squeeze() - self.targ).unsqueeze(-1) / self.targ.shape[0]
        


class Model():
    def __init__(self, w1, b1, w2, b2):
        """
        Initializes the object with the given weights and biases.

        Parameters:
            w1 (numpy.ndarray): The weight matrix for the first linear layer.
            b1 (numpy.ndarray): The bias vector for the first linear layer.
            w2 (numpy.ndarray): The weight matrix for the second linear layer.
            b2 (numpy.ndarray): The bias vector for the second linear layer.
        """
        self.layers = [Lin(w1, b1), Relu(), Lin(w2, b2)]
        self.loss = Mse()
        
    def __call__(self, x, targ):
        """
        Calls the layers sequentially on the input tensor and computes the loss between the output tensor and the target tensor.

        Args:
            x (tensor): The input tensor.
            targ (tensor): The target tensor.

        Returns:
            tensor: The loss between the output tensor and the target tensor.
        """
        for l in self.layers:
            x = l(x)
        return self.loss(x, targ)
    
    def backward(self):
        """
        Backward pass through the neural network.

        This function performs the backward pass through the neural network by 
        calling the backward method of each layer in reverse order. The backward 
        method of each layer computes the gradients of the layer's parameters and 
        the gradients of the input with respect to the loss. Finally, the backward 
        method of the loss function is called to compute the gradients of the input 
        with respect to the loss.

        Parameters:
            None

        Returns:
            None
        """
        self.loss.backward()
        for l in reversed(self.layers):
            l.backward()
            
    def parameters(self):
        """
        Yield the parameters of all layers in the neural network.

        Returns:
            A generator object that yields the parameters of each layer.
        """
        for l in self.layers:
            yield from l.parameters()
            
    # def grad_parameters(self):
    #     for l in self.layers:
    #         yield from l.grad_parameters()
    
class Module():
    def __call__(self, *args):
        """
        Calls the function with the given arguments and returns the output.

        Args:
            *args: The arguments to pass to the function.

        Returns:
            The output of the function.
        """
        self.args = args
        self.out = self.forward(*args)
        return self.out
    
    def forward(self):
        raise Exception('not implemented')
        
    def backward(self):
        self.bwd(self.out, *self.args)
        
    def bwd(self):
        raise Exception('not implemented')
        
    def parameters(self):
        return []
    
    def grad_parameters(self):
        return []
    
class Relu(Module):
    def forward(self, inp):
        return inp.clamp(min=0.)
    
    def bwd(self, out, inp):
        inp.g = (inp > 0).float() * out.g
        
class Lin(Module):
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
    def forward(self, inp):
        return inp@self.w + self.b
    
    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ out.g
        self.b.g = out.g.sum(0)
        
    # 
    
class Mse(Module):
    def forward(self, inp, targ):
        return (inp.squeeze() - targ).pow(2).mean()
    
    def bwd(self, out, inp, targ):
        inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / targ.shape[0]