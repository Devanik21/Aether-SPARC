"""Predictive coding mechanisms for neuromorphic state update transitions."""

def predict(model: "torch.nn.Module", state: "torch.Tensor") -> "torch.Tensor":
    return model.forward(state)
