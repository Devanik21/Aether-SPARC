"""Predictive coding mechanisms for neuromorphic state update transitions."""

def predict(model, state):
    return model.forward(state)
