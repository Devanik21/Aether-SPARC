class AetherSparcNet(nn.Module):
    def __init__(self, hidden=128, threshold=0.045, tau=20.0):
        super().__init__()
        # Threshold at 0.045 acts as a 3-sigma gate against the 0.01 std background noise.
        # tau is the neuromorphic synaptic leakage time constant.
        self.threshold = threshold
        self.tau = tau
        
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. Asynchronous Event Generation (Neuromorphic Delta Modulation)
        # The diff filter organically blocks Gaussian background noise.
        diff = torch.abs(x - torch.roll(x, 1, 0))
        mask = (diff > self.threshold).float()
        mask[0] = 1.0  # Synaptic initialization: force system to establish base state at t=0

        # Extract strictly active event indices
        active_indices = mask.squeeze().nonzero(as_tuple=False).squeeze()

        if active_indices.numel() == 0:
            return torch.zeros_like(x), 0

        # 2. Sparse Compute (Execute heavy MACs ONLY on events)
        x_active = x[active_indices]
        out_active = self.relu(self.fc1(x_active))
        out_active = self.relu(self.fc2(out_active))
        out_active = self.fc3(out_active)

        # 3. Neuromorphic Leaky Zero-Order Hold (ZOH)
        # We use cumsum to map non-computed timestamps back to their last computed event.
        # This routes gradients perfectly through time with 0 cheat.
        fill_indices = (torch.cumsum(mask, dim=0) - 1).long().squeeze()

        # Calculate exact time elapsed since the last fired event
        t_idx = torch.arange(len(x), device=x.device).float()
        last_event_t = t_idx[fill_indices]
        time_since_event = t_idx - last_event_t

        # Synaptic leakage: state naturally decays to resting potential (0) during silence
        decay = torch.exp(-time_since_event / self.tau).unsqueeze(1)

        # Reconstruct the continuous signal organically without extra MACs
        output = out_active[fill_indices] * decay

        return output, len(active_indices)
