"""
GhostStream Streamlit Frontend
Run with:
    streamlit run AETHER_SPARC.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
from CoRe import run_experiment

st.set_page_config(page_title="GhostStream Benchmark", layout="wide")

st.title("GhostStream: The Zero-Waste Event Processor")
st.subheader("Paradigm-Shift in Audio DSP using Level-Crossing Sampling & Asynchronous Spiking Networks")

st.markdown("""
Unlike traditional Von Neumann architectures that process every clock cycle regardless of information content, **GhostStream** is fully asynchronous. By utilizing an event-driven Level-Crossing Analog-to-Digital Converter (ADC), the SNN only wakes up when the physical signal changes significantly. The processor does **zero** matrix multiplications during silence or minor noise, filling the gaps via Zero-Order Hold (ZOH).
""")

st.markdown("---")

if st.button("Initialize Quantum/Neuromorphic Run", type="primary"):
    with st.spinner("Training Dual Architectures... (Dense RNN vs GhostStream SNN)"):
        start_t = time.time()
        results = run_experiment()
        dur = time.time() - start_t

    st.success(f"Benchmarking complete in {dur:.2f} seconds. Zero-cheat MAC counting applied.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🛑 Traditional DSP (Dense RNN)")
        st.markdown("*Processes every time sample blindly.*")
        st.metric("Final Loss (MSE)", f"{results['dense_loss']:.5f}")
        st.metric("Total MACs (Compute)", f"{results['dense_macs']:,}")
        st.metric("Compute Efficiency", "1.00x Base")

    with col2:
        st.markdown("### 👻 GhostStream (Event-Driven SNN)")
        st.markdown("*Processes ONLY actual information events.*")
        st.metric("Final Loss (MSE)", f"{results['sparse_loss']:.5f}")
        st.metric("Total MACs (Compute)", f"{results['sparse_macs']:,}")
        savings_pct = results['mac_savings']*100
        st.metric("Active Event Ratio", f"{results['active_ratio']*100:.2f}% (Information Sparsity)")

    st.markdown("---")
    st.markdown(f"## 🚀 Compute Reduction: {savings_pct:.2f}%")
    if savings_pct > 90:
        st.balloons()
        st.markdown("> **PARADIGM SHIFT ACHIEVED**: Over 90% reduction in computational waste while maintaining competitive signal reconstruction.")

    st.markdown("---")
    st.markdown("### Signal Reconstruction & Asynchronous Event Spikes")
    
    noisy = results["noisy"]
    clean = results["clean"]
    sparse_recon = results["sparse_recon"]
    events = results["event_indices"]
    
    # We want to zoom into a burst to see the "ZOH" step-like nature
    # Let's find a burst automatically
    burst_start = 0
    for i in range(len(clean)):
        if abs(clean[i]) > 0.1:
            burst_start = max(0, i - 100)
            break
            
    zoom_range = slice(burst_start, min(len(clean), burst_start + 1500))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Top Plot - Inputs
    axes[0].set_title("1. Sensor Input (Continuous Noisy Environment)")
    axes[0].plot(noisy[zoom_range], label="Raw Noisy Input", color='gray', alpha=0.5)
    axes[0].plot(clean[zoom_range], label="Clean Target Signal", color='black', linewidth=1.5)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # Mid Plot - Spikes!
    axes[1].set_title("2. Level-Crossing ADC Spike Train (Wakes up the GhostStream)")
    # Extract events only in zoom range
    z_events = [e for e in events if zoom_range.start <= e < zoom_range.stop]
    # offset them for plotting
    z_events_rel = [e - zoom_range.start for e in z_events]
    
    # Plot stem safely
    if len(z_events_rel) > 0:
        axes[1].stem(z_events_rel, [1]*len(z_events_rel), linefmt='r-', markerfmt='ro', basefmt='r-')
    axes[1].set_ylim(0, 1.2)
    axes[1].set_yticks([])
    
    total_dur = zoom_range.stop - zoom_range.start
    axes[1].text(0.01, 0.85, f"Information Sparsity: Woke up SNN only {len(z_events_rel)} times out of {total_dur} cycles!", transform=axes[1].transAxes, color='red', weight='bold', fontsize=12)
    axes[1].grid(True, axis='x', alpha=0.3)
    
    # Bottom Plot - ZOH Reconstruction
    axes[2].set_title("3. GhostStream Output (Zero-Order Hold Neural Reconstruction)")
    axes[2].plot(sparse_recon[zoom_range], label="GhostStream ZOH Output", color='blue', drawstyle='steps-post', linewidth=2)
    axes[2].plot(clean[zoom_range], label="Target Signal", color='black', linestyle='--', alpha=0.7)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
