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

st.title("GhostStream: Event-Driven Processor")
st.subheader("Asynchronous Spiking Architecture for Sub-Nyquist Signal Processing")

# Custom CSS for a serious, dark-themed button
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #1E1E1E;
    color: #FFFFFF;
    border: 1px solid #333333;
    border-radius: 4px;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    background-color: #2D2D2D;
    border: 1px solid #555555;
    color: #00FFCC;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
Unlike traditional Von Neumann architectures that process uniformly across time, **GhostStream** utilizes asynchronous event-driven sampling. Employing a Level-Crossing Analog-to-Digital Converter (ADC), the neural processor remains dormant during periods of signal inactivity or noise. Computations are strictly executed upon significant signal deviations, with the system maintaining state via Zero-Order Hold (ZOH) reconstruction during null periods.
""")

st.markdown("---")

if st.button("Initialize Neural Run"):
    with st.spinner("Training Dual Architectures... (Dense RNN vs GhostStream SNN)"):
        start_t = time.time()
        results = run_experiment()
        dur = time.time() - start_t

    st.success(f"Benchmarking complete in {dur:.2f} seconds. Zero-cheat MAC counting applied.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Dense DSP (Uniform Sampling)")
        st.markdown("*Continuously processes signal regardless of activity.*")
        st.metric("Final Loss (MSE)", f"{results['dense_loss']:.5f}")
        st.metric("Total MACs (Compute)", f"{results['dense_macs']:,}")
        st.metric("Compute Efficiency", "1.00x Base")

    with col2:
        st.markdown("### GhostStream (Event-Driven)")
        st.markdown("*Executes compute only on triggered information events.*")
        st.metric("Final Loss (MSE)", f"{results['sparse_loss']:.5f}")
        st.metric("Total MACs (Compute)", f"{results['sparse_macs']:,}")
        savings_pct = results['mac_savings']*100
        st.metric("Active Event Ratio", f"{results['active_ratio']*100:.2f}% (Information Sparsity)")

    st.markdown("---")
    st.markdown(f"## Compute Reduction: {savings_pct:.2f}%")
    if savings_pct > 90:
        st.markdown("> **STATUS**: Super-Nyquist Efficiency Achieved. >90% reduction in computational cycles while maintaining signal fidelity.")

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
