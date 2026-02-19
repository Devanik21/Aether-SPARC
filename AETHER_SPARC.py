"""
AETHER-SPARC Streamlit Frontend

Run with:
    streamlit run AETHER_SPARC_app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
from CoRe import run_experiment, SyntheticBurstAudio


st.set_page_config(page_title="AETHER-SPARC Benchmark", layout="wide")

st.title("AETHER-SPARC")
st.subheader("An Asynchronous Event-Triggered Sparse Proportional Compute Architecture")

st.markdown("---")

if st.button("Run Strict Fair Benchmark"):
    with st.spinner("Running identical training for DenseDSP and AETHER-SPARC..."):
        results = run_experiment()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### DenseDSP")
        st.metric("Loss", f"{results['dense_loss']:.6f}")
        st.metric("MACs", f"{results['dense_macs']:,}")

    with col2:
        st.markdown("### AETHER-SPARC")
        st.metric("Loss", f"{results['sparse_loss']:.6f}")
        st.metric("MACs", f"{results['sparse_macs']:,}")
        st.metric("Active Event Ratio", f"{results['active_ratio']*100:.2f}%")

    st.markdown("---")
    st.success(f"MAC Reduction: {results['mac_savings']*100:.2f}%")

st.markdown("---")
st.markdown("### Example Generated Signal")

dataset = SyntheticBurstAudio()
noisy, clean = dataset.generate()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(noisy, label="Noisy Input", alpha=0.6)
ax.plot(clean, label="Clean Target", alpha=0.6)
ax.legend()

st.pyplot(fig)
