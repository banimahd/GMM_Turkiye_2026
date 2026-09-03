"""
GMM TURKIYE 2026 - STREAMLIT APP
===================================
Ground Motion Model for Turkiye (2026)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="GMM Turkiye 2026",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
    
    .main-header {
        font-family: 'Times New Roman', serif;
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-family: 'Times New Roman', serif;
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .result-box {
        font-family: 'Times New Roman', serif;
        background-color: #e8f0fe;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #d0d0d0;
        font-size: 1.1rem;
        text-align: right;
    }
    .result-label {
        font-family: 'Times New Roman', serif;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stButton button {
        font-family: 'Times New Roman', serif;
        background-color: #2196F3;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #1976D2;
    }
    .stSlider label, .stSelectbox label {
        font-family: 'Times New Roman', serif;
        font-size: 1.1rem !important;
        font-weight: bold !important;
    }
    .stNumberInput label {
        font-family: 'Times New Roman', serif;
        font-size: 1.1rem !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# GMM PREDICTOR CLASS
# ============================================================================

class GMMPredictor:
    """GMM Turkiye 2026 Predictor using extracted weights"""

    def __init__(self, weights_file='weights_GMM_Turkie_2026.mat'):
        """Initialize the predictor by loading weights from MATLAB file."""
        self.weights = []
        self.param_max = np.array([100, 3, 8, 200, 1500])
        self.param_min = np.array([0, 1, 4, 0, 100])
        self.is_loaded = False

        try:
            if not os.path.exists(weights_file):
                st.warning(f"Weights file not found: {weights_file}")
                return

            data = sio.loadmat(weights_file)

            if "weights_python" in data:
                self.weights = data["weights_python"][0]
            else:
                for key in data.keys():
                    if 'weight' in key.lower() and not key.startswith('__'):
                        self.weights = data[key][0]
                        break

            if "Param_Max_save" in data:
                self.param_max = data["Param_Max_save"].flatten()
                self.param_min = data["Param_Min_save"].flatten()
            elif "Param_Max" in data:
                self.param_max = data["Param_Max"].flatten()
                self.param_min = data["Param_Min"].flatten()

            if len(self.weights) > 0:
                self.is_loaded = True
                st.success(f"✅ Loaded {len(self.weights)} networks")

        except Exception as e:
            st.error(f"Error loading weights: {e}")

    @staticmethod
    def mapminmax_apply(x, xmin, xmax, ymin=-1, ymax=1):
        """Apply mapminmax preprocessing (scales to [-1, 1])"""
        range_x = xmax - xmin
        range_x[range_x == 0] = 1e-10
        if np.isscalar(range_x) or range_x.size == 1:
            range_x = max(range_x, 1e-10)
        else:
            range_x[range_x == 0] = 1e-10
        return (ymax - ymin) * (x - xmin) / range_x + ymin

    @staticmethod
    def mapminmax_reverse(y, xmin, xmax, ymin=-1, ymax=1):
        """Reverse mapminmax preprocessing"""
        range_x = xmax - xmin
        range_x[range_x == 0] = 1e-10
        if np.isscalar(range_x) or range_x.size == 1:
            range_x = max(range_x, 1e-10)
        else:
            range_x[range_x == 0] = 1e-10
        return (y - ymin) * range_x / (ymax - ymin) + xmin

    def normalize_input(self, input_raw):
        """Normalize input using GMM formula"""
        range_x = self.param_max - self.param_min
        range_x[range_x == 0] = 1e-10
        return 0.60 * (input_raw - self.param_min.reshape(-1, 1)) / range_x.reshape(-1, 1) + 0.20

    def predict(self, fd, fm, mw, rjb, vs30):
        """Predict ground motion parameters."""
        if not self.is_loaded:
            return self._get_default_outputs()

        input_raw = np.array([fd, fm, mw, rjb, vs30], dtype=np.float64).reshape(-1, 1)
        input_norm = self.normalize_input(input_raw)

        predictions = []

        for net in self.weights:
            try:
                w1 = net["fc1_Weights"][0, 0]
                b1 = net["fc1_Bias"][0, 0]
                w2 = net["fc5_Weights"][0, 0]
                b2 = net["fc5_Bias"][0, 0]

                w1 = np.array(w1, dtype=np.float64)
                b1 = np.array(b1, dtype=np.float64).reshape(-1, 1)
                w2 = np.array(w2, dtype=np.float64)
                b2 = np.array(b2, dtype=np.float64).reshape(-1, 1)

                xmin = np.array(net["input_xmin"][0, 0], dtype=np.float64).flatten().reshape(-1, 1)
                xmax = np.array(net["input_xmax"][0, 0], dtype=np.float64).flatten().reshape(-1, 1)
                ymin = np.array(net["input_ymin"][0, 0], dtype=np.float64).flatten()
                ymax = np.array(net["input_ymax"][0, 0], dtype=np.float64).flatten()

                xmin_out = np.array(net["output_xmin"][0, 0], dtype=np.float64).flatten().reshape(-1, 1)
                xmax_out = np.array(net["output_xmax"][0, 0], dtype=np.float64).flatten().reshape(-1, 1)
                ymin_out = np.array(net["output_ymin"][0, 0], dtype=np.float64).flatten()
                ymax_out = np.array(net["output_ymax"][0, 0], dtype=np.float64).flatten()

                input_proc = self.mapminmax_apply(input_norm, xmin, xmax, ymin, ymax)
                hidden = np.tanh(w1 @ input_proc + b1)
                output_proc = w2 @ hidden + b2
                output = self.mapminmax_reverse(output_proc, xmin_out, xmax_out, ymin_out, ymax_out)

                predictions.append(output.flatten())

            except Exception:
                continue

        if not predictions:
            return self._get_default_outputs()

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)

        results = {
            'PGA': round(np.exp(mean_pred[0]) * 986, 2),
            'PGV': round(np.exp(mean_pred[1]), 2),
            'Ia': round(np.exp(mean_pred[2]), 3),
            'D_5_75': round(np.exp(mean_pred[3]), 2),
            'D_5_95': round(np.exp(mean_pred[4]), 2),
            'T_m': round(np.exp(mean_pred[5]), 2),
            'CAV': round(np.exp(mean_pred[6]), 2),
            'PSa_003sec': round(np.exp(mean_pred[7]) * 986, 2),
            'PSa_005sec': round(np.exp(mean_pred[8]) * 986, 2),
            'PSa_0075sec': round(np.exp(mean_pred[9]) * 986, 2),
            'PSa_01sec': round(np.exp(mean_pred[10]) * 986, 2),
            'PSa_015sec': round(np.exp(mean_pred[11]) * 986, 2),
            'PSa_02sec': round(np.exp(mean_pred[12]) * 986, 2),
            'PSa_025sec': round(np.exp(mean_pred[13]) * 986, 2),
            'PSa_03sec': round(np.exp(mean_pred[14]) * 986, 2),
            'PSa_04sec': round(np.exp(mean_pred[15]) * 986, 2),
            'PSa_05sec': round(np.exp(mean_pred[16]) * 986, 2),
            'PSa_075sec': round(np.exp(mean_pred[17]) * 986, 2),
            'PSa_10sec': round(np.exp(mean_pred[18]) * 986, 2),
            'PSa_15sec': round(np.exp(mean_pred[19]) * 986, 2),
            'PSa_20sec': round(np.exp(mean_pred[20]) * 986, 2),
            'PSa_25sec': round(np.exp(mean_pred[21]) * 986, 2),
            'PSa_30sec': round(np.exp(mean_pred[22]) * 986, 2),
            'PSa_35sec': round(np.exp(mean_pred[23]) * 986, 2),
            'PSa_40sec': round(np.exp(mean_pred[24]) * 986, 2),
        }

        return results

    @staticmethod
    def _get_default_outputs():
        """Return default outputs when weights are not available."""
        return {
            'PGA': 0.0, 'PGV': 0.0, 'Ia': 0.0,
            'D_5_75': 0.0, 'D_5_95': 0.0, 'T_m': 0.0, 'CAV': 0.0,
            'PSa_003sec': 0.0, 'PSa_005sec': 0.0, 'PSa_0075sec': 0.0,
            'PSa_01sec': 0.0, 'PSa_015sec': 0.0, 'PSa_02sec': 0.0,
            'PSa_025sec': 0.0, 'PSa_03sec': 0.0, 'PSa_04sec': 0.0,
            'PSa_05sec': 0.0, 'PSa_075sec': 0.0, 'PSa_10sec': 0.0,
            'PSa_15sec': 0.0, 'PSa_20sec': 0.0, 'PSa_25sec': 0.0,
            'PSa_30sec': 0.0, 'PSa_35sec': 0.0, 'PSa_40sec': 0.0,
        }


# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def plot_spectra(results):
    """Plot spectral acceleration vs period."""
    periods = np.array([0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                        0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4])

    psa_keys = ['PSa_003sec', 'PSa_005sec', 'PSa_0075sec', 'PSa_01sec',
                'PSa_015sec', 'PSa_02sec', 'PSa_025sec', 'PSa_03sec',
                'PSa_04sec', 'PSa_05sec', 'PSa_075sec', 'PSa_10sec',
                'PSa_15sec', 'PSa_20sec', 'PSa_25sec', 'PSa_30sec',
                'PSa_35sec', 'PSa_40sec']

    y_vals = np.array([results[k] for k in psa_keys])
    y_vals = np.maximum(y_vals, 1e-10)

    # Set font to Times New Roman for all text
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['mathtext.fontset'] = 'stix'

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(periods, y_vals, 'b-', linewidth=2.5, label='GMM Turkiye 2026')

    ax.set_xlabel('T (s)', fontsize=16, fontweight='bold')
    ax.set_ylabel('PSa (cm/s²)', fontsize=16, fontweight='bold')

    ax.set_xticks([0.03, 0.1, 1, 4])
    ax.set_xticklabels(['0.03', '0.1', '1', '4'], fontsize=14)
    ax.set_xlim(0.03, 4)

    min_val = np.min(y_vals) * 0.8 if np.min(y_vals) > 0 else 0.001
    max_val = np.max(y_vals) * 1.5
    ax.set_ylim(min_val, max_val)

    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=12, width=1, length=4)

    ax.grid(True, which='both', linestyle='--', alpha=0.5, linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_facecolor('#f8f9fa')
    ax.legend(loc='best', frameon=False, fontsize=14)

    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown('<div class="main-header">📊 GMM Turkiye 2026</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;margin-bottom:2rem;font-family:Times New Roman,serif;font-size:1.1rem;">Ground Motion Model for Turkiye</div>', unsafe_allow_html=True)

    # Initialize predictor in session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = GMMPredictor('weights_GMM_Turkie_2026.mat')

    predictor = st.session_state.predictor

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📁 Batch Processing", "ℹ️ About"])

    # ========================================================================
    # TAB 1: Single Prediction
    # ========================================================================
    with tab1:
        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.markdown('<div class="sub-header">📥 Input Parameters</div>', unsafe_allow_html=True)

            with st.container(border=True):
                mw = st.slider(
                    "Mw",
                    min_value=4.0,
                    max_value=7.8,
                    value=7.0,
                    step=0.1,
                    help="Magnitude (4.0 - 7.8)"
                )

                rjb = st.slider(
                    "RJB (km)",
                    min_value=0.1,
                    max_value=200.0,
                    value=10.0,
                    step=0.1,
                    help="Joyner-Boore Distance (0.1 - 200 km)"
                )

                vs30 = st.slider(
                    "VS30 (m/s)",
                    min_value=131.0,
                    max_value=1380.0,
                    value=360.0,
                    step=1.0,
                    help="Shear Wave Velocity (131 - 1380 m/s)"
                )

                fd = st.slider(
                    "FD (km)",
                    min_value=0.0,
                    max_value=100.0,
                    value=3.0,
                    step=0.5,
                    help="Fault Distance (0 - 100 km)"
                )

                fm = st.selectbox(
                    "FM",
                    options=['Normal', 'Reverse', 'Strike Slip'],
                    index=1,
                    help="Fault Mechanism"
                )

                fm_map = {'Normal': 1, 'Reverse': 2, 'Strike Slip': 3}
                fm_val = fm_map[fm]

                run_button = st.button("▶ RUN PREDICTION", use_container_width=True)

        with col2:
            st.markdown('<div class="sub-header">📤 Prediction Results</div>', unsafe_allow_html=True)

            if run_button:
                with st.spinner("Predicting..."):
                    results = predictor.predict(fd, fm_val, mw, rjb, vs30)

                    # Display results in single column
                    # 1. PGA
                    st.markdown('<div class="result-label">PGA (cm/s²)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box">{results["PGA"]:.2f}</div>', unsafe_allow_html=True)

                    # 2. PGV
                    st.markdown('<div class="result-label">PGV (cm/s)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box">{results["PGV"]:.2f}</div>', unsafe_allow_html=True)

                    # 3. CAV
                    st.markdown('<div class="result-label">CAV (cm/s)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box">{results["CAV"]:.2f}</div>', unsafe_allow_html=True)

                    # 4. Ia
                    st.markdown('<div class="result-label">Ia (cm/s)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box">{results["Ia"]:.3f}</div>', unsafe_allow_html=True)

                    # 5. Tm
                    st.markdown('<div class="result-label">Tm (s)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box">{results["T_m"]:.2f}</div>', unsafe_allow_html=True)

                    # 6. D 5%-75%
                    st.markdown('<div class="result-label">D 5%-75% (s)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box">{results["D_5_75"]:.2f}</div>', unsafe_allow_html=True)

                    # 7. D 5%-95%
                    st.markdown('<div class="result-label">D 5%-95% (s)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-box">{results["D_5_95"]:.2f}</div>', unsafe_allow_html=True)

                # Plot spectrum
                st.markdown('<div class="sub-header">📈 Spectral Acceleration vs Period</div>', unsafe_allow_html=True)
                fig = plot_spectra(results)
                st.pyplot(fig)

    # ========================================================================
    # TAB 2: Batch Processing
    # ========================================================================
    with tab2:
        st.markdown('<div class="sub-header">📁 Batch Processing</div>', unsafe_allow_html=True)

        st.info("""
        Upload an Excel file with the following columns (in this order):
        - **Mw** (Magnitude)
        - **VS30** (Shear Wave Velocity)
        - **RJB** (Joyner-Boore Distance)
        - **FD** (Fault Distance)
        - **FM** (Fault Mechanism: 1=Normal, 2=Reverse, 3=Strike Slip)
        """)

        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)

                st.write("**Preview of uploaded data:**")
                st.dataframe(df.head(), use_container_width=True)

                if st.button("📊 Process Excel File", use_container_width=True):
                    with st.spinner("Processing..."):
                        # Process each row
                        results_list = []
                        for _, row in df.iterrows():
                            try:
                                mw = float(row.iloc[0]) if len(row) > 0 else 0
                                vs30 = float(row.iloc[1]) if len(row) > 1 else 0
                                rjb = float(row.iloc[2]) if len(row) > 2 else 0
                                fd = float(row.iloc[3]) if len(row) > 3 else 0
                                fm = int(row.iloc[4]) if len(row) > 4 else 1

                                results = predictor.predict(fd, fm, mw, rjb, vs30)
                                results_list.append(results)
                            except Exception:
                                results_list.append(GMMPredictor._get_default_outputs())

                        # Create output DataFrame
                        output_cols = ['Mw', 'VS30', 'RJB', 'FD', 'FM',
                                       'PGA', 'PGV', 'CAV', 'Ia', 'Tm', 'D 5-75', 'D 5-95',
                                       'PSa_003sec', 'PSa_005sec', 'PSa_0075sec', 'PSa_01sec',
                                       'PSa_015sec', 'PSa_02sec', 'PSa_025sec', 'PSa_03sec',
                                       'PSa_04sec', 'PSa_05sec', 'PSa_075sec', 'PSa_10sec',
                                       'PSa_15sec', 'PSa_20sec', 'PSa_25sec', 'PSa_30sec',
                                       'PSa_35sec', 'PSa_40sec']

                        out_data = []
                        for i, row in df.iterrows():
                            r = results_list[i] if i < len(results_list) else GMMPredictor._get_default_outputs()
                            out_row = [
                                row.iloc[0] if len(row) > 0 else 0,
                                row.iloc[1] if len(row) > 1 else 0,
                                row.iloc[2] if len(row) > 2 else 0,
                                row.iloc[3] if len(row) > 3 else 0,
                                row.iloc[4] if len(row) > 4 else 1,
                                r['PGA'], r['PGV'], r['CAV'], r['Ia'], r['T_m'], r['D_5_75'], r['D_5_95'],
                                r['PSa_003sec'], r['PSa_005sec'], r['PSa_0075sec'], r['PSa_01sec'],
                                r['PSa_015sec'], r['PSa_02sec'], r['PSa_025sec'], r['PSa_03sec'],
                                r['PSa_04sec'], r['PSa_05sec'], r['PSa_075sec'], r['PSa_10sec'],
                                r['PSa_15sec'], r['PSa_20sec'], r['PSa_25sec'], r['PSa_30sec'],
                                r['PSa_35sec'], r['PSa_40sec']
                            ]
                            out_data.append(out_row)

                        out_df = pd.DataFrame(out_data, columns=output_cols)

                        st.success("✅ Processing complete!")

                        st.write("**Results:**")
                        st.dataframe(out_df, use_container_width=True)

                        csv = out_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="GMM_Turkiye_2026_Results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ========================================================================
    # TAB 3: About
    # ========================================================================
    with tab3:
        st.markdown('<div class="sub-header">ℹ️ About</div>', unsafe_allow_html=True)

        st.markdown("""
        ### GMM Turkiye 2026

        Ground Motion Model for Turkiye (2026)

        **Input Parameters:**
        - **Mw**: Magnitude (4.0 - 7.8)
        - **RJB**: Joyner-Boore Distance (0.1 - 200 km)
        - **VS30**: Shear Wave Velocity (131 - 1380 m/s)
        - **FD**: Fault Distance (0 - 100 km)
        - **FM**: Fault Mechanism (Normal, Reverse, Strike Slip)

        **Output Parameters:**
        - **PGA**: Peak Ground Acceleration (cm/s²)
        - **PGV**: Peak Ground Velocity (cm/s)
        - **CAV**: Cumulative Absolute Velocity (cm/s)
        - **Ia**: Arias Intensity (cm/s)
        - **Tm**: Mean Period (s)
        - **D 5%-75%**: Duration between 5% and 75% (s)
        - **D 5%-95%**: Duration between 5% and 95% (s)
        - **PSa**: Spectral Accelerations at 18 periods (cm/s²)

        **Developer:** Amir Banimahd
        """)

        with st.expander("📊 Network Architecture"):
            st.markdown("""
            - **Input Layer:** 5 parameters
            - **Hidden Layer:** 15 neurons (tanh activation)
            - **Output Layer:** 25 parameters
            - **Ensemble:** 10 networks
            """)


if __name__ == "__main__":
    main()
