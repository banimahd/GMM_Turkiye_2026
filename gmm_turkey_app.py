import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.io
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="GMM-Turkey: Ground Motion Predictor", layout="wide")


st.markdown("""
<style>
    .main-title {
        font-size: 3.0rem !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
        text-align: center !important;
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
        letter-spacing: -0.5px !important;
    }
    .sub-title {
        text-align: center !important;
        font-size: 1.1rem !important;
        color: #666666 !important;
        margin-top: -0.3rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: 400 !important;
    }
    .section-header {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #16213e !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .stSidebar .stMarkdown h1 {
        font-size: 1.5rem !important;
    }
    .stSidebar .stMarkdown h2 {
        font-size: 1.2rem !important;
    }
    .stSidebar .stNumberInput label {
        font-size: 1.0rem !important;
        font-weight: 500 !important;
    }
    .stButton button {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
    }
    .stDownloadButton button {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    .output-label {
        color: #000000 !important;
    }
    .output-value {
        color: #1565C0 !important;
        font-size: 2.0rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


class PINNModel(nn.Module):
    def __init__(self, layer_sizes):
        super(PINNModel, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_models_from_mat(weights_file):
    try:
        data = scipy.io.loadmat(weights_file)
        
        # Extract max/min values and network
        param_max = data['Param_Max'].flatten()
        param_min = data['Param_Min'].flatten()
        net_save = data['Net_Save'][0]
        
        models = []
        for i in range(len(net_save)):
            net_struct = net_save[i]
            
            # Extract weights and biases from MATLAB network
            # This assumes a typical MATLAB feedforwardnet structure
            try:
                # Get layer sizes
                input_size = net_struct['inputs'][0,0][0][0][0][0][0][0][0][0]
                layer_sizes = [input_size]
                
                # Extract weights and biases for each layer
                state_dict = {}
                for j in range(len(net_struct['layers'][0])):
                    layer = net_struct['layers'][0][j]
                    if 'weights' in layer.dtype.names and 'biases' in layer.dtype.names:
                        W = layer['weights'][0,0]
                        b = layer['biases'][0,0]
                        if W is not None and b is not None:
                            if isinstance(W, np.ndarray) and isinstance(b, np.ndarray):
                                state_dict[f'network.{j*2}.weight'] = torch.tensor(W.T, dtype=torch.float32)
                                state_dict[f'network.{j*2}.bias'] = torch.tensor(b.flatten(), dtype=torch.float32)
                                layer_sizes.append(W.shape[0])
                
                model = PINNModel(layer_sizes)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                models.append(model)
            except:
                continue
        
        return models, param_max, param_min
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


@st.cache_resource
def load_gmm_models():
    # Load from GitHub raw URL
    import urllib.request
    url = "https://raw.githubusercontent.com/banimahd/GMM_Turkiye_2025/main/GUI_All_Params/GMM_Turkie_2025.mat"
    
    try:
        urllib.request.urlretrieve(url, "GMM_Turkie_2025.mat")
        models, param_max, param_min = load_models_from_mat("GMM_Turkie_2025.mat")
        return models, param_max, param_min
    except:
        st.error("Could not load model file. Please check the file path.")
        return None, None, None


def normalize_inputs(inputs, param_min, param_max):
    return 0.60 * ((inputs - param_min) / (param_max - param_min)) + 0.20


def call_back_excel(mw, vs30, rjb, fd, fm, models, param_max, param_min):
    # Order: [FD; FM; Mw; RJB; VS30]
    inputs = np.array([fd, fm, mw, rjb, vs30], dtype=np.float32).reshape(-1, 1)
    
    # Normalize inputs
    inputs_norm = normalize_inputs(inputs, param_min.reshape(-1, 1), param_max.reshape(-1, 1))
    inputs_tensor = torch.tensor(inputs_norm.T, dtype=torch.float32)
    
    # Ensemble prediction (10 models)
    outputs_mean = np.zeros(25)
    for model in models:
        with torch.no_grad():
            output = model(inputs_tensor).numpy().flatten()
            outputs_mean += output / len(models)
    
    # Convert outputs (using MATLAB's exp and scaling)
    pga = round(np.exp(outputs_mean[0]) * 986, 3)
    pgv = round(np.exp(outputs_mean[1]), 3)
    ia = round(np.exp(outputs_mean[2]), 3)
    d5_75 = round(np.exp(outputs_mean[3]), 3)
    d5_95 = round(np.exp(outputs_mean[4]), 3)
    tm = round(np.exp(outputs_mean[5]), 3)
    cav = round(np.exp(outputs_mean[6]), 3)
    
    # PSa values (scaled by 986)
    psas = [round(np.exp(outputs_mean[i]) * 986, 5) for i in range(7, 25)]
    
    output = [pga, pgv, ia, d5_75, d5_95, tm, cav] + psas
    return np.array(output)


def plot_data(ax, outputs):
    periods = np.array([0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                        0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    
    # PSa values are from index 7 to 24
    y = np.array(outputs[7:25])
    y = np.maximum(y, 1e-10)
    
    ax.clear()
    ax.loglog(periods, y, 'b-', linewidth=2.5, label='Predicted PSa')
    
    ax.legend(loc='best', frameon=False, fontsize=18)
    ax.set_xlabel('T (s)', fontsize=24, fontweight='bold')
    ax.set_ylabel('PSa (cm/s²)', fontsize=24, fontweight='bold')
    
    ax.set_xticks([0.03, 0.1, 1, 4])
    ax.set_xticklabels(['0.03', '0.1', '1', '4'], fontsize=20)
    
    ax.tick_params(axis='y', labelsize=20)
    
    # Dynamic y-limits
    if np.any(y > 0):
        min_val = np.min(y[y > 0]) * 0.8
        max_val = np.max(y) * 1.5
        ax.set_ylim(min_val, max_val)
    else:
        ax.set_ylim(1e-4, 10)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.5, linewidth=0.8)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    ax.set_facecolor('#f8f9fa')


def process_excel_file(df, models, param_max, param_min):
    results = []
    data = df.values
    
    for i in range(len(data)):
        row = data[i]
        try:
            mw = float(row[0])
            vs30 = float(row[1])
            rjb = float(row[2])
            fd = float(row[3])
            fm = int(row[4])
            
            output = call_back_excel(mw, vs30, rjb, fd, fm, models, param_max, param_min)
            results.append(output)
        except:
            continue
    
    return np.array(results)


def main():
    st.markdown('<p class="main-title">GMM-Turkey</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ground Motion Model for Turkey (2025)</p>', unsafe_allow_html=True)
    
    with st.spinner("Loading GMM models..."):
        models, param_max, param_min = load_gmm_models()
        
        if models is None:
            st.error("Failed to load models. Please check the file path.")
            return
        
        st.success(f"✅ Loaded {len(models)} ensemble models")
    
    with st.sidebar:
        st.markdown('<p class="section-header">📥 Inputs</p>', unsafe_allow_html=True)
        
        mw = st.number_input("Mw", min_value=4.0, max_value=7.8, value=7.0, step=0.1)
        rjb = st.number_input("RJB (km)", min_value=0.1, max_value=200.0, value=30.0, step=0.1)
        vs30 = st.number_input("VS30 (m/s)", min_value=131.0, max_value=1380.0, value=360.0, step=10.0)
        fd = st.number_input("FD (km)", min_value=0.0, max_value=35.0, value=10.0, step=1.0)
        fm = st.selectbox("FM", options=['Normal', 'Reverse', 'Strike Slip'], index=2)
        
        if st.button("📊 Plot", type="primary", use_container_width=True):
            fm_map = {'Normal': 1, 'Reverse': 2, 'Strike Slip': 3}
            fm_val = fm_map[fm]
            
            with st.spinner("Calculating..."):
                output = call_back_excel(mw, vs30, rjb, fd, fm_val, models, param_max, param_min)
                
                st.session_state['output'] = output
                st.session_state['calculated'] = True
                
                # Store individual values
                names = ['PGA', 'PGV', 'Ia', 'D5-75', 'D5-95', 'Tm', 'CAV']
                for i, name in enumerate(names):
                    st.session_state[f'output_{name}'] = output[i]
        
        st.markdown("---")
        st.markdown('<p style="font-size:0.8rem; color:#888; text-align:center; margin-top:1rem;">Made by Amir Banimahd</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.4, 0.6])
    
    with col1:
        st.markdown('<p class="section-header">📤 Outputs</p>', unsafe_allow_html=True)
        
        # PGA, PGV, Ia
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            pga = st.session_state.get('output_PGA', 0)
            st.markdown(f'<p style="font-size:0.9rem; font-weight:600; margin:0;">PGA</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="output-value">{pga:.3f}</p>', unsafe_allow_html=True)
            st.caption("cm/s²")
        with col_a2:
            pgv = st.session_state.get('output_PGV', 0)
            st.markdown(f'<p style="font-size:0.9rem; font-weight:600; margin:0;">PGV</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="output-value">{pgv:.3f}</p>', unsafe_allow_html=True)
            st.caption("cm/s")
        with col_a3:
            ia = st.session_state.get('output_Ia', 0)
            st.markdown(f'<p style="font-size:0.9rem; font-weight:600; margin:0;">Ia</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="output-value">{ia:.3f}</p>', unsafe_allow_html=True)
            st.caption("cm/s")
        
        # D5-75, D5-95, Tm
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            d5_75 = st.session_state.get('output_D5-75', 0)
            st.markdown(f'<p style="font-size:0.9rem; font-weight:600; margin:0;">D5-75</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="output-value">{d5_75:.3f}</p>', unsafe_allow_html=True)
            st.caption("s")
        with col_b2:
            d5_95 = st.session_state.get('output_D5-95', 0)
            st.markdown(f'<p style="font-size:0.9rem; font-weight:600; margin:0;">D5-95</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="output-value">{d5_95:.3f}</p>', unsafe_allow_html=True)
            st.caption("s")
        with col_b3:
            tm = st.session_state.get('output_Tm', 0)
            st.markdown(f'<p style="font-size:0.9rem; font-weight:600; margin:0;">Tm</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="output-value">{tm:.3f}</p>', unsafe_allow_html=True)
            st.caption("s")
        
        # CAV
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            cav = st.session_state.get('output_CAV', 0)
            st.markdown(f'<p style="font-size:0.9rem; font-weight:600; margin:0;">CAV</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="output-value">{cav:.3f}</p>', unsafe_allow_html=True)
            st.caption("cm/s")
        
        st.markdown("---")
        st.markdown('<p class="section-header">📁 Excel Processor</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file, header=0)
                # Check if first row is numeric
                test_row = df.iloc[0].values
                float(test_row[0])
            except:
                df = pd.read_excel(uploaded_file, header=None)
            
            st.caption(f"Rows: {len(df)}")
            if st.button("Process Excel", use_container_width=True):
                with st.spinner("Processing..."):
                    results = process_excel_file(df, models, param_max, param_min)
                    
                    if len(results) == 0:
                        st.error("No valid data rows found.")
                    else:
                        param_names = ['PGA', 'PGV', 'Ia', 'D5-75', 'D5-95', 'Tm', 'CAV',
                                       'PSa_0.030', 'PSa_0.050', 'PSa_0.075', 'PSa_0.100',
                                       'PSa_0.150', 'PSa_0.200', 'PSa_0.250', 'PSa_0.300',
                                       'PSa_0.400', 'PSa_0.500', 'PSa_0.750', 'PSa_1.000',
                                       'PSa_1.500', 'PSa_2.000', 'PSa_2.500', 'PSa_3.000',
                                       'PSa_3.500', 'PSa_4.000']
                        input_names = ['Mw', 'VS30', 'RJB', 'FD', 'FM']
                        
                        valid_data = df.iloc[:len(results), :5].values
                        
                        df_result = pd.DataFrame(np.hstack([valid_data, results]),
                                                 columns=input_names + param_names)
                        st.success(f"✅ {len(results)} rows processed!")
                        
                        from io import BytesIO
                        with BytesIO() as output:
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df_result.to_excel(writer, sheet_name='Results', index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=output.getvalue(),
                                file_name="results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
    
    with col2:
        st.markdown('<p class="section-header">📈 PSa vs Period</p>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        fig.patch.set_facecolor('#fafafa')
        fig.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.92)
        
        if st.session_state.get('calculated', False):
            plot_data(ax, st.session_state['output'])
        else:
            periods = np.array([0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                                0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4])
            y_empty = np.ones(len(periods)) * 1e-4
            ax.loglog(periods, y_empty, 'b-', linewidth=0.5, alpha=0.3, label='Ready')
            ax.set_xlabel('T (s)', fontsize=24, fontweight='bold')
            ax.set_ylabel('PSa (cm/s²)', fontsize=24, fontweight='bold')
            ax.set_xticks([0.03, 0.1, 1, 4])
            ax.set_xticklabels(['0.03', '0.1', '1', '4'], fontsize=20)
            ax.set_xlim(0.03, 4)
            ax.set_ylim(1e-4, 10)
            ax.grid(True, which='both', linestyle='--', alpha=0.4, linewidth=0.8)
            ax.tick_params(axis='both', labelsize=20)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.set_facecolor('#f5f5f5')
            ax.legend(loc='best', frameon=False, fontsize=18)
        
        st.pyplot(fig)


if __name__ == "__main__":
    main()
