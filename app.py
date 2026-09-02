import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import os
import urllib.request
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
    .stSidebar .stNumberInput label {
        font-size: 1.0rem !important;
        font-weight: 500 !important;
    }
    .stButton button {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
    }
    .output-label {
        color: #000000 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    .output-value {
        color: #1565C0 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .output-unit {
        color: #666 !important;
        font-size: 0.8rem !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)


def build_model_from_weights(layer_sizes, weights_list, biases_list):
    """Build a PyTorch model from MATLAB weights and biases"""
    
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn.Tanh())
    
    model = nn.Sequential(*layers)
    
    # Set weights and biases
    with torch.no_grad():
        for i in range(len(weights_list)):
            # Get weights and biases from MATLAB
            W = np.array(weights_list[i], dtype=np.float32)
            b = np.array(biases_list[i], dtype=np.float32).flatten()
            
            # Calculate layer index in PyTorch (each Linear layer is at even index)
            layer_idx = i * 2
            if layer_idx < len(model):
                model[layer_idx].weight.data = torch.tensor(W.T, dtype=torch.float32)
                model[layer_idx].bias.data = torch.tensor(b, dtype=torch.float32)
    
    return model


def load_models_from_mat(weights_file):
    try:
        data = scipy.io.loadmat(weights_file)
        
        param_max = data['Param_Max'].flatten()
        param_min = data['Param_Min'].flatten()
        
        # Get layer sizes
        if 'layer_sizes' in data:
            layer_sizes_list = data['layer_sizes'][0]
        else:
            layer_sizes_list = [[5, 20, 25] for _ in range(10)]
        
        # Get weights and biases
        weights_list = data['weights_list'][0]
        biases_list = data['biases_list'][0]
        
        models = []
        
        for i in range(len(weights_list)):
            try:
                # Get layer sizes for this model
                if isinstance(layer_sizes_list[i], np.ndarray):
                    sizes = layer_sizes_list[i].flatten().tolist()
                else:
                    sizes = layer_sizes_list[i]
                
                # Get weights and biases for each layer
                if isinstance(weights_list[i], np.ndarray) and weights_list[i].size == 1:
                    layer_weights = [weights_list[i].item()]
                else:
                    layer_weights = weights_list[i]
                
                if isinstance(biases_list[i], np.ndarray) and biases_list[i].size == 1:
                    layer_biases = [biases_list[i].item()]
                else:
                    layer_biases = biases_list[i]
                
                # Ensure they are lists
                if not isinstance(layer_weights, list):
                    layer_weights = [layer_weights]
                if not isinstance(layer_biases, list):
                    layer_biases = [layer_biases]
                
                # Ensure all items are numpy arrays
                layer_weights = [np.array(w, dtype=np.float32) for w in layer_weights]
                layer_biases = [np.array(b, dtype=np.float32) for b in layer_biases]
                
                model = build_model_from_weights(sizes, layer_weights, layer_biases)
                model.eval()
                models.append(model)
                
            except Exception as e:
                st.warning(f"⚠️ Model {i+1} loading error: {e}")
                continue
        
        return models, param_max, param_min
    
    except Exception as e:
        st.error(f"Error loading .mat file: {e}")
        return None, None, None


@st.cache_resource
def load_gmm_models():
    if not os.path.exists("GMM_Turkie_2025_weights.mat"):
        try:
            url = "https://raw.githubusercontent.com/banimahd/GMM_Turkiye_2026/main/GMM_Turkie_2025_weights.mat"
            urllib.request.urlretrieve(url, "GMM_Turkie_2025_weights.mat")
            st.info("✅ Model weights downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Could not download model file: {e}")
            return None, None, None
    
    models, param_max, param_min = load_models_from_mat("GMM_Turkie_2025_weights.mat")
    return models, param_max, param_min


def normalize_inputs(inputs, param_min, param_max):
    return 0.60 * ((inputs - param_min) / (param_max - param_min)) + 0.20


def call_back_excel(mw, vs30, rjb, fd, fm, models, param_max, param_min):
    inputs = np.array([fd, fm, mw, rjb, vs30], dtype=np.float32).reshape(1, -1)
    
    inputs_norm = normalize_inputs(inputs, param_min.reshape(1, -1), param_max.reshape(1, -1))
    inputs_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
    
    outputs_mean = np.zeros(25)
    for model in models:
        with torch.no_grad():
            output = model(inputs_tensor).numpy().flatten()
            outputs_mean += output / len(models)
    
    pga = round(np.exp(outputs_mean[0]) * 986, 2)
    pgv = round(np.exp(outputs_mean[1]), 2)
    ia = round(np.exp(outputs_mean[2]), 2)
    d5_75 = round(np.exp(outputs_mean[3]), 2)
    d5_95 = round(np.exp(outputs_mean[4]), 2)
    tm = round(np.exp(outputs_mean[5]), 2)
    cav = round(np.exp(outputs_mean[6]), 2)
    
    psas = [round(np.exp(outputs_mean[i]) * 986, 2) for i in range(7, 25)]
    
    output = [pga, pgv, ia, d5_75, d5_95, tm, cav] + psas
    return np.array(output)


def plot_data(ax, outputs):
    periods = np.array([0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                        0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    
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
        
        if models is None or len(models) == 0:
            st.error("Failed to load models. Please check the file path.")
            st.info("Make sure GMM_Turkie_2025_weights.mat is in the same folder.")
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
                
                names = ['PGA', 'PGV', 'Ia', 'D5-75', 'D5-95', 'Tm', 'CAV']
                for i, name in enumerate(names):
                    st.session_state[f'output_{name}'] = output[i]
        
        st.markdown("---")
        st.markdown('<p style="font-size:0.8rem; color:#888; text-align:center; margin-top:1rem;">Made by Amir Banimahd</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.4, 0.6])
    
    with col1:
        st.markdown('<p class="section-header">📤 Outputs</p>', unsafe_allow_html=True)
        
        row1 = st.columns(3)
        outputs_row1 = [
            ('PGA', 'cm/s²', 'output_PGA'),
            ('PGV', 'cm/s', 'output_PGV'),
            ('Ia', 'cm/s', 'output_Ia')
        ]
        for col, (name, unit, key) in zip(row1, outputs_row1):
            val = st.session_state.get(key, 0)
            with col:
                st.markdown(f'<p class="output-label">{name}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="output-value">{val:.2f}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="output-unit">{unit}</p>', unsafe_allow_html=True)
        
        row2 = st.columns(3)
        outputs_row2 = [
            ('D5-75', 's', 'output_D5-75'),
            ('D5-95', 's', 'output_D5-95'),
            ('Tm', 's', 'output_Tm')
        ]
        for col, (name, unit, key) in zip(row2, outputs_row2):
            val = st.session_state.get(key, 0)
            with col:
                st.markdown(f'<p class="output-label">{name}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="output-value">{val:.2f}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="output-unit">{unit}</p>', unsafe_allow_html=True)
        
        row3 = st.columns(1)
        val = st.session_state.get('output_CAV', 0)
        with row3[0]:
            st.markdown(f'<p class="output-label">CAV</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="output-value">{val:.2f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="output-unit">cm/s</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="section-header">📁 Excel Processor</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file, header=0)
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
