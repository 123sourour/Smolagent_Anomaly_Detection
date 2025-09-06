import streamlit as st
import pandas as pd
import json
import plotly.express as px
import tempfile
import os

from smolagents import CodeAgent, InferenceClientModel
from anomaly_detection_tools import (
    LoadDatasetTool,
    PreprocessDataTool,
    IsolationForestTool,
    LocalOutlierFactorTool,
    OneClassSVMTool,
    DataOverviewTool,
    ReportGeneratorTool
)

@st.cache_resource
def initialize_agent():
    try:
        model = InferenceClientModel(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            token="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"
            )
        tools = [
            LoadDatasetTool(),
            PreprocessDataTool(),
            IsolationForestTool(),
            LocalOutlierFactorTool(),
            OneClassSVMTool(),
            DataOverviewTool(),
            ReportGeneratorTool()
        ]
        agent = CodeAgent(tools=tools, model=model)
        return agent

    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return None


st.set_page_config(
    page_title=" SmolaAgents Anomaly Detection Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.5rem; color: #ff7f0e; border-bottom: 2px solid #ff7f0e; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #ff7f0e; color: white; }
    .agent-response { background-color: #f0f8ff; padding: 1rem; border-left: 4px solid #1f77b4; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    try:
        st.image("versifai_logo.png", width=120)
    except:
        st.write("VersifAI")
with col3:
    try:
        st.image("ULFG1.png", width=100)
    except:
        st.write("ULFG")

st.markdown('<h1 class="main-header">🤖  SmolaAgents Anomaly Detection Platform</h1>', unsafe_allow_html=True)

agent = initialize_agent()
if agent is None:
    st.error("❌ **Fatal Error:** Failed to initialize the AI agent framework.")
    st.stop()

if 'app_data' not in st.session_state:
    st.session_state.app_data = {
        "loaded_data": None,
        "data_overview": None,
        "preprocessed_data": None,
        "detection_results": {},
        "agent_responses": []
    }

def ask_agent(instruction, context=None):
    try:
        with st.spinner("🤖 AI Agent is working..."):
            if context:
                full_instruction = f"{instruction}\n\nContext: {context}"
            else:
                full_instruction = instruction

            st.session_state.app_data['agent_responses'].append(f"**User Instruction:** {instruction}")
            response = agent.run(full_instruction)

            return response
    except Exception as e:
        error_msg = f"Agent error: {str(e)}"
        st.error(f"❌ {error_msg}")
        st.session_state.app_data['agent_responses'].append(f"**Error:** {error_msg}")
        return None


def clear_all_data():
    st.session_state.app_data = {
        "loaded_data": None,
        "data_overview": None,
        "preprocessed_data": None,
        "detection_results": {},
        "agent_responses": []
    }
    st.success("🗑️ All data has been cleared.")


def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def extract_json_from_response(response):
    try:
        if isinstance(response, dict):
            return response
        elif isinstance(response, str):
            import re
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            if matches:
                return json.loads(matches[-1])
            else:
                return json.loads(response)
        else:
            return response
    except:
        return response



with st.sidebar:
    st.markdown("## 🤖 AI Agent Control Panel")
    if st.button("🗑️ Clear All Data & Reset", type="primary"):
        clear_all_data()
        st.rerun()

    st.markdown("---")
    st.markdown("### ✅ Status")
    status_map = st.session_state.app_data
    st.markdown(f"**Data Loaded:** {'✅' if status_map['loaded_data'] else '❌'}")
    st.markdown(f"**Data Overview:** {'✅' if status_map['data_overview'] else '❌'}")
    st.markdown(f"**Data Preprocessed:** {'✅' if status_map['preprocessed_data'] else '❌'}")
    st.markdown(f"**Algorithms Run:** {len(status_map['detection_results'])}")

    st.markdown("---")
    st.markdown("### 🔍 Agent Activity")
    if st.button("View Agent Logs"):
        with st.expander("Recent Agent Interactions", expanded=True):
            for response in st.session_state.app_data['agent_responses'][-5:]:  # Show last 5
                st.markdown(f"<div class='agent-response'>{response}</div>", unsafe_allow_html=True)



st.markdown('<div class="section-header">📁 Step 1: AI-Powered Data Loading</div>', unsafe_allow_html=True)
with st.container(border=True):
    dataset_option = st.radio("Choose data source:", ["Sample Dataset", "Upload Custom File"], horizontal=True,
                              key="data_source")

    if dataset_option == "Sample Dataset":
        sample_dataset = st.selectbox("Select sample dataset:",
                                      ["Credit Card Transactions", "IoT Sensor Data", "Network Logs"])
        if st.button("🤖 Ask AI to Load Sample Dataset", type="primary"):
            instruction = f"Load the sample dataset '{sample_dataset}' using the load_dataset tool. Return the loaded data in JSON format."
            response = ask_agent(instruction)

            if response:
                try:
                    result_data = extract_json_from_response(response)
                    st.session_state.app_data['loaded_data'] = json.dumps(result_data) if isinstance(result_data, (
                    dict, list)) else str(response)
                    st.success(f"✅ AI successfully loaded **{sample_dataset}** dataset!")
                    st.markdown(f"<div class='agent-response'>🤖 **AI Response:** {response}</div>",
                                unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error processing AI response: {e}")

    else:
        uploaded_file = st.file_uploader("Upload your dataset (CSV, XLSX):", type=['csv', 'xlsx', 'xls'])
        if uploaded_file and st.button("🤖 Ask AI to Load Uploaded File", type="primary"):
            temp_path = save_uploaded_file(uploaded_file)
            if temp_path:
                instruction = f"Load the dataset from the file path '{temp_path}' using the load_dataset tool. The file is a {uploaded_file.name.split('.')[-1].upper()} file. Return the loaded data in JSON format."
                response = ask_agent(instruction)
                os.unlink(temp_path)
                if response:
                    try:
                        result_data = extract_json_from_response(response)
                        st.session_state.app_data['loaded_data'] = json.dumps(result_data) if isinstance(result_data, (
                        dict, list)) else str(response)
                        st.success(f"✅ AI successfully loaded **{uploaded_file.name}**!")
                        st.markdown(f"<div class='agent-response'>🤖 **AI Response:** {response}</div>",
                                    unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error processing AI response: {e}")

if st.session_state.app_data['loaded_data']:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📊 Step 2: AI Data Analysis</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if st.button("🤖 Ask AI to Analyze Data"):
                instruction = "Analyze the loaded dataset using the data_overview tool. Provide comprehensive statistics including total records, features, missing values, data types, and any patterns you observe."
                context = f"Dataset: {st.session_state.app_data['loaded_data'][:500]}..."  # First 500 chars as context
                response = ask_agent(instruction, context)

                if response:
                    try:
                        result_data = extract_json_from_response(response)
                        if isinstance(result_data, dict) and "error" not in result_data:
                            st.session_state.app_data['data_overview'] = result_data
                            st.success("✅ AI completed data analysis!")
                        else:
                            st.session_state.app_data['data_overview'] = {"ai_response": str(response)}
                            st.success("✅ AI analyzed the data!")
                        st.markdown(f"<div class='agent-response'>🤖 **AI Analysis:** {response}</div>",
                                    unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error processing AI analysis: {e}")

            if st.session_state.app_data['data_overview']:
                overview = st.session_state.app_data['data_overview']
                if isinstance(overview, dict) and 'total_records' in overview:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Records", overview.get('total_records', 'N/A'))
                    c2.metric("Features", overview.get('total_features', 'N/A'))
                    c3.metric("Missing Values", overview.get('missing_values', 'N/A'))

                    if 'anomaly_rate' in overview:
                        st.metric("Anomaly Rate", f"{overview['anomaly_rate']:.2f}%")

                with st.expander("📋 View AI Analysis Details"):
                    st.json(overview)

    with col2:
        st.markdown('<div class="section-header">🔧 Step 3: AI Data Preprocessing</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("AI will handle missing values, encode variables, and scale features.")
            if st.button("🤖 Ask AI to Preprocess Data", type="primary"):
                instruction = "Preprocess the loaded dataset using the preprocess_data tool. Handle missing values, encode categorical variables, separate features from labels, and scale numerical features. Return the preprocessed data with both original and scaled features."
                context = f"Dataset: {st.session_state.app_data['loaded_data'][:500]}..."
                response = ask_agent(instruction, context)

                if response:
                    try:
                        result_data = extract_json_from_response(response)
                        st.session_state.app_data['preprocessed_data'] = json.dumps(result_data) if isinstance(
                            result_data, dict) else str(response)
                        st.success("✅ AI completed data preprocessing!")
                        st.markdown(f"<div class='agent-response'>🤖 **AI Preprocessing:** {response}</div>",
                                    unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error processing AI preprocessing: {e}")

            if st.session_state.app_data['preprocessed_data']:
                st.info("🤖 Data is preprocessed by AI and ready for anomaly detection.")

if st.session_state.app_data['preprocessed_data']:
    st.markdown('<div class="section-header">🎯 Step 4: AI Anomaly Detection</div>', unsafe_allow_html=True)
    with st.container(border=True):
        algorithm = st.selectbox("Select anomaly detection algorithm:",
                                 ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"])

        params = {}
        if algorithm == "Isolation Forest":
            c1, c2 = st.columns(2)
            params['n_estimators'] = c1.slider("Number of Estimators", 50, 500, 100, 10, key='if_est')
            params['contamination'] = c2.slider("Contamination Rate", 0.01, 0.5, 0.1, 0.01, key='if_cont')
            button_label = "🤖 Ask AI to Run Isolation Forest"
            tool_instruction = f"isolation_forest_detector with n_estimators={params['n_estimators']} and contamination={params['contamination']}"

        elif algorithm == "Local Outlier Factor":
            c1, c2 = st.columns(2)
            params['n_neighbors'] = c1.slider("Number of Neighbors", 5, 50, 20, 1, key='lof_n')
            params['contamination'] = c2.slider("Contamination Rate", 0.01, 0.5, 0.1, 0.01, key='lof_cont')
            button_label = "🤖 Ask AI to Run Local Outlier Factor"
            tool_instruction = f"local_outlier_factor_detector with n_neighbors={params['n_neighbors']} and contamination={params['contamination']}"

        elif algorithm == "One-Class SVM":
            c1, c2, c3 = st.columns(3)
            params['nu'] = c1.slider("Nu Parameter", 0.01, 0.5, 0.1, 0.01, key='svm_nu')
            params['kernel'] = c2.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key='svm_kernel')
            params['gamma'] = c3.selectbox("Gamma", ["scale", "auto"], key='svm_gamma')
            button_label = "🤖 Ask AI to Run One-Class SVM"
            tool_instruction = f"one_class_svm_detector with nu={params['nu']}, kernel={params['kernel']}, and gamma={params['gamma']}"

        if st.button(button_label, type="primary"):
            instruction = f"Apply {algorithm} anomaly detection on the preprocessed data using the {tool_instruction}. Use the scaled features from the preprocessed data. Analyze the results and provide insights about the detected anomalies, including performance metrics if ground truth labels are available."
            context = f"Preprocessed data: {st.session_state.app_data['preprocessed_data'][:1000]}..."
            response = ask_agent(instruction, context)

            if response:
                try:
                    result_data = extract_json_from_response(response)
                    st.session_state.app_data['detection_results'][algorithm] = result_data if isinstance(result_data,
                                                                                                          dict) else {
                        "ai_response": str(response)}
                    st.success(f"✅ AI successfully completed **{algorithm}** analysis!")
                    st.markdown(f"<div class='agent-response'>🤖 **AI Detection Analysis:** {response}</div>",
                                unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error processing AI detection results: {e}")

if st.session_state.app_data['detection_results']:
    st.markdown('<div class="section-header">📈 Step 5: AI Results Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "⚖️ AI Comparison", "📋 AI Report"])

    with tab1:
        selected_algo = st.selectbox("Select algorithm to visualize:",
                                     list(st.session_state.app_data['detection_results'].keys()))
        if selected_algo:
            result = st.session_state.app_data['detection_results'][selected_algo]

            if isinstance(result, dict) and 'total_samples' in result:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Samples", result.get('total_samples', 'N/A'))
                c2.metric("Anomalies Found", result.get('anomaly_count', 'N/A'))
                rate = (result.get('anomaly_count', 0) / result.get('total_samples', 1)) * 100 if result.get(
                    'total_samples', 1) > 0 else 0
                c3.metric("Detection Rate", f"{rate:.2f}%")

                if 'metrics' in result and result['metrics']:
                    c4.metric("F1-Score", f"{result.get('metrics', {}).get('f1_score', 0):.3f}")
                else:
                    c4.metric("F1-Score", "N/A")

                if result.get('anomaly_count') and result.get('total_samples'):
                    c1, c2 = st.columns(2)
                    with c1:
                        normal_count = result['total_samples'] - result['anomaly_count']
                        fig_pie = px.pie(
                            values=[normal_count, result['anomaly_count']],
                            names=['Normal', 'Anomaly'],
                            title="AI Detected Anomaly Distribution",
                            color_discrete_map={'Normal': '#2E8B57', 'Anomaly': '#DC143C'}
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with c2:
                        # Performance metrics bar chart (if available)
                        if 'metrics' in result and result['metrics']:
                            metrics_data = result['metrics']
                            fig_bar = px.bar(
                                x=list(metrics_data.keys()),
                                y=list(metrics_data.values()),
                                title="AI Model Performance Metrics",
                                labels={'x': 'Metric', 'y': 'Score'}
                            )
                            fig_bar.update_layout(yaxis_range=[0, 1])
                            st.plotly_chart(fig_bar, use_container_width=True)

                    if 'scores' in result and result['scores']:
                        fig_hist = px.histogram(
                            x=result['scores'],
                            nbins=50,
                            title="AI Anomaly Score Distribution",
                            labels={'x': 'Anomaly Score', 'y': 'Frequency'}
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.markdown("### AI Analysis Result:")
                st.markdown(f"<div class='agent-response'>{result}</div>", unsafe_allow_html=True)

    with tab2:
        if len(st.session_state.app_data['detection_results']) > 1:
            if st.button("🤖 Ask AI to Compare All Results", type="primary"):
                instruction = "Compare all the anomaly detection results from different algorithms. Analyze their performance, detection rates, and provide recommendations on which algorithm performs best for this dataset."
                context = f"Results: {json.dumps(st.session_state.app_data['detection_results'])}"
                response = ask_agent(instruction, context)

                if response:
                    st.markdown("### 🤖 AI Comparison Analysis:")
                    st.markdown(f"<div class='agent-response'>{response}</div>", unsafe_allow_html=True)


            comp_data = []
            for algo, res in st.session_state.app_data['detection_results'].items():
                if isinstance(res, dict) and 'anomaly_count' in res:
                    row = {
                        'Algorithm': algo,
                        'Anomalies Detected': res.get('anomaly_count', 0),
                        'Detection Rate (%)': (res.get('anomaly_count', 0) / res.get('total_samples',
                                                                                     1)) * 100 if res.get(
                            'total_samples') else 0
                    }
                    if 'metrics' in res and res['metrics']:
                        row.update(res['metrics'])
                    comp_data.append(row)

            if comp_data:
                comp_df = pd.DataFrame(comp_data).set_index('Algorithm')
                st.dataframe(
                    comp_df.style.format("{:.3f}").background_gradient(cmap='viridis'),
                    use_container_width=True
                )
        else:
            st.info("Run at least two algorithms to see AI comparison.")

    with tab3:
        if st.session_state.app_data['data_overview']:
            report_algo = st.selectbox(
                "Select algorithm for AI report:",
                list(st.session_state.app_data['detection_results'].keys()),
                key="report_algo_select"
            )
            if st.button("🤖 Ask AI to Generate Comprehensive Report", type="primary"):
                instruction = f"Generate a comprehensive anomaly detection report for the {report_algo} algorithm. Include dataset overview, detection results, performance analysis, insights, and actionable recommendations. Use the generate_anomaly_report tool and enhance it with your analysis."
                context = f"Algorithm: {report_algo}, Results: {json.dumps(st.session_state.app_data['detection_results'][report_algo])}, Overview: {json.dumps(st.session_state.app_data['data_overview'])}"
                response = ask_agent(instruction, context)

                if response:
                    st.markdown("### 📄 AI-Generated Comprehensive Report:")
                    st.markdown(f"<div class='agent-response'>{response}</div>", unsafe_allow_html=True)

                    st.download_button(
                        "💾 Download AI Report",
                        str(response),
                        f"ai_anomaly_report_{report_algo.replace(' ', '_').lower()}.md",
                        "text/markdown"
                    )
        else:
            st.warning("Please complete data analysis in Step 2 before generating an AI report.")

st.markdown('<div class="section-header">🧠 Advanced AI Analysis</div>', unsafe_allow_html=True)
with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🤖 Ask AI for Custom Analysis", type="primary"):
            custom_instruction = st.text_area(
                "Enter your custom instruction for the AI:",
                placeholder="e.g., 'Analyze the correlation between features and anomalies' or 'Suggest optimal parameters for better detection'",
                key="custom_ai_instruction"
            )
            if custom_instruction:
                full_context = {
                    "loaded_data": st.session_state.app_data.get('loaded_data', 'None'),
                    "data_overview": st.session_state.app_data.get('data_overview', 'None'),
                    "preprocessed_data": st.session_state.app_data.get('preprocessed_data', 'None'),
                    "detection_results": st.session_state.app_data.get('detection_results', {})
                }
                response = ask_agent(custom_instruction, str(full_context)[:2000])

                if response:
                    st.markdown("### 🤖 AI Custom Analysis:")
                    st.markdown(f"<div class='agent-response'>{response}</div>", unsafe_allow_html=True)

    with col2:
        if st.button("🤖 Ask AI for Recommendations", type="primary"):
            instruction = "Based on all the data analysis and anomaly detection results, provide actionable recommendations for improving anomaly detection performance, data quality, and next steps for investigation."
            full_context = {
                "data_overview": st.session_state.app_data.get('data_overview', 'None'),
                "detection_results": st.session_state.app_data.get('detection_results', {})
            }
            response = ask_agent(instruction, str(full_context)[:2000])

            if response:
                st.markdown("### 🤖 AI Recommendations:")
                st.markdown(f"<div class='agent-response'>{response}</div>", unsafe_allow_html=True)

