import streamlit as st
import pandas as pd
from openai import OpenAI
import google.generativeai as genai
import plotly.express as px

# Initialize models
@st.cache_resource
def init_models():
    # OpenAI client for GPT-4, GPT-3.5, GPT-4 Mini
    openai_client = OpenAI(api_key=st.secrets["OPENAI_KEY"])
    # Gemini API configuration
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    return {
        "GPT-4": openai_client,
        "GPT-3.5": openai_client,
        "GPT-4 Mini": openai_client,
        "Gemini 2.0": genai
    }

def analyze_sentiment(text, model, model_type):
    try:
        if model_type == "GPT-4":
            response = model.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Analyze sentiment and extract key points:\n1. Classify as Positive/Negative/Neutral\n2. List top positive aspects\n3. List top improvement areas"},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content

        elif model_type == "GPT-3.5":
            response = model.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Analyze sentiment and extract key points:\n1. Classify as Positive/Negative/Neutral\n2. List top positive aspects\n3. List top improvement areas"},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content

        elif model_type == "GPT-4 Mini":
            response = model.chat.completions.create(
                model="gpt-4-omni-mini",  # Use correct model name per OpenAI docs[11]
                messages=[
                    {"role": "system", "content": "Analyze sentiment and extract key points:\n1. Classify as Positive/Negative/Neutral\n2. List top positive aspects\n3. List top improvement areas"},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content

        elif model_type == "Gemini 2.0":
            gemini_model = genai.GenerativeModel('gemini-2.0-pro')  # Or 'gemini-2.0-flash' as available[3][7][8]
            response = gemini_model.generate_content(
                f"""Analyze sentiment and extract:
                - Sentiment classification
                - Top 5 positive aspects
                - Top 5 improvement areas from: {text}"""
            )
            return response.text

    except Exception as e:
        return str(e)

# Streamlit UI
st.title("Multi-LLM Sentiment Analysis Dashboard")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Sample Data")
    st.dataframe(df.head())

    if st.button("Run Analysis"):
        models = init_models()
        results = {}

        with st.spinner("Analyzing with GPT-4..."):
            gpt4_results = [analyze_sentiment(text, models["GPT-4"], "GPT-4")
                            for text in df['Feedback']]
            results['GPT-4'] = gpt4_results

        with st.spinner("Analyzing with GPT-3.5..."):
            gpt35_results = [analyze_sentiment(text, models["GPT-3.5"], "GPT-3.5")
                             for text in df['Feedback']]
            results['GPT-3.5'] = gpt35_results

        with st.spinner("Analyzing with GPT-4 Mini..."):
            gpt4mini_results = [analyze_sentiment(text, models["GPT-4 Mini"], "GPT-4 Mini")
                                for text in df['Feedback']]
            results['GPT-4 Mini'] = gpt4mini_results

        with st.spinner("Analyzing with Gemini 2.0..."):
            gemini_results = [analyze_sentiment(text, models["Gemini 2.0"], "Gemini 2.0")
                              for text in df['Feedback']]
            results['Gemini 2.0'] = gemini_results

        # Visualization
        st.subheader("Sentiment Distribution Comparison")
        fig = px.bar(pd.DataFrame(results).applymap(
            lambda x: x.split('\n')[0].split(': ')[1]),
            barmode='group')
        st.plotly_chart(fig)

        # Detailed Results
        st.subheader("Detailed Analysis")
        model_choice = st.selectbox("Select Model", list(results.keys()))

        col1, col2 = st.columns(2)
        with col1:
            st.write("Top Positive Aspects")
            st.write(results[model_choice][0].split('\n')[1])

        with col2:
            st.write("Top Improvement Areas")
            st.write(results[model_choice][0].split('\n')[2])

        # Comparison Table
        st.subheader("Model Comparison Metrics")
        comparison_data = {
            'Model': list(results.keys()),
            'Positive%': [45, 42, 38, 41],  # Replace with actual calculations
            'Negative%': [25, 28, 30, 27],
            'Neutral%': [30, 30, 32, 32]
        }
        st.dataframe(pd.DataFrame(comparison_data))
