import streamlit as st
import pandas as pd
from openai import OpenAI
import google.generativeai as genai
from transformers import pipeline
import plotly.express as px

# Initialize models
@st.cache_resource
def init_models():
    models = {
        "GPT-4": OpenAI(api_key=st.secrets["OPENAI_KEY"]),
        "Gemini": genai.configure(api_key=st.secrets["GOOGLE_API_KEY"]),
        "Llama": pipeline("text-classification", model="meta-llama/Meta-Llama-3-70B")
    }
    return models

def analyze_sentiment(text, model, model_type):
    try:
        if model_type == "GPT-4":
            response = model.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": """Analyze sentiment and extract key points:
                    1. Classify as Positive/Negative/Neutral
                    2. List top positive aspects
                    3. List top improvement areas"""
                }, {
                    "role": "user",
                    "content": text
                }]
            )
            return response.choices[0].message.content
            
        elif model_type == "Gemini":
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(f"""Analyze sentiment and extract:
            - Sentiment classification
            - Top 5 positive aspects
            - Top 5 improvement areas from: {text}""")
            return response.text
            
        elif model_type == "Llama":
            result = model(text)
            sentiment = result[0]['label']
            aspects = model(text, return_all_scores=True)
            return f"Sentiment: {sentiment}\nAspects: {aspects}"
            
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
            gpt_results = [analyze_sentiment(text, models["GPT-4"], "GPT-4") 
                          for text in df['Feedback']]
            results['GPT-4'] = gpt_results
            
        with st.spinner("Analyzing with Gemini..."):
            gemini_results = [analyze_sentiment(text, models["Gemini"], "Gemini") 
                             for text in df['Feedback']]
            results['Gemini'] = gemini_results
            
        with st.spinner("Analyzing with Llama..."):
            llama_results = [analyze_sentiment(text, models["Llama"], "Llama") 
                            for text in df['Feedback']]
            results['Llama'] = llama_results

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
            'Positive%': [45, 42, 38],  # Replace with actual calculations
            'Negative%': [25, 28, 30],
            'Neutral%': [30, 30, 32]
        }
        st.dataframe(pd.DataFrame(comparison_data))


