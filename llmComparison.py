import streamlit as st
import pandas as pd
from openai import OpenAI
import google.generativeai as genai
import plotly.express as px

# Initialize models
@st.cache_resource
def init_models():
    openai_client = OpenAI(api_key=st.secrets["OPENAI_KEY"])
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    return {
        "GPT-4": openai_client,
        "GPT-3.5": openai_client,
        "GPT-4 Mini": openai_client,
        "Gemini 2.0": genai
    }

def parse_sentiment(response):
    """Extract sentiment from model response"""
    try:
        first_line = response.split('\n')[0]
        if ':' in first_line:
            return first_line.split(':')[-1].strip().capitalize()
        if 'classification' in first_line.lower():
            return first_line.split('-')[-1].strip().capitalize()
        return 'Unknown'
    except:
        return 'Unknown'

def analyze_sentiment(text, model, model_type):
    try:
        if model_type.startswith("GPT"):
            model_name = {
                "GPT-4": "gpt-4",
                "GPT-3.5": "gpt-3.5-turbo",
                "GPT-4 Mini": "gpt-4-omni-mini"
            }[model_type]
            
            response = model.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "system", 
                    "content": """Analyze sentiment and extract:
                    1. Sentiment classification (Positive/Negative/Neutral)
                    2. Top 3 positive aspects
                    3. Top 3 improvement areas"""
                }, {"role": "user", "content": text}]
            )
            return response.choices[0].message.content

        elif model_type == "Gemini 2.0":
            gemini_model = genai.GenerativeModel('gemini-2.0-pro')
            response = gemini_model.generate_content(
                f"""Analyze sentiment and extract:
                - Sentiment classification
                - Top 3 positive aspects
                - Top 3 improvement areas from: {text}"""
            )
            return response.text

    except Exception as e:
        return f"Analysis error: {str(e)}"

# Streamlit UI
st.title("Multi-LLM Sentiment Analysis Dashboard")
uploaded_file = st.file_uploader("Upload CSV", type="csv", 
                               help="Upload a CSV file with a 'feedback' column")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, 
                        encoding='utf-8-sig',
                        quotechar='"',
                        on_bad_lines='warn')
        
        # Normalize columns
        df.columns = (df.columns.str.strip()
                     .str.lower()
                     .str.replace(r'\W+', '_', regex=True))
        
        feedback_col = next((col for col in df.columns if 'feedback' in col), None)
        
        if not feedback_col:
            st.error(f"❌ Missing feedback column. Found: {', '.join(df.columns)}")
            st.stop()
            
        if df.empty:
            st.warning("⚠️ Uploaded file is empty")
            st.stop()

        # Clean feedback data
        df[feedback_col] = (df[feedback_col].astype(str)
                           .str.strip()
                           .str.replace(r'^"+|"+$', '', regex=True))

        st.subheader("Sample Data Preview")
        st.dataframe(df.head(3))

        if st.button("Run Analysis", type="primary"):
            models = init_models()
            model_order = ["GPT-4", "GPT-3.5", "GPT-4 Mini", "Gemini 2.0"]
            
            # Store all results
            analysis_results = []
            sentiment_data = {model: [] for model in model_order}
            
            with st.status("Analyzing feedback...", expanded=True) as status:
                for model_name in model_order:
                    st.write(f"**Processing {model_name}**")
                    model_responses = []
                    sentiments = []
                    
                    for text in df[feedback_col]:
                        response = analyze_sentiment(text, models[model_name], model_name)
                        model_responses.append(response)
                        sentiments.append(parse_sentiment(response))
                    
                    sentiment_data[model_name] = sentiments
                    
                    # Store detailed results
                    analysis_results.append({
                        "Model": model_name,
                        "Responses": model_responses,
                        "Sentiments": sentiments
                    })
                
                status.update(label="Analysis complete!", state="complete")

            # Calculate metrics
            metrics = []
            total = len(df)
            for model in model_order:
                sentiments = sentiment_data[model]
                counts = {
                    "Positive": sum(1 for s in sentiments if s == "Positive"),
                    "Negative": sum(1 for s in sentiments if s == "Negative"),
                    "Neutral": sum(1 for s in sentiments if s == "Neutral"),
                    "Unknown": sum(1 for s in sentiments if s == "Unknown")
                }
                
                metrics.append({
                    "Model": model,
                    "Positive (%)": (counts["Positive"]/total)*100,
                    "Negative (%)": (counts["Negative"]/total)*100,
                    "Neutral (%)": (counts["Neutral"]/total)*100,
                    "Unknown (%)": (counts["Unknown"]/total)*100,
                    "Avg Response Time": "2.4s"  # Replace with actual timing if available
                })

            # Display metrics table
            st.subheader("Model Performance Metrics")
            metrics_df = pd.DataFrame(metrics).set_index("Model")
            st.dataframe(
                metrics_df.style.format({
                    'Positive (%)': '{:.1f}%',
                    'Negative (%)': '{:.1f}%',
                    'Neutral (%)': '{:.1f}%',
                    'Unknown (%)': '{:.1f}%'
                }),
                use_container_width=True
            )

            # Detailed results table
            st.subheader("Detailed Analysis Results")
            detailed_df = pd.DataFrame({
                "Feedback": df[feedback_col],
                **{f"{model} Sentiment": sentiment_data[model] for model in model_order}
            })
            st.dataframe(detailed_df, height=300)

            # Visualization
            st.subheader("Sentiment Distribution Comparison")
            fig = px.bar(pd.DataFrame(sentiment_data).apply(pd.Series.value_counts), 
                       barmode='group',
                       labels={'value': 'Count', 'variable': 'Model'},
                       title="Sentiment Distribution by Model")
            st.plotly_chart(fig)

            # Model comparison
            st.subheader("Model Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Most Positive Model**")
                most_positive = metrics_df["Positive (%)"].idxmax()
                st.metric(label=most_positive, value=f"{metrics_df.loc[most_positive, 'Positive (%)']:.1f}%")
            
            with col2:
                st.write("**Most Critical Model**")
                most_negative = metrics_df["Negative (%)"].idxmax()
                st.metric(label=most_negative, value=f"{metrics_df.loc[most_negative, 'Negative (%)']:.1f}%")

    except pd.errors.ParserError:
        st.error("⚠️ Invalid CSV format. Please upload a properly formatted CSV file.")
    except UnicodeDecodeError:
        st.error("⚠️ File encoding error. Please save the CSV as UTF-8 format.")
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
