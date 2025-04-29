import streamlit as st
import pandas as pd
from collections import defaultdict
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
        "Gemini 1.5": genai.GenerativeModel('gemini-1.5-flash'),
        "Gemini 2.0" Flash: genai.GenerativeModel('gemini-2.0-flash')
    }

def parse_response(response):
    """Extract components from model response"""
    result = {
        'sentiment': 'Unknown',
        'positives': [],
        'improvements': []
    }
    try:
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        # Extract sentiment from first line
        first_line = lines[0].lower()
        if 'positive' in first_line:
            result['sentiment'] = 'Positive'
        elif 'negative' in first_line:
            result['sentiment'] = 'Negative'
        elif 'neutral' in first_line:
            result['sentiment'] = 'Neutral'
        # Extract positives and improvements
        section = None
        for line in lines[1:]:
            line_lower = line.lower()
            if 'positive' in line_lower or 'good' in line_lower:
                section = 'positives'
            elif 'improvement' in line_lower or 'areas' in line_lower:
                section = 'improvements'
            elif section and line.startswith(('- ', '* ', '• ', '1.', '2.', '3.')):
                clean_line = line.split('. ', 1)[-1].strip()
                if section == 'positives':
                    result['positives'].append(clean_line)
                else:
                    result['improvements'].append(clean_line)
    except Exception as e:
        st.error(f"Parsing error: {str(e)}")
    return result

def analyze_sentiment(text, model, model_type):
    try:
        if model_type.startswith("GPT"):
            model_name = {
                "GPT-4": "gpt-4",
                "GPT-3.5": "gpt-3.5-turbo",
                "GPT-4 Mini": "gpt-4o-mini"
            }[model_type]
            response = model.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "system", 
                    "content": """Analyze sentiment and extract:
                    1. Sentiment classification (Positive/Negative/Neutral)
                    2. Top 5 positive aspects
                    3. Top 5 improvement areas"""
                }, {"role": "user", "content": text}]
            )
            return response.choices[0].message.content

        elif model_type in ["Gemini 1.5", "Gemini 2.0 Flash"]:
            response = model.generate_content(
                f"""Analyze sentiment and extract:
                - Sentiment classification
                - Top 5 positive aspects
                - Top 5 improvement areas from: {text}"""
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
            model_order = ["GPT-4", "GPT-3.5", "GPT-4 Mini", "Gemini 1.5", "Gemini 2.0"]
            analysis_data = {model: {'sentiments': [], 'positives': [], 'improvements': []} 
                           for model in model_order}
            with st.status("Analyzing feedback...", expanded=True) as status:
                for model_name in model_order:
                    st.write(f"**Processing {model_name}**")
                    for text in df[feedback_col]:
                        response = analyze_sentiment(text, models[model_name], model_name)
                        parsed = parse_response(response)
                        analysis_data[model_name]['sentiments'].append(parsed['sentiment'])
                        analysis_data[model_name]['positives'].extend(parsed['positives'])
                        analysis_data[model_name]['improvements'].extend(parsed['improvements'])
                status.update(label="Analysis complete!", state="complete")

            # Calculate metrics and top aspects
            results = []
            for model in model_order:
                total = len(df)
                sentiments = analysis_data[model]['sentiments']
                pos = sum(1 for s in sentiments if s == 'Positive')
                neg = sum(1 for s in sentiments if s == 'Negative')
                neu = sum(1 for s in sentiments if s == 'Neutral')
                pos_counter = defaultdict(int)
                for aspect in analysis_data[model]['positives']:
                    if aspect: pos_counter[aspect] += 1
                imp_counter = defaultdict(int)
                for aspect in analysis_data[model]['improvements']:
                    if aspect: imp_counter[aspect] += 1
                top_pos = sorted(pos_counter.items(), key=lambda x: x[1], reverse=True)[:5]
                top_imp = sorted(imp_counter.items(), key=lambda x: x[1], reverse=True)[:5]
                results.append({
                    "Model": model,
                    "Positive%": f"{(pos/total)*100:.1f}%",
                    "Negative%": f"{(neg/total)*100:.1f}%",
                    "Neutral%": f"{(neu/total)*100:.1f}%",
                    "Top Positives": "\n".join([f"- {k} ({v})" for k,v in top_pos]),
                    "Top Improvements": "\n".join([f"- {k} ({v})" for k,v in top_imp])
                })

            # Display results table
            st.subheader("Model Comparison Results")
            results_df = pd.DataFrame(results).set_index("Model")
            st.dataframe(
                results_df.style.set_properties(**{
                    'white-space': 'pre-wrap',
                    'text-align': 'left'
                }),
                height=500,
                use_container_width=True
            )

            # Visualization
            st.subheader("Sentiment Distribution Comparison")
            fig = px.bar(
                pd.DataFrame({model: analysis_data[model]['sentiments'] for model in model_order})
                .apply(pd.Series.value_counts, normalize=True).T*100,
                labels={'value': 'Percentage', 'variable': 'Sentiment'},
                title="Sentiment Distribution by Model (%)",
                barmode='group'
            )
            st.plotly_chart(fig)

    except pd.errors.ParserError:
        st.error("⚠️ Invalid CSV format. Please upload a properly formatted CSV file.")
    except UnicodeDecodeError:
        st.error("⚠️ File encoding error. Please save the CSV as UTF-8 format.")
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
