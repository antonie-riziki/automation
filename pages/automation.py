import streamlit as st
import seaborn as sb 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import io
import africastalking
import google.generativeai as genai

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.lib.utils import ImageReader
from PIL import Image


from sklearn.impute import SimpleImputer

sys.path.insert(1, './pages')
print(sys.path.insert(1, '../pages/'))

from ct_model import the_explainer

from dotenv import load_dotenv

load_dotenv()

st.image('https://www.greeningafrika.com/wp-content/uploads/2024/09/Groundbreaking-Technology-Uses-Small-Mammal-Footprints-to-Monitor-Climate-Change.webp', use_container_width=True)

st.divider()

# Header and Subheader
st.title("Automated Data Analyzer")
st.subheader("Smart Data Cleaning and Analysis")
st.write("This tool automates the process of analyzing and cleaning CSV data by detecting errors, handling missing values, and generating an analytical report.")

# Store file paths
uploaded_files = {}



def load_data(file):
    try:
        df = pd.read_csv(file, encoding='latin-1')
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def clean_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)  # Remove duplicates
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill missing numeric values
    df.fillna("Unknown", inplace=True)  # Fill missing categorical values
    return df

def generate_report(df):
    # report = {
    #     "Total Rows": len(df),
    #     "Total Columns": len(df.columns),
    #     "Missing Values": df.isnull().sum().sum(),
    #     "Duplicate Rows": df.duplicated().sum(),
    #     "Column Data Types": df.dtypes.to_dict(),
    #     "Basic Statistics": df.describe().to_dict()
    # }
    report = df.describe()
 
    return report


def get_df_info(df):
     buffer = io.StringIO ()
     df.info (buf=buffer)
     lines = buffer.getvalue ().split ('\n')
     # lines to print directly
     lines_to_print = [0, 1, 2, -2, -3]
     for i in lines_to_print:
         st.write (lines [i])
     # lines to arrange in a df
     list_of_list = []
     for x in lines [0:-3]:
         list = x.split ()
         list_of_list.append (list)
     info_df = pd.DataFrame (list_of_list)
     # info_df.drop (columns=['index', 'null'], axis=1, inplace=True)
     st.dataframe(info_df)


simple_impute = SimpleImputer()

def get_categorical_series(df):
    categories = []
    simple_impute = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for i in df.select_dtypes(include=['object']):
        categories.append(i)
    df[categories] = simple_impute.fit_transform(df[categories])
    return df.head()


def get_quantitative_series(df):
    numericals = []
    simple_impute = SimpleImputer(missing_values=np.nan, strategy='mean')
    for i in df.select_dtypes(include=['float64', 'int64']):
        numericals.append(i)
    df[numericals] = simple_impute.fit_transform(df[numericals])
    return df.head()


def get_heatmap(df):
    cmap_options = ['coolwarm', 'viridis', 'mako', 'cubehelix', 'rocket', 'flare', 'magma', 'Greens', 'Reds_r', 'BuGn_r', 'terrain_r']
    cmap_selection = st.pills('color map', options=cmap_options)

    numerical_categories = []
    for i in df.select_dtypes(include=['float64','int64']):
        numerical_categories.append(i)
        fig, ax = plt.subplots(figsize=(10, 6))
        sb.heatmap(df[numerical_categories].corr(), annot=True, fmt=".2f", cmap=cmap_selection, ax=ax)
        plt.title(f'Pearsons correlation of columns', fontdict={'size':14})
    return st.pyplot(fig)



# Function to generate Gemini insights
def get_gemini_insights(df, selected_columns):
    
    prompt = f"""
    Perform exploratory data analysis on the following dataset columns:
    {selected_columns}

    Provide a concise, precise, and quantitative explanation of trends, distributions, 
    and correlations found in the data. Focus on key statistical insights.
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash", 
        system_instruction = '''
            Role: You are an experienced data analyst with deep expertise in exploratory data analysis (EDA), statistical reasoning, and business intelligence. 
            Your task is to analyze datasets, extract key patterns, and generate concise, quantitative, and insightful summaries for decision-making.

            ### Guidelines for Analysis:
            1️. Be precise and quantitative – Use statistics, trends, and patterns to explain insights (e.g., mean, median, standard deviation, skewness).
            2️. Identify key distributions – Determine if the dataset exhibits normality, skewness, or outliers. Mention the spread, peaks, and anomalies.
            3️. Highlight correlations – Describe potential relationships within the data (e.g., positive correlation between income and age).
            4️. Explain business impact – Where relevant, connect insights to real-world implications (e.g., high attrition rate in employees over 40 suggests retention issues).
            5️. Avoid generic statements – Focus on actionable insights backed by data-driven reasoning.

            ### Handling General Questions:
            - If the user asks a **general question** (e.g., "What is the mean sample size across all studies?"), analyze the dataset holistically.
            - Identify relevant information based on the context of the question and infer the best approach to compute the answer.
            - If multiple interpretations exist, provide the most meaningful one along with reasoning.
            - If additional clarification is needed, ask the user a follow-up question.

            ### Expected Output Format:
            - **Summary:** Provide a concise, data-backed statement (e.g., "The dataset shows a right-skewed distribution with a median value of 52.")
            - **Key Observations:** Highlight significant patterns or anomalies (e.g., "There is an unexpected spike in sales during Q4, suggesting seasonal demand.")
            - **Correlations & Trends:** Identify relationships within the data (e.g., "An increase in customer retention correlates with higher engagement levels (+0.72).")
            - **Actionable Insights:** Provide practical recommendations (e.g., "The high standard deviation in expenditure suggests unstable spending habits among customers.")
            - **General Questions Response:** Answer based on available data (e.g., "The average sample size across all studies in the dataset is 450, with a standard deviation of 85. The distribution suggests that most studies have a sample size between 365 and 535.")

        ''')


    response = model.generate_content(
    prompt,
    generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
    )
)
    return st.write(response.text)

# Function to generate and download PDF report using ReportLab
def generate_pdf(df, selected_columns, insights):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    
    # Ensure insights is not None
    insights = insights if insights else "No insights available."

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(200, 750, "Exploratory Data Analysis Report")
    
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, 720, f"Selected Columns: {', '.join(selected_columns)}")

    pdf.setFont("Helvetica", 11)
    
    # **Format and wrap insights text**
    y_position = 690  # Start drawing from this Y position
    max_width = 500   # Max width for text wrapping
    lines = simpleSplit(insights, pdf._fontname, pdf._fontsize, max_width)
    
    for line in lines:
        pdf.drawString(50, y_position, line)
        y_position -= 20  # Move down for next line
        
        # Add new page if needed
        if y_position < 100:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y_position = 750  # Reset position for new page

    pdf.showPage()  # Move to next page for plots

    # Save plots for selected columns
    for col in selected_columns:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sb.histplot(df[col], kde=True, ax=ax)
            plt.title(f"Distribution of {col}")
            
            img_path = f"{col}.png"
            plt.savefig(img_path, format="png")
            plt.close()

            pdf.drawImage(img_path, 100, 400, width=400, height=300)
            pdf.showPage()

    pdf.save()
    
    buffer.seek(0)
    return buffer





# def group_items(df):

#     grp_cols = []

#     for i in df.select_dtypes(include=['object']):
#         grp_cols.append(i)

#     grpby_columns = st.multiselect('select series to group', grp_cols)


#     if grpby_columns is not None:
#         group_by_cols = df.groupby(grpby_columns)

#         st.write(group_by_cols.head())

#     else: 
#         st.markdown('select column(s) to group')

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])



if uploaded_file:
    file_name = uploaded_file.name
    uploaded_files[file_name] = uploaded_file

    df = load_data(uploaded_file)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    if df is not None:
        st.write("### Original Data Preview")
        
        
        st.dataframe(df.head())
        
        cleaned_df = clean_data(df)
        
        col1, col2 = st.columns(2)

        with col1:
            report = generate_report(cleaned_df)
            st.write("### Data Report")
            st.dataframe(report)
            st.json(report)
            st.dataframe(df.isnull().sum())

        with col2:
            st.write(df.shape)
            get_df_info(df)
        
        vis1, vis2 = st.columns(2)

        with vis1:
            get_categorical_series(df)

            st.write('For categorical analysis')

            category_choice = st.selectbox(label="select series", options=[i for i in df.select_dtypes(include="object")])

            
            entry_options = [20, 50, 100, 200]
            st.write(len(df[category_choice].value_counts()))
            category_entries = st.pills('select entries', options=entry_options, key='cat_entries')

            st.bar_chart(df[category_choice].value_counts().head(category_entries))

            plt.title(f"Categorical analysis for the {category_choice}")

        with vis2:
            get_quantitative_series(df)

            ('For Quantitative analysis')

            quantitative_choice = st.selectbox(label="select series", options=[i for i in df.select_dtypes(include=["float64", "int64"])])

            # entry_options = [20, 50, 100, 200]
            st.write(len(df[category_choice].value_counts()))
            quantitative_entries = st.pills('select entries', options=entry_options, key='qty_entry')

            # df = px.data.tips()
            # fig = px.histogram(df[quantitative_choice])
            # fig.show()

            st.bar_chart(df[quantitative_choice].head(quantitative_entries))

        get_heatmap(df)
        
        st.write("### Cleaned Data Preview")
        st.dataframe(cleaned_df.head())
        
        # Option to download cleaned data
        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned Data", csv, "cleaned_data.csv", "text/csv")



# Allow user to select previously uploaded files
# if uploaded_files:
#     selected_file = st.selectbox("Select a previously uploaded file:", list(uploaded_files.keys()))
#     if selected_file:
#         df = load_data(uploaded_files[selected_file])
#         if df is not None:
#             st.write("### Selected File Data Preview")
#             st.dataframe(df.head())
#         else:
#         	pass


    def get_chat_response(prompt):

        # df = load_data(uploaded_file)
        model = genai.GenerativeModel("gemini-1.5-flash",

            system_instruction = f'''

            You are an AI-powered Data Analyst tasked with performing Exploratory Data Analysis (EDA) on the dataset {df}.  

            Your job is to engage in a **conversation** with the user, answering their questions based on the dataset provided.  

            You should be able to:  

            - Explain **trends, anomalies, and distributions** in the data.  
            - Perform **computational analysis** when asked (mean, median, standard deviation, correlations, regressions, etc.).  
            - Generate **and display** visual representations such as histograms, box plots, scatter plots, and correlation matrices **directly in Streamlit** instead of returning Python code.  
            - Answer natural language questions with **accurate, data-driven responses**.  

            **How to Handle Visualization Requests:**  
            - When the user requests a visualization (e.g., "Plot a histogram for column_name"), **generate and display the plot directly in Streamlit** instead of returning Python code.  
            - Use **Matplotlib and Seaborn** for visualizations.  
            - Ensure plots are **clear, well-labeled, and easy to interpret**.  
            - Avoid returning raw Python code unless explicitly requested.  

            **Examples of User Requests:**  
            1. "What is the mean value of column1?"  
            2. "Find the correlation between column_A and column_B."  
            3. "Generate a histogram for column_name." (→ **Show the plot in Streamlit**)  
            4. "Show the top 5 highest values in column_name."  

            If a user asks a question that requires **computation or visualization**, perform the necessary analysis and return a well-structured response.  
            **Ensure responses are precise, insightful, and backed by data.**  
            If you need clarification, ask the user before proceeding.  

        ''')


       # Generate AI response
        response = model.generate_content(
            prompt,
            generation_config = genai.GenerationConfig(
            max_output_tokens=1000,
            temperature=0.1, 
          )
        )
        

        response_text = response.text

        return response_text



# Grouping Data and Visualization
if uploaded_file:
    st.write("### Data Grouping and Visualization")
    
    # User selects columns to group by
    selected_columns = st.multiselect("Select columns to group by:", df.columns)
    
    # User selects aggregation method
    aggregation_method = st.selectbox("Select aggregation method:", ["Count", "Min", "Max", "Sum", "Mean", "Median", "Std Dev", "Var", "Mean absolute dev", "Product"])
    
    if selected_columns:
        if aggregation_method == "Count":
            grouped_df = df.groupby(selected_columns).size().reset_index(name='agg_column')
        else:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_column = st.selectbox("Select numerical column for aggregation:", numerical_columns)
            
            if aggregation_method == "Sum":
                grouped_df = df.groupby(selected_columns)[agg_column].sum().reset_index()
            elif aggregation_method == "Min":
                grouped_df = df.groupby(selected_columns)[agg_column].min().reset_index()
            elif aggregation_method == "Max":
                grouped_df = df.groupby(selected_columns)[agg_column].max().reset_index()
            elif aggregation_method == "Mean":
                grouped_df = df.groupby(selected_columns)[agg_column].mean().reset_index()
            elif aggregation_method == "Median":
                grouped_df = df.groupby(selected_columns)[agg_column].median().reset_index()
            elif aggregation_method == "Std Dev":
                # params = st.select
                grouped_df = df.groupby(selected_columns)[agg_column].std().reset_index()
            elif aggregation_method == "Var":
                grouped_df = df.groupby(selected_columns)[agg_column].var().reset_index()
            elif aggregation_method == "Mean absolute dev":
                grouped_df = df.groupby(selected_columns)[agg_column].median().reset_index()
            elif aggregation_method == "Product":
                grouped_df = df.groupby(selected_columns)[agg_column].prod().reset_index()
        
        st.write("### Grouped Data")
        st.dataframe(grouped_df.head())
        
        # User selects type of plot
        plot_type = st.selectbox("Select plot type:", ["Bar", "Line", "Pie"])
        
        # User selects visualization library
        library_choice = st.selectbox("Select visualization library:", ["Matplotlib", "Seaborn"])
        
        # Generate plot
        fig, ax = plt.subplots()

        col_chart, col_exp = st.columns(2)

        with col_chart:

            

            ent, srt = st.columns(2)
            
            with ent:
                grp_entries = st.pills('select entries', options=entry_options, key='grp_entries')
                grouped_df = grouped_df.head(grp_entries)

            with srt:
                sorting_data = ['a-z', 'z-a']
                sort_data = st.pills('sort data', options=sorting_data, key='sort_data')
                
                if sort_data == 'a-z':
                    grouped_df = grouped_df.head(grp_entries).sort_values(by=agg_column, ascending=True)
                elif sort_data == 'z-a': 
                    grouped_df = grouped_df.head(grp_entries).sort_values(by=agg_column, ascending=False)

            
        
            if plot_type == "Bar":
                if library_choice == "Matplotlib":
                    plt.figure(figsize=(15, 10))
                    plt.title(f'Bar Graph for the {selected_columns} grouped into {agg_column}')
                    ax.barh(grouped_df[selected_columns[0]], grouped_df[agg_column])
                else:
                    plt.figure(figsize=(18, 15))
                    sb.barplot(data=grouped_df.head(grp_entries), x=selected_columns[0], y=agg_column, ax=ax)
                    plt.xticks(rotation=-90)
            elif plot_type == "Line":
                if library_choice == "Matplotlib":
                    ax.plot(grouped_df[selected_columns[0]], grouped_df[agg_column])
                else:
                    sb.lineplot(data=grouped_df, x=selected_columns[0], y=agg_column, ax=ax)
                    
            elif plot_type == "Pie":
                ax.pie(grouped_df[agg_column], labels=grouped_df[selected_columns[0]], autopct='%1.1f%%')
            
            st.pyplot(fig)

       

        # with col_exp:
        #     st.subheader('''The Explainer''')


        #     prompt = (
        #         f"You have selected the '{selected_columns}' and are analyzing the '{aggregation_method}' aggregation method. "
        #         f"This graph provides insights into how '{grouped_df[agg_column]}' varies within '{selected_columns}'.\n\n"
        #         f"Key Insights:\n"
        #     )

        #     the_explainer(prompt)


    selected_columns = st.multiselect("Select columns for EDA:", df.columns)

    if selected_columns:
        # Generate insights using Gemini LLM
        with st.spinner("Generating insights..."):
            insights = get_gemini_insights(df, selected_columns)
        
        st.write("### AI-Powered Insights")
        st.write(insights)

        # Generate plots
        st.write("### Data Visualizations")
        for col in selected_columns:
            plt.figure(figsize=(8,10))
            fig, ax = plt.subplots()
            sb.histplot(df[col], kde=True, ax=ax)
            plt.title(f"Distribution of {col}")
            st.pyplot(fig)

        # Generate and provide PDF download
        pdf_output = generate_pdf(df, selected_columns, insights)
        st.download_button(label="Download EDA Report as PDF", data=pdf_output, file_name="EDA_Report.pdf", mime="application/pdf")

    chat_inp, chat_outp = st.columns(2)

    with chat_inp:

         # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

        # Display chat history
        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("How may I help?"):
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            chat_output = get_chat_response(prompt)
            
            # Append AI response
            with st.chat_message("assistant"):
                st.markdown(chat_output)

            st.session_state.messages.append({"role": "assistant", "content": chat_output})


