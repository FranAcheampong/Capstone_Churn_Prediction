#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import time
import matplotlib.pyplot as plt
from io import BytesIO


# In[2]:


# Load the saved trained models, transformers, and the scalers
num_imputer = joblib.load('numerical_imputer.joblib')
cat_imputer = joblib.load('cat_imputer.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')
lr_model = joblib.load('lr_smote_model.joblib')
cnb_model = joblib.load('cnb_smote_model.joblib')


# In[10]:


def preprocess_input(input_data):
    input_df = pd.DataFrame(input_data, index=[0])

    cat_columns = [col for col in input_df.columns if input_df[col].dtype == 'object']
    num_columns = [col for col in input_df.columns if input_df[col].dtype != 'object']

    input_df_imputed_cat = cat_imputer.transform(input_df[cat_columns])
    input_df_imputed_num = num_imputer.transform(input_df[num_columns])

    input_encoded_df = pd.DataFrame(encoder.transform(input_df_imputed_cat).toarray(),
                                    columns=encoder.get_feature_names_out(cat_columns))

    input_df_scaled = scaler.transform(input_df_imputed_num)
    input_scaled_df = pd.DataFrame(input_df_scaled, columns=num_columns)
    final_df = pd.concat([input_encoded_df, input_scaled_df], axis=1)
    final_df = final_df.reindex(columns=original_feature_names, fill_value=0)

    return final_df


# In[4]:


original_feature_names = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE',
                          'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 'REGULARITY', 'FREQ_TOP_PACK',
                          'REGION_DAKAR', 'REGION_DIOURBEL', 'REGION_FATICK', 'REGION_KAFFRINE', 'REGION_KAOLACK',
                          'REGION_KEDOUGOU', 'REGION_KOLDA', 'REGION_LOUGA', 'REGION_MATAM', 'REGION_SAINT-LOUIS',
                          'REGION_SEDHIOU', 'REGION_TAMBACOUNDA', 'REGION_THIES', 'REGION_ZIGUINCHOR',
                          'TENURE_Long-term', 'TENURE_Medium-term', 'TENURE_Mid-term', 'TENURE_Short-term',
                          'TENURE_Very short-term', 'TOP_PACK_VAS', 'TOP_PACK_data', 'TOP_PACK_international',
                          'TOP_PACK_messaging', 'TOP_PACK_other_services', 'TOP_PACK_social_media',
                          'TOP_PACK_voice']

# Set up the Streamlit app
st.set_page_config(layout="wide")

# Main page - Churn Prediction
st.title('CUSTOMER CHURN PREDICTION APP ')

# Main page - Churn Prediction
st.markdown("Churn is a one of the biggest problem in the telecom industry. Research has shown that the average monthly churn rate among the top 4 wireless carriers in the US is 1.9% - 2%")
st.image("bg.png", use_column_width=True)

    # How to use
st.sidebar.image("welcome.jpg", use_column_width=True)
st.sidebar.title("ENTER THE DETAILS OF THE CUSTOMER HERE")

# Define a dictionary of models with their names, actual models, and types
models = {
    'Logistic Regression': {'Logistic Regression': lr_model, 'type': 'logistic_regression'},
    #'ComplementNB': {'ComplementNB': cnb_model, 'type': 'Complement NB'}
}

# Allow the user to select a model from the sidebar
model_name = st.sidebar.selectbox('Select Model', list(models.keys()))

# Retrieve the selected model and its type from the dictionary
model = models[model_name]['Logistic Regression']
model_type = models[model_name]['type']


# Collect input from the user
st.sidebar.title('Enter Customer Details')
input_features = {
    'MONTANT': st.sidebar.number_input('Top-up Amount (MONTANT)'),
    'FREQUENCE_RECH': st.sidebar.number_input('Number of Times the Customer Refilled (FREQUENCE_RECH)'),
    'REVENUE': st.sidebar.number_input('Monthly income of the client (REVENUE)'),
    'ARPU_SEGMENT': st.sidebar.number_input('Income over 90 days / 3 (ARPU_SEGMENT)'),
    'FREQUENCE': st.sidebar.number_input('Number of times the client has made an income (FREQUENCE)'),
    'DATA_VOLUME': st.sidebar.number_input('Number of Connections (DATA_VOLUME)'),
    'ON_NET': st.sidebar.number_input('Inter Expresso Call (ON_NET)'),
    'ORANGE': st.sidebar.number_input('Call to Orange (ORANGE)'),
    'TIGO': st.sidebar.number_input('Call to Tigo (TIGO)'),
    'ZONE1': st.sidebar.number_input('Call to Zone 1 (ZONE1)'),
    'ZONE2': st.sidebar.number_input('Call to Zone 2 (ZONE2)'),
    'REGULARITY': st.sidebar.number_input('Number of Times the Client is Active for 90 Days (REGULARITY)'),
    'FREQ_TOP_PACK': st.sidebar.number_input('Number of Times the Client has Activated the Top Packs (FREQ_TOP_PACK)'),
    'REGION': st.sidebar.selectbox('Location of Each Client (REGION)', ['DAKAR','DIOURBEL','FATICK','AFFRINE','KAOLACK',
                                                                        'KEDOUGOU','KOLDA','LOUGA','MATAM','SAINT-LOUIS',
                                                                        'SEDHIOU','TAMBACOUNDA','HIES','ZIGUINCHOR' ]),
                                                                       
    'TENURE': st.sidebar.selectbox('Duration in the Network (TENURE)', ['Long-term','Medium-term','Mid-term','Short-term',
                                                                       'Very short-term']),
    'TOP_PACK': st.sidebar.selectbox('Most Active Pack (TOP_PACK)', ['VAS', 'data', 'international', 
                                                                     'messaging','other_services', 'social_media',
                                                                     'voice'])
                                                                      
}

# Input validation
valid_input = True
error_messages = []

# Validate numeric inputs
numeric_ranges = {
    'MONTANT': [0, 1000000],
    'FREQUENCE_RECH': [0, 100],
    'REVENUE': [0, 1000000],
    'ARPU_SEGMENT': [0, 100000],
    'FREQUENCE': [0, 100],
    'DATA_VOLUME': [0, 100000],
    'ON_NET': [0, 100000],
    'ORANGE': [0, 100000],
    'TIGO': [0, 100000],
    'ZONE1': [0, 100000],
    'ZONE2': [0, 100000],
    'REGULARITY': [0, 100],
    'FREQ_TOP_PACK': [0, 100]
}

for feature, value in input_features.items():
    range_min, range_max = numeric_ranges.get(feature, [None, None])
    if range_min is not None and range_max is not None:
        if not range_min <= value <= range_max:
            valid_input = False
            error_messages.append(f"{feature} should be between {range_min} and {range_max}.")

#Churn Prediction

def predict_churn(input_data, model):
    # Preprocess the input data
    preprocessed_data = preprocess_input(input_data)

     # Calculate churn probabilities using the model
    probabilities = model.predict_proba(preprocessed_data)

    # Determine churn labels based on the model type
    if model_type == "logistic_regression":
        churn_labels = ["No Churn", "Churn"]
    #elif model_type == "ComplementNB":
        churn_labels = ["Churn", "No Churn"]
    # Extract churn probability for the first sample
    churn_probability = probabilities[0]

    # Create a dictionary mapping churn labels to their indices
    churn_indices = {label: idx for idx, label in enumerate(churn_labels)}

    # Determine the index with the highest churn probability
    churn_index = np.argmax(churn_probability)

    # Return churn labels, churn probabilities, churn indices, and churn index
    return churn_labels, churn_probability, churn_indices, churn_index

# Predict churn based on user input
if st.sidebar.button('Predict Churn'):
    try:
        with st.spinner("Wait, Results loading..."):
        # Simulate a long-running process
            progress_bar = st.progress(0)
            step = 20  # A big step will reduce the execution time
            for i in range(0, 100, step):
                time.sleep(0.1)
                progress_bar.progress(i + step)

                #churn_labels, churn_probability = predict_churn(input_features, model)  # Pass model1 or model2 based on the selected model
        churn_labels, churn_probability, churn_indices, churn_index = predict_churn(input_features, model)

        st.subheader('CHURN PREDICTION RESULTS')



        col1, col2 = st.columns(2)

        if churn_labels[churn_index] == "Churn":
            churn_prob = churn_probability[churn_index]
            with col1:
                st.error(f"CHURN ALERT! This customer is likely to churn with a probability of {churn_prob * 100:.2f}% 😢")
                resized_churn_image = Image.open('Churn.jpeg')
                resized_churn_image = resized_churn_image.resize((350, 300))  # Adjust the width and height as desired
                st.image(resized_churn_image)
                # Add suggestions for retaining churned customers in the 'Churn' group 
            with col2:
                st.info("ADVICE TO EXPRESSOR MANAGEMENT:\n"
                    "- Identify Reasons for Churn\n"
                    "- Offer Incentives\n"
                    "- Showcase Improvements\n"
                    "- Gather Feedback\n"
                    "- Customer Surveys\n"
                    "- Personalized Recommendations\n"
                    "- Reestablish Trust\n"
                    "- Follow-Up Communication\n"
                    "- Reactivation Campaigns\n"
                    "- Improve product or service offerings based on customer feedback\n"
                    "- SUMMARY NOTE\n"
                    "- Remember that winning back churning customers takes time and persistence.\n" 
                    "- It\s crucial to genuinely address their concerns and provide value to rebuild their trust in your business\n" 
                    "- Regularly evaluate the effectiveness of your strategies and adjust them as needed based on customer responses and feedback\n")
        else:
            #churn_index = churn_indices["No Churn"]
            churn_prob = churn_probability[churn_index]
            with col1:
                st.success(f"This customer is not likely to churn with a probability of {churn_prob * 100:.2f}% 😀")
                resized_not_churn_image = Image.open('NotChurn.jpeg')
                resized_not_churn_image = resized_not_churn_image.resize((350, 300))  # Adjust the width and height as desired
                st.image(resized_not_churn_image)
                # Add suggestions for retaining churned customers in the 'Churn' group
            with col2:
                st.info("ADVICE TO EXPRESSOR MANAGEMENT:\n"
                    "- Quality Products/Services\n"
                    "- Personalized Experience\n"
                    "- Loyalty Programs\n"
                    "- Excellent Customer Service\n"
                    "- Exclusive Content\n"
                    "- Early Access\n"
                    "- Personal Thank-You Notes\n"
                    "- Surprise Gifts or Discounts\n"
                    "- Feedback Opportunities\n"
                    "- Community Engagement\n"
                    "- Anniversary Celebrations\n"
                    "- Refer-a-Friend Programs\n"
                    "- SUMMARY NOTE\n" 
                    "- Remember that the key to building lasting loyalty is consistency.\n" 
                    "- Continuously demonstrate your commitment to meeting customers needs and enhancing their experience."
                    "- Regularly assess the effectiveness of your loyalty initiatives and adapt them based on customer feedback and preferences.")

        st.subheader('Churn Probability')

        # Create a donut chart to display probabilities
        fig = go.Figure(data=[go.Pie(
            labels=churn_labels,
            values=churn_probability,
            hole=0.5,
            textinfo='label+percent',
            marker=dict(colors=['#FFA07A', '#6495ED', '#FFD700', '#32CD32', '#FF69B4', '#8B008B']))])

        fig.update_traces(
            hoverinfo='label+percent',
            textfont_size=12,
            textposition='inside',
            texttemplate='%{label}: %{percent:.2f}%'
            )

        fig.update_layout(
            title='Churn Probability',
            title_x=0.5,
            showlegend=False,
            width=500,
            height=500
            )

        st.plotly_chart(fig, use_container_width=True)

        # Calculate the average churn rate (replace with your actual value)

        st.subheader('Customer Churn Probability Comparison')

        average_churn_rate = 19

        # Convert the overall churn rate to churn probability
        main_data_churn_probability = average_churn_rate / 100

        # Retrieve the predicted churn probability for the selected customer
        predicted_churn_prob = churn_probability[churn_index]

        if churn_labels[churn_index] == "Churn":
            churn_prob = churn_probability[churn_index]
            # Create a bar chart comparing the churn probability with the average churn rate
            labels = ['Churn Probability', 'Average Churn Probability']
            values = [predicted_churn_prob, main_data_churn_probability]

            fig = go.Figure(data=[go.Bar(x=labels, y=values)])
            fig.update_layout(
                xaxis_title='Churn Probability',
                yaxis_title='Probability',
                title='Comparison with Average Churn Rate',
                yaxis=dict(range=[0, 1])  # Set the y-axis limits between 0 and 1
            )

            # Add explanations
            if predicted_churn_prob > main_data_churn_probability:
                churn_comparison = "higher"
            elif predicted_churn_prob < main_data_churn_probability:
                churn_comparison = "lower"
            else:
                churn_comparison = "equal"

            explanation = f"This bar chart compares the churn probability of the selected customer "                             f"with the average churn rate of all customers. It provides insights into how the "                             f"individual customer's churn likelihood ({predicted_churn_prob:.2f}) compares to the "                             f"overall trend. The 'Churn Probability' represents the likelihood of churn "                             f"for the selected customer, while the 'Average Churn Rate' represents the average "                             f"churn rate across all customers ({main_data_churn_probability:.2f}).\n\n"                             f"The customer's churn rate is {churn_comparison} than the average churn rate."

            st.plotly_chart(fig)
            st.write(explanation)
        else:
    # Create a bar chart comparing the no-churn probability with the average churn rate
            labels = ['No-Churn Probability', 'Average Churn Probability']
            values = [1 - predicted_churn_prob, main_data_churn_probability]

            fig = go.Figure(data=[go.Bar(x=labels, y=values)])
            fig.update_layout(
                xaxis_title='Churn Probability',
                yaxis_title='Probability',
                title='Comparison with Average Churn Rate',
                yaxis=dict(range=[0, 1])  # Set the y-axis limits between 0 and 1
            )

            explanation = f"This bar chart compares the churn probability of the selected customer "                             f"with the average churn rate of all customers. It provides insights into how the "                             f"individual customer's churn likelihood ({predicted_churn_prob:.2f}) compares to the "                             f"overall trend."                             f"The prediction indicates that the customer is not likely to churn. "                         f"The churn probability is lower than the no-churn probability."

            st.plotly_chart(fig)
            st.write(explanation)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




