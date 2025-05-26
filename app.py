import streamlit as st
import joblib
import pandas as pd
import numpy as np


# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')


model = load_model()

# App title and description
st.title('Customer Churn Prediction')
st.write("""
Predict whether a customer is likely to churn based on their service details.
""")

# Input form
with st.form('customer_details'):
    st.header('Customer Information')

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input('Monthly Charges ($)', min_value=0, value=70)
        total_charges = st.number_input('Total Charges ($)', min_value=0, value=tenure * monthly_charges)

    with col2:
        contract = st.selectbox('Contract Type',
                                ['Month-to-month', 'One year', 'Two year'])
        internet_service = st.selectbox('Internet Service',
                                        ['DSL', 'Fiber optic', 'No'])
        payment_method = st.selectbox('Payment Method',
                                      ['Electronic check', 'Mailed check',
                                       'Bank transfer', 'Credit card'])

    # Additional features
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])

    submitted = st.form_submit_button('Predict Churn Risk')

# When form is submitted
if submitted:
    # Prepare input data
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract_' + contract.replace(' ', '_'): 1,
        'InternetService_' + internet_service.replace(' ', '_'): 1,
        'PaymentMethod_' + payment_method.replace(' ', '_'): 1,
        'OnlineSecurity_' + online_security.replace(' ', '_'): 1,
        'TechSupport_' + tech_support.replace(' ', '_'): 1,
        'PaperlessBilling_' + paperless_billing: 1
    }

    # Convert to DataFrame with all possible columns (matching training data)
    # You'll need to include all columns your model was trained on
    features = pd.DataFrame(columns=[
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'Contract_Month-to-month', 'Contract_One_year', 'Contract_Two_year',
        'InternetService_DSL', 'InternetService_Fiber_optic', 'InternetService_No',
        # Include all other features your model expects
    ])

    # Fill in the provided values
    for key, value in input_data.items():
        features[key] = [value]

    # Fill missing columns with 0
    features = features.fillna(0)

    # Make prediction
    try:
        probability = model.predict_proba(features)[0][1]
        prediction = model.predict(features)[0]

        # Display results
        st.subheader('Prediction Results')

        if prediction == 1:
            st.error(f'High Churn Risk: {probability:.1%} probability')
            st.write('Recommended actions: Offer discount, loyalty program, or personalized support')
        else:
            st.success(f'Low Churn Risk: {probability:.1%} probability')

        # Show feature importance (if using tree-based model)
        if hasattr(model, 'feature_importances_'):
            st.subheader('Key Factors Influencing This Prediction')
            importances = pd.DataFrame({
                'Feature': features.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)

            for _, row in importances.iterrows():
                st.write(f"- {row['Feature']}: {row['Importance']:.2f}")

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")