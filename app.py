import streamlit as st
import pickle
from sklearn.feature_extraction.text import Tfi
Vectorizer

df = pd.read_csv('prod.csv')  # Replace 'your_dataframe.csv' with the path to your dataframe

# Load TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load KNN model for TF-IDF
with open('knn_tfidf.pkl', 'rb') as f:
    knn_tfidf = pickle.load(f)

def recommend_products(product_description):
    # Vectorize input product description
    description_vec = tfidf_vectorizer.transform([product_description])
    
    # Find k-nearest neighbors based on product description
    distances, indices = knn_tfidf.kneighbors(description_vec.toarray())
    
    # Get recommended product indices based on product description
    recommended_indices = indices[0]
    
    # Get recommended products based on product description
    recommended_products = df.iloc[recommended_indices]
    
    return recommended_products[[ 'product_title']]

# Assuming df is your original dataframe containing product details
#df = X_encoded.reset_index(drop=True)  # Reset index for df to match indices in X_encoded

# Streamlit app
st.title('Product Recommendation System')

# Input for product description
product_description = st.text_input('Enter Product Description', '')

# Button to generate recommendations
if st.button('Generate Recommendations'):
    recommended_products = recommend_products(product_description)
    
    st.subheader('Recommended Products based on Product Description:')
    st.write(recommended_products)

