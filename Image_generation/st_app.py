import streamlit as st
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

# Define the FastAPI base URL
API_BASE_URL = "http://127.0.0.1:8000"  # Replace with your FastAPI server URL

# Streamlit app title
st.title("Image Search Microservice")

# Tabs for different functionalities
tab1, tab2 = st.tabs(["Store Images", "Generate Image Search"])

# Tab 1: Store Images
with tab1:
    st.header("Store Images")
    st.write("Upload a CSV file containing image URLs to store images and generate captions.")

    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    num_images = st.number_input("Number of images to process (optional)", min_value=1, step=1, value=10)

    if uploaded_file:
        # Display the uploaded file
        st.write("Uploaded file:")
        st.dataframe(pd.read_csv(uploaded_file))

        # Call the FastAPI `/store_image` endpoint
        if st.button("Store Images"):
            try:
                # Save the uploaded file locally
                with open("uploaded_file.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Make the API request
                response = requests.post(
                    f"{API_BASE_URL}/store_image",
                    json={"file_path": "uploaded_file.csv", "num_images": num_images},
                )

                # Display the response
                if response.status_code == 200:
                    st.success(response.json()["message"])
                else:
                    st.error(f"Error: {response.json()['detail']}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Tab 2: Generate Image Search
with tab2:
    st.header("Generate Image Search")
    st.write("Enter a query to retrieve similar images.")

    # Input for the query
    query = st.text_input("Enter your query")

    # Number of similar images to retrieve
    top_k = st.number_input("Number of similar images to retrieve", min_value=1, step=1, value=5)

    if st.button("Search Images"):
        try:
            # Call the FastAPI `/generate_image` endpoint
            response = requests.post(
                f"{API_BASE_URL}/generate_image",
                json={"query": query, "top_k": top_k},
            )

            # Display the response
            if response.status_code == 200:
                data = response.json()
                st.write(f"Query: {data['query']}")
                st.write("Retrieved Images:")

                # Display retrieved images
                for img_data in data["retrieved_images"]:
                    st.image(img_data["image_path"], caption=img_data["explanation"], use_container_width=True)
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")