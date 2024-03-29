import time
import streamlit as st
import requests
import gensim.downloader as api
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from wordcloud import WordCloud
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import to_hex
import matplotlib.cm as cm


def plot_embeddings_with_clusters(name, embeddings, method='kmeans'):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=4, random_state=0).fit(embeddings)
        cluster_labels = kmeans.labels_
    elif method == 'dbscan':
        eps = 1  # Minimum distance between points to be considered neighbors (adjust as needed)
        min_samples = 20  # Minimum number of samples in a neighborhood for a point to be a core point (adjust as needed)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        cluster_labels = dbscan.labels_

    tsne = TSNE(n_components=2, random_state=0)  # Reduce to 2D for visualization
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 6))
    
    # use plt.scatter to plot the clusters with their labels
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=cluster_labels, s=20, cmap='viridis')
    
    # show legend with cluster labels
    plt.legend()

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    # plt.legend()
    plt.grid(True)

    # Display the plot in Streamlit
    st.write(name)
    st.pyplot(plt)
    
    return cluster_labels

# Ask for user input
field_of_study_options = [
    "Computer Science",
    "Medicine",
    "Chemistry",
    "Biology",
    "Materials Science",
    "Physics",
    "Geology",
    "Psychology",
    "Art",
    "History",
    "Geography",
    "Sociology",
    "Business",
    "Political Science",
    "Economics",
    "Philosophy",
    "Mathematics",
    "Engineering",
    "Environmental Science",
    "Agricultural and Food Sciences",
    "Education",
    "Law",
    "Linguistics"
]


# Example list of supported fields
field_of_study = st.selectbox("Select your field of study:", field_of_study_options)
years = st.text_input("Enter the range of years (e.g., 2010-2023):")
num_articles = st.number_input("Enter the desired number of articles:", min_value=1, max_value=100)

# Add a submit button
if st.button("Submit"):
    # Display user input
    st.write("Your Input:")
    st.write(f"- Field of Study: {field_of_study}")

    if "-" in years:  # Check if valid year range format (e.g., 2010-2023)
        start_year, end_year = years.split("-")
        st.write(f"- Years: {start_year} to {end_year}")
    else:
        st.warning("Invalid year range format. Please enter years in YYYY-YYYY format.")

    st.write(f"- Number of Articles: {num_articles}")

    # Construct API URL

    # Define the endpoint URL
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    # Define the query parameters
    query_params = {
        'query': field_of_study,
        'limit': num_articles,
        'year': years,
        'fields': 'abstract,embedding'
    }

    headers = {'x-api-key': "805K9CNW1O2P4CexBB6z94L5Lh8J7ioR8LFmQHVY"}
    
    response_success = False
    while not response_success:
        # Make the request with the specified parameters
        response = requests.get(url, params=query_params, headers=headers)

        # base_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk?fieldsOfStudy={}&year={}&limit={}"
        # api_url = base_url.format(field_of_study, years, num_articles)

        # Make the API call using requests
        # response = requests.get(api_url)

        # Check for successful response (status code 200)
        if response.status_code != 200:
            st.error(f"API request failed with status code: {response.status_code}")
            st.error(f"Error message: {response.text}")
            st.warning("Retrying in 5 seconds...")
            time.sleep(5)  # Wait for 5 second before retrying
            
        else:    
            response_success = True


    # Process response data (assuming JSON format)
    else:
        data = response.json()['data']  # Assuming JSON response, adjust parsing if needed

        # Extract article abstracts
        abstracts = []
        specter_embeddings = []
        for paper in data:
            if 'abstract' in paper.keys() and 'embedding' in paper.keys():
                if paper['abstract'] is not None and paper['embedding']['vector'] is not None:
                    abstracts.append(paper['abstract'])
                    specter_embeddings.append(paper['embedding']['vector'])
        specter_embeddings = np.array(specter_embeddings)
  

        st.write("Example Abstract:")
        st.write(abstracts[:1])
        st.write("Number of Abstracts Retrieved:")
        st.write(len(abstracts))

        if abstracts:  # Check if articles are retrieved

            # Step 1: Load Pre-trained Word Embedding Model
            model_name = "glove-wiki-gigaword-300"  # Example: GloVe model
            model = api.load(model_name)

            # Step 2: Generate Embeddings
            def embed_abstract(abstract):
                tokens = word_tokenize(abstract.lower())  # Tokenize and lowercase
                embeddings = []
                for token in tokens:
                    if token in model:
                        embeddings.append(model[token])
                if embeddings:
                    return np.mean(embeddings, axis=0)  # Average of word embeddings
                else:
                    return None  # Return None if no embeddings found

            # Step 3: Aggregate Embeddings
            st.write("Embedding abstracts...")
            abstract_embeddings = []
            for abstract in abstracts:
                embedding = embed_abstract(abstract)
                if embedding is not None:
                    abstract_embeddings.append(embedding)
            abstract_embeddings = np.array(abstract_embeddings)
            
            plot_embeddings_with_clusters("DBSCAN Clustering of glove embeddings", abstract_embeddings, method='dbscan')
            plot_embeddings_with_clusters("KMeans Clustering of glove embeddings", abstract_embeddings, method='kmeans')
            plot_embeddings_with_clusters("DBSCAN Clustering of Semantic Scholar SPECTER embeddings", specter_embeddings, method='dbscan')
            cluster_labels = plot_embeddings_with_clusters("KMeans Clustering of Semantic Scholar SPECTER embeddings", specter_embeddings, method='kmeans')
            
            
            # add word cloud for each cluster
            for i in range(len(set(cluster_labels))):
                cluster_abstracts = [abstract for j, abstract in enumerate(abstracts) if cluster_labels[j] == i]
                wordcl = WordCloud().generate(" ".join(cluster_abstracts))
                plt.imshow(wordcl, interpolation='bilinear')
                plt.axis('off')
                st.write(f"Word Cloud for Cluster {i}:")
                st.pyplot(plt)
            

        else:
            st.warning("No articles found in the response.")
