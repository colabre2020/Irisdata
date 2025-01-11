# Author : SP

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load Iris dataset
def load_iris_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    return df, iris

# Load Wine dataset
def load_wine_data():
    wine = datasets.load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['class'] = wine.target_names[wine.target]
    return df, wine

# Load Digits dataset
def load_digits_data():
    digits = datasets.load_digits()
    df = pd.DataFrame(digits.data)
    df['target'] = digits.target
    return df, digits

# Load Breast Cancer dataset
def load_cancer_data():
    cancer = datasets.load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    return df, cancer

# Streamlit UI layout
def app():
    # Title of the app
    st.title("Interactive ML with Multiple Datases from sklearn")

    # Left Pane for Dataset and Task Selection
    st.sidebar.header("Select Dataset and Task")

    # Select Dataset
    dataset_options = ["Iris", "Wine", "Digits", "Breast Cancer"]
    dataset = st.sidebar.selectbox("Select Dataset", dataset_options)

    # Select Task
    task = st.sidebar.selectbox("Select Task", ["EDA", "Classification", "Clustering", "Dimensionality Reduction", "Model Evaluation"])

    # Load Dataset based on user selection
    if dataset == "Iris":
        df, data = load_iris_data()
    elif dataset == "Wine":
        df, data = load_wine_data()
    elif dataset == "Digits":
        df, data = load_digits_data()
    elif dataset == "Breast Cancer":
        df, data = load_cancer_data()

    # Display dataset overview in main area
    st.write(f"### {dataset} Dataset")
    st.write("Here is a preview of the dataset:")
    st.write(df.head())

    # Task-based functionalities
    if task == "EDA":
        st.header("Exploratory Data Analysis (EDA)")

        # Matrix plot (correlation heatmap or scatter matrix)
        plot_type = st.selectbox("Select Plot Type", ["Correlation Heatmap", "Pairwise Scatter Matrix"])

        if plot_type == "Correlation Heatmap":
            st.subheader("Correlation Heatmap")
            # Calculate the correlation matrix (drop non-numeric columns like 'target' or 'species')
            corr_matrix = df.drop('species', axis=1, errors='ignore').drop('target', axis=1, errors='ignore').corr()
            fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig)

        elif plot_type == "Pairwise Scatter Matrix":
            st.subheader("Pairwise Scatter Matrix")
            # Generate pairwise scatter matrix
            fig = px.scatter_matrix(df, dimensions=df.columns[:-1], color="species" if dataset != "Digits" and dataset != "Breast Cancer" else "target", title=f"{dataset} Dataset Pairwise Scatter Matrix")
            st.plotly_chart(fig)

    elif task == "Classification":
        st.header("Classification (K-Nearest Neighbors)")
        # Prepare data
        X = data.data
        y = data.target

        # Split data into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        # Make predictions
        y_pred = knn.predict(X_test)

        # Display results
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    elif task == "Clustering":
        st.header("Clustering (K-Means)")
        # Prepare data
        X = data.data

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=len(set(data.target)), random_state=42)
        clusters = kmeans.fit_predict(X)

        # Display results using Plotly scatter plot
        st.subheader("K-Means Clustering Result")
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=clusters, labels={'x': 'Feature 1', 'y': 'Feature 2'}, title=f"K-Means Clustering of {dataset} Dataset")
        st.plotly_chart(fig)

    elif task == "Dimensionality Reduction":
        st.header("Dimensionality Reduction (PCA)")
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(data.data)

        # Display results using Plotly scatter plot
        st.subheader("PCA Result (2D)")
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=data.target, labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'}, title=f"PCA of {dataset} Dataset")
        st.plotly_chart(fig)

    elif task == "Model Evaluation":
        st.header("Model Evaluation")
        # Prepare data
        X = data.data
        y = data.target

        # Split data into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        # Make predictions
        y_pred = knn.predict(X_test)

        # Display metrics
        st.subheader("Accuracy")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

# Run the app
if __name__ == "__main__":
    app()
