import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def load_data(training_path, testing_path):
    """Load the dataset from csv files."""
    ALS_TestingData = pd.read_csv(testing_path)
    df = pd.read_csv(training_path)
    return ALS_TestingData, df

def preprocess_data(df):
    """Preprocess the data by dropping non-informative columns and scaling."""
    df = df.drop(['ID', 'SubjectID'], axis=1)
    SS_scaler = StandardScaler() 
    df_scaled = SS_scaler.fit_transform(df)
    
    # Normalize data
    df_normalized = normalize(df_scaled)
    
    # Apply PCA
    pca = PCA(2)
    scaled_pca = pca.fit_transform(df_normalized)
    return df, scaled_pca

def plot_clusters(scaled_pca, kmeans, title):
    """Plot clusters determined by KMeans."""
    label_prediction = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    plt.figure(figsize=(10, 7))
    for i in range(kmeans.n_clusters):
        cluster_data = scaled_pca[label_prediction == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}')
        
    plt.scatter(centers[:, 0], centers[:, 1], marker="*", color='r', label='Centers')
    plt.title(f'{kmeans.n_clusters} clusters for normalized data with PCA applied')
    plt.legend()
    plt.show()

def kmeans_clustering(df, scaled_pca, num_clusters):
    """Perform KMeans clustering and plot the clusters."""
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans.fit(scaled_pca)
    
    df['kmean'] = kmeans.labels_
    df['kmean'].value_counts()
    
    plot_clusters(scaled_pca, kmeans, f'{num_clusters} clusters for normalized data with PCA applied')
    
    return kmeans

def feature_importance(df_scaled):
    """Plot feature importance using a RandomForestClassifier."""
    X, y = df_scaled.iloc[:, :-1], df_scaled.iloc[:, -1]
    clf = RandomForestClassifier(n_estimators=100).fit(X, y)
    importances = clf.feature_importances_
    
    # Convert importances and features to a DataFrame for easy sorting 
    data = np.array([clf.feature_importances_, X.columns]).T
    columns = list(pd.DataFrame(data, columns=['Importance', 'Feature'])
           .sort_values("Importance", ascending=False)
           .head(7).Feature.values)
    tidy = df_scaled[columns+['kmean']].melt(id_vars='kmean')
    sns.barplot(x='kmean', y='value', hue='variable', data=tidy)
    plt.show()

def find_optimal_k(scaled_pca):
    """Run KMeans with different K values and plot the SSE to find the optimal K."""
    k_rng = range(1, 10)
    sse = []
    for k in k_rng:
        km = KMeans(n_clusters=k, n_init='auto')
        km.fit(scaled_pca)
        sse.append(km.inertia_)
    
    plt.figure(figsize=(10, 7))
    plt.title('Elbow method for selecting optimal K')
    plt.xlabel('K')
    plt.ylabel('Sum of squared error')
    plt.plot(k_rng, sse)
    plt.axvline(x=3, color='r', linestyle='--')
    plt.show()

def main():
    # Load data
    training_path = 'C:\\Users\\eriks\\OneDrive - Østfold University College\\Bachelor\\År 3\\Semester_1\\Praktisk_Machine_learning\\Assignments\\Assignment_3\\ALS_TrainingData_2223.csv'
    testing_path = 'C:\\Users\\eriks\\OneDrive - Østfold University College\\Bachelor\\År 3\\Semester_1\\Praktisk_Machine_learning\\Assignments\Assignment_3\\ALS_TestingData_78.csv'
    ALS_TestingData, df = load_data(training_path, testing_path)

    # Preprocess data
    df, scaled_pca = preprocess_data(df)

    # Find optimal K using Elbow method
    find_optimal_k(scaled_pca)
    
    # Perform clustering with different K values
    num_clusters_list = [2, 3, 4, 8]
    for num_clusters in num_clusters_list:
        kmeans = kmeans_clustering(df, scaled_pca, num_clusters)
   
    # Feature importance plot
    minmax_scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(minmax_scaler.fit_transform(df))
    df_scaled['kmean'] = kmeans.labels_
    feature_importance(df_scaled)

if __name__ == "__main__":
    main()