# Load necessary libraries
library(ggplot2)
library(dplyr)
library(cluster)
library(factoextra)
library(clusterSim)

# Load dataset
setwd("C:/Users/mredw/OneDrive/Documents/data-science-assignment-main")
spotify_data <- read.csv("data.csv")

# Select numerical features
features <- c("valence", "acousticness", "danceability", "duration_ms", 
              "energy", "instrumentalness", "key", "liveness", "loudness", 
              "mode", "popularity", "speechiness", "tempo")

df <- spotify_data[, features]  # Ensure 'data' is 'spotify_data'

# ------------------------------
# Proper Feature Scaling (Z-score Standardization)
# ------------------------------
df_scaled <- scale(df)  # Standardization (better than Min-Max for K-Means)

# ------------------------------
# Apply PCA for Dimensionality Reduction
# ------------------------------
pca_result <- prcomp(df_scaled, center = TRUE, scale. = TRUE)
explained_variance <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))

# Choose number of components to retain 95% variance
num_components <- which(explained_variance >= 0.95)[1]
df_pca <- as.data.frame(pca_result$x[, 1:num_components])

# ------------------------------
# Step 1: Auto-Test Multiple k Values (K-Means)
# ------------------------------
set.seed(42)
k_range <- 2:20  
results <- data.frame(k = integer(), silhouette = numeric(), davies_bouldin = numeric())

for (k in k_range) {
  kmeans_result <- kmeans(df_pca, centers = k, nstart = 10, iter.max = 300, algorithm = "Lloyd")
  
  # Reduce dataset size for Silhouette Score
  sample_size <- min(10000, nrow(df_pca))  # Limit to 10,000 samples
  sample_indices <- sample(1:nrow(df_pca), sample_size)
  df_sample <- df_pca[sample_indices, ]
  sample_clusters <- kmeans_result$cluster[sample_indices]
  
  # Compute Silhouette Score on sample
  sil_sample <- silhouette(sample_clusters, dist(df_sample))
  avg_silhouette <- mean(sil_sample[, 3])
  
  # Compute Davies-Bouldin Index
  db_index <- index.DB(df_sample, sample_clusters)$DB
  
  # Store results
  results <- rbind(results, data.frame(k = k, silhouette = avg_silhouette, davies_bouldin = db_index))
  
  message(paste("Tested k =", k, "| Silhouette:", round(avg_silhouette, 3), "| DB Index:", round(db_index, 3)))
}

# Plot Silhouette Score vs k
ggplot(results, aes(x = k, y = silhouette)) +
  geom_line() + geom_point() +
  labs(title = "Silhouette Score vs Number of Clusters",
       x = "Number of Clusters (k)", y = "Silhouette Score") +
  theme_minimal()

# Plot Davies-Bouldin Index vs k
ggplot(results, aes(x = k, y = davies_bouldin)) +
  geom_line() + geom_point() +
  labs(title = "Davies-Bouldin Index vs Number of Clusters",
       x = "Number of Clusters (k)", y = "Davies-Bouldin Index") +
  theme_minimal()

# ------------------------------
# Step 2: Choose Best k Based on Metrics
# ------------------------------
optimal_k <- results$k[which.max(results$silhouette)]  # Best k based on silhouette

message(paste("Optimal k chosen:", optimal_k))

# ------------------------------
# Step 3: Final Clustering with Optimal k (K-Means)
# ------------------------------
set.seed(42)
kmeans_result <- kmeans(df_pca, centers = optimal_k, nstart = 10, iter.max = 300, algorithm = "Lloyd")
spotify_data$cluster_kmeans <- as.factor(kmeans_result$cluster)

# ------------------------------
# Step 4: PCA for Visualization
# ------------------------------
spotify_data$pca_x <- df_pca[, 1]
spotify_data$pca_y <- df_pca[, 2]

# Plot K-Means Clusters
ggplot(spotify_data, aes(x = pca_x, y = pca_y, color = cluster_kmeans)) +
  geom_point(alpha = 0.7) +
  labs(title = paste("PCA Projection with k =", optimal_k, " (K-Means)")) +
  theme_minimal()

# ------------------------------
# Additional Scatter Plots
# ------------------------------
# Scatter plot of Danceability vs Energy, colored by K-Means clusters
ggplot(spotify_data, aes(x = danceability, y = energy, color = cluster_kmeans)) +
  geom_point(alpha = 0.7) +
  labs(title = "Danceability vs Energy (K-Means Clusters)",
       x = "Danceability", y = "Energy") +
  scale_color_brewer(palette = "Set1") +
  theme_minimal() +
  theme(legend.title = element_blank())

# Scatter plot of Valence vs Loudness, colored by K-Means clusters
ggplot(spotify_data, aes(x = valence, y = loudness, color = cluster_kmeans)) +
  geom_point(alpha = 0.7) +
  labs(title = "Valence vs Loudness (K-Means Clusters)",
       x = "Valence", y = "Loudness") +
  scale_color_brewer(palette = "Set1") +
  theme_minimal() +
  theme(legend.title = element_blank())

# ------------------------------
# Save results
# ------------------------------
write.csv(results, "clustering_results.csv", row.names = FALSE)
write.csv(spotify_data, "clustered_songs.csv", row.names = FALSE)

# Print clustering result
# Ensure 'cluster_kmeans' column exists
if (!"cluster_kmeans" %in% colnames(spotify_data)) {
  stop("Error: 'cluster_kmeans' column not found in the dataset!")
}

# Save each K-Means cluster into separate files
unique_clusters <- unique(spotify_data$cluster_kmeans)

for (cl in unique_clusters) {
  cluster_subset <- subset(spotify_data, cluster_kmeans == cl)
  filename <- paste0("cluster_kmeans_", cl, ".csv")
  write.csv(cluster_subset, filename, row.names = FALSE)
}

message("Cluster files saved successfully!")
