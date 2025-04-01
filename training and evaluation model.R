library(FNN)  # Fast Nearest Neighbor Search
library(proxy)  # For cosine similarity

# Load data
clustered_songs <- read.csv("clustered_songs.csv")

# Select numerical features
features <- c("valence", "acousticness", "danceability", "duration_ms", 
              "energy", "instrumentalness", "key", "liveness", "loudness", 
              "mode", "popularity", "speechiness", "tempo")

df <- clustered_songs[, features] 

# ------------------------------
# Step 5: Optimized Hybrid CBF + Clustering
# ------------------------------
recommend_songs <- function(song_id, cluster_type = "kmeans", top_n = 5) {
  clustered_songs <- read.csv("clustered_songs.csv")
  
  # Find the song's cluster
  song_info <- clustered_songs[clustered_songs$id == song_id, , drop = FALSE]
  if (nrow(song_info) == 0) {
    message("Song ID not found in dataset.")
    return(NULL)
  }
  cluster_label <- song_info[[paste0("cluster_", cluster_type)]]
  
  # Load the corresponding cluster file
  cluster_file <- paste0("cluster_", cluster_type, "_", cluster_label, ".csv")
  if (!file.exists(cluster_file)) {
    message("Cluster file not found.")
    return(NULL)
  }
  
  cluster_songs <- read.csv(cluster_file)
  
  # Ensure required columns exist
  required_cols <- c("id", "name", "artists")
  missing_cols <- setdiff(required_cols, colnames(cluster_songs))
  if (length(missing_cols) > 0) {
    message("Missing columns in the dataset: ", paste(missing_cols, collapse = ", "))
    return(NULL)
  }
  
  # Extract song features
  song_features <- cluster_songs[, features, drop = FALSE]
  song_matrix <- as.matrix(song_features)
  
  # Find the song index
  song_index <- which(cluster_songs$id == song_id)
  if (length(song_index) == 0) {
    message("Song ID not found in the cluster file.")
    return(NULL)
  }
  
  # Compute nearest neighbors using KNN
  song_vector <- song_matrix[song_index, , drop = FALSE]
  nn_result <- get.knnx(song_matrix, song_vector, k = top_n + 1)  # +1 to exclude itself
  
  recommended_indices <- nn_result$nn.index[-1]  # Exclude self
  
  # Select only name, artist, and id in the correct order
  recommendations <- cluster_songs[recommended_indices, c("name", "artists", "id"), drop = FALSE]
  return(recommendations)
}

# ------------------------------
# Compute Cosine Similarity in Batches
# ------------------------------
compute_cosine_similarity_batch <- function(df, batch_size = 10000) {
  num_songs <- nrow(df)
  num_batches <- ceiling(num_songs / batch_size)
  similarity_results <- list()
  
  for (i in 1:num_batches) {
    start_index <- ((i - 1) * batch_size) + 1
    end_index <- min(i * batch_size, num_songs)
    
    batch_data <- df[start_index:end_index, , drop = FALSE]
    batch_matrix <- as.matrix(batch_data)
    
    similarity_matrix <- proxy::dist(batch_matrix, method = "cosine")
    similarity_results[[i]] <- as.data.frame(as.matrix(similarity_matrix))
    
    print(paste("Processed batch", i, "of", num_batches))
  }
  
  final_similarity <- do.call(rbind, similarity_results)
  return(final_similarity)
}

# ------------------------------
# Precision@K Evaluation (Popularity-Based)
# ------------------------------
evaluate_precision_at_k <- function(recommendations, input_song_id, threshold = 0.1) {
  # Get the popularity of the input song
  input_popularity <- clustered_songs[clustered_songs$id == input_song_id, "popularity"]
  
  if (length(input_popularity) == 0) {
    message("Input song ID not found in dataset.")
    return(NA)
  }
  
  # Get popularity values of recommended songs
  recommended_popularities <- clustered_songs[clustered_songs$id %in% recommendations$id, "popularity"]
  
  if (length(recommended_popularities) == 0) {
    return(0)  # No valid recommendations
  }
  
  # Define acceptable range (e.g., ±10% similarity threshold)
  lower_bound <- input_popularity * (1 - threshold)
  upper_bound <- input_popularity * (1 + threshold)
  
  # Count how many recommendations fall within this range
  relevant_recommendations <- sum(recommended_popularities >= lower_bound & recommended_popularities <= upper_bound)
  
  # Compute Precision@K
  precision <- relevant_recommendations / length(recommended_popularities)
  
  return(precision)
}

# ------------------------------
# Pick a Random Song & Get Recommendations
# ------------------------------
if ("id" %in% colnames(clustered_songs)) {
  set.seed(Sys.time())
  random_song_id <- sample(clustered_songs$id, 1)
  
  # Ensure required columns exist
  required_cols <- c("id", "name", "artists", "popularity")
  existing_cols <- intersect(required_cols, colnames(clustered_songs))
  
  if (length(existing_cols) < 3) {  # Ensure at least 'id', 'name', and 'popularity' exist
    message("Dataset is missing too many key columns: ", paste(setdiff(required_cols, existing_cols), collapse = ", "))
  } else {
    # Print the selected song
    random_song <- clustered_songs[clustered_songs$id == random_song_id, existing_cols, drop = FALSE]
    print(random_song)
    
    # Get recommendations
    recommendations <- recommend_songs(random_song_id)
    print(recommendations)
    
    # Evaluate Precision@K
    precision_score <- evaluate_precision_at_k(recommendations, random_song_id)
    print(paste("Precision@K:", precision_score))
  }
} else {
  message("Column 'id' not found in the dataset.")
}

