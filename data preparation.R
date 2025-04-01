# Import required library
library(dplyr)
library(ggplot2)
library(recommenderlab)
library(readxl)
library(corrplot)
library(reshape2)
library(tidyverse)
library(plotly)
library(Rtsne)        
library(scales) 
library (caret)
library(data.table)
library(DT)
library(patchwork)

# Check if package is installed
"dplyr" %in% installed.packages()
"ggplot2" %in% installed.packages()
"recommenderlab" %in% installed.packages()
"reshape2" %in% installed.packages()
"readxl" %in% installed.packages()
"corrplot" %in% installed.packages()
"tidyverse" %in% installed.packages()
"plotly" %in% installed.packages()
"Rtsne" %in% installed.packages()
"scales" %in% installed.packages()
"caret" %in% installed.packages()
"data.table" %in% installed.packages()
"DT" %in% installed.packages()
"patchwork" %in% installed.packages()

# Receiving the data
getwd()

# Be sure to move the file from Downloads to Documents. Adjust the file directory if needed
setwd("C:/Users/mredw/OneDrive/Documents/data-science-assignment-main")
spotify_data <- read.csv("data.csv")
genre_data <- read.csv("data_by_genres.csv")
year_data <- read.csv("data_by_year.csv")
list.files()


View(spotify_data)
View(genre_data)
View(year_data)

#Viewing the Structure of data
str(spotify_data)
str(genre_data)
str(year_data)

#Tabular view
datatable(spotify_data)
datatable(genre_data)
datatable(year_data)


#summary statistic
summary(spotify_data)
summary(genre_data)
summary(year_data)


#DATA PREPARATION/ PREPROCESS
#Normalization, Handling missing values, and Outlier Treatment

#Handling missing values
colSums(is.na(spotify_data)) #returns all zero
colSums(is.na(year_data))  #returns all zero
colSums(is.na(genre_data))  #returns all zero

#None missing values needs to be handled in spotify_data, year_data, genre_data





#Normalization
normalize_columns <- function(df, cols) {
  df[cols] <- lapply(df[cols], function(x) {
    (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  })
  return(df)
}

audio_features <- c('acousticness', 'danceability', 'energy', 'instrumentalness', 
                    'liveness', 'valence', 'speechiness', 'tempo', 'loudness')

spotify_data <- normalize_columns(spotify_data, intersect(audio_features, names(spotify_data)))
year_data <- normalize_columns(year_data, intersect(audio_features, names(year_data)))
genre_data <- normalize_columns(genre_data, intersect(audio_features, names(genre_data)))


#Verify with different variables 
summary(spotify_data$acousticness)  # Should show Min = 0, Max = 1
summary(spotify_data$danceability) 
summary(spotify_data$energy) 
summary(spotify_data$instrumentalness) 
summary(spotify_data$liveness) 
summary(spotify_data$valence) 
summary(spotify_data$tempo) 
summary(spotify_data$loudness) 

#End of Normalization




#Outlier Treatment


cap_outliers <- function(df, cols) {
  df[cols] <- lapply(df[cols], function(x) {
    qnt <- quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
    caps <- quantile(x, probs = c(0.05, 0.95), na.rm = TRUE)
    iqr <- 1.5 * IQR(x, na.rm = TRUE)
    x[x < (qnt[1] - iqr)] <- caps[1]
    x[x > (qnt[2] + iqr)] <- caps[2]
    x
  })
  return(df)
}



# Select a column with outliers (e.g., 'tempo' or 'loudness')
target_col <- "tempo"

# Create before/after data
original_data <- spotify_data[[target_col]]
capped_data <- cap_outliers(spotify_data, target_col)[[target_col]]

# Create data frames for plotting
plot_data <- data.frame(
  value = c(original_data, capped_data),
  type = rep(c("Original", "Capped"), each = length(original_data))
)

# Create boxplots
p1 <- ggplot(plot_data, aes(x = type, y = value, fill = type)) +
  geom_boxplot() +
  labs(title = paste("Outlier Treatment:", target_col),
       x = "",
       y = target_col) +
  theme_minimal() +
  scale_fill_manual(values = c("Original" = "lightblue", "Capped" = "salmon"))

# Create density plots for comparison
p2 <- ggplot(plot_data, aes(x = value, fill = type)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution Comparison",
       x = target_col,
       y = "Density") +
  theme_minimal() +
  scale_fill_manual(values = c("Original" = "lightblue", "Capped" = "salmon"))

# Combine plots
combined_plot <- p1 + p2 + plot_layout(ncol = 1)

# Display the plot
print(combined_plot)

spotify_data <- cap_outliers(spotify_data, audio_features)
year_data <- cap_outliers(year_data, audio_features)
genre_data <- cap_outliers(genre_data, audio_features)


#End of Outlier Treatment







#Feature correaltion with dependent variable
# Assuming 'data' is your dataframe with the same structure as in Python
feature_names <- c('acousticness', 'danceability', 'energy', 'instrumentalness',
                   'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                   'duration_ms', 'explicit', 'key', 'mode', 'year')

# Select only the features we want to analyze
X <- spotify_data[, feature_names]
y <- spotify_data$popularity

# Calculate correlation matrix
correlation_matrix <- cor(cbind(X, popularity = y), method = "pearson")

# Visualize the correlation matrix
corrplot(correlation_matrix, method = "circle", type = "upper", 
         tl.col = "black", tl.srt = 45, 
         title = "Feature Correlation with Popularity",
         mar = c(0, 0, 1, 0))

# Alternative visualization with ggplot2
# First reshape the correlation matrix
cor_df <- as.data.frame(as.table(correlation_matrix))
names(cor_df) <- c("Var1", "Var2", "Correlation")

# Filter to only show correlations with popularity
popularity_cor <- cor_df %>%
  filter(Var2 == "popularity" & Var1 != "popularity") %>%
  arrange(desc(abs(Correlation)))

# Create bar plot
ggplot(popularity_cor, aes(x = reorder(Var1, Correlation), y = Correlation, fill = Correlation)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                       midpoint = 0, limits = c(-1, 1)) +
  coord_flip() +
  labs(title = "Feature Correlation with Popularity",
       x = "Feature",
       y = "Correlation Coefficient") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14))


#end of Feature correlation with Popularity


#Data Understanding by Visualization 
#How Music Evolve over the Years

# Function to get decade from year
get_decade <- function(year) {
  period_start <- floor(year / 10) * 10
  decade <- paste0(period_start, "s")
  return(decade)
}

spotify_data$decade <- sapply(spotify_data$year, get_decade) #Creating new column "decade" by applying the function in line 114

# Create countplot
library(ggplot2)
ggplot(spotify_data, aes(x = decade)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Song Count by Decade", 
       x = "Decade", 
       y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5))
#end of Countplot

#Plotly line graph
sound_features <- c('acousticness', 'danceability', 'energy', 
                    'instrumentalness', 'liveness', 'valence')# Define the sound features

# Creating the plotly graph using year_data 
fig <- year_data %>%
  plot_ly() %>%
  add_trace(x = ~year, y = ~acousticness, name = 'Acousticness', type = 'scatter', mode = 'lines') %>%
  add_trace(x = ~year, y = ~danceability, name = 'Danceability', type = 'scatter', mode = 'lines') %>%
  add_trace(x = ~year, y = ~energy, name = 'Energy', type = 'scatter', mode = 'lines') %>%
  add_trace(x = ~year, y = ~instrumentalness, name = 'Instrumentalness', type = 'scatter', mode = 'lines') %>%
  add_trace(x = ~year, y = ~liveness, name = 'Liveness', type = 'scatter', mode = 'lines') %>%
  add_trace(x = ~year, y = ~valence, name = 'Valence', type = 'scatter', mode = 'lines') %>%
  layout(
    title = "Audio Features Over Time",
    xaxis = list(title = "Year"),
    yaxis = list(title = "Feature Value"),
    hovermode = "x unified"
  )

fig  # Displaying the plot


#Identifying the characteristic of genre using genre_data
# Get top 10 genres
top10_genres <- genre_data %>%
  arrange(desc(popularity)) %>%
  slice(1:10)

# Reshape data for grouped bars
top10_long <- top10_genres %>%
  pivot_longer(
    cols = c(valence, energy, danceability, acousticness),
    names_to = "feature",
    values_to = "value"
  )

# Plot with ggplot2
ggplot(top10_long, aes(x = genres, y = value, fill = feature)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "Top 10 Genres by Popularity",
    x = "Genres",
    y = "Feature Value",
    fill = "Feature"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-labels for readability