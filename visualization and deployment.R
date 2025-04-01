library(shiny)
library(FNN)  # Fast Nearest Neighbor Search
library(dplyr)
library(ggplot2)  # For scatter plot
library(proxy)    # For cosine similarity calculation

# Load data
clustered_songs <- read.csv("clustered_songs.csv")

# Define numerical features for similarity calculation
features <- c("valence", "acousticness", "danceability", "duration_ms", 
              "energy", "instrumentalness", "key", "liveness", "loudness", 
              "mode", "popularity", "speechiness", "tempo")

# Define UI function
define_ui <- function() {
  fluidPage(
    titlePanel("Song Recommendation System"),
    
    sidebarLayout(
      sidebarPanel(
        # Alphabet Picker UI
        selectInput("alphabet", "Pick a Letter:", choices = LETTERS),
        uiOutput("song_selector"),  # Dynamic song selector
        hr(),
        # Scatter Plot Comparison Selection
        selectInput("scatter_choice", "Select Scatter Plot Comparison:",
                    choices = c("Energy vs Danceability", "Valence vs Energy", "Tempo vs Speechiness")),
        hr(),
        # Display Precision@K
        textOutput("precision_k")
      ),
      
      mainPanel(
        # Recommendations Table
        tableOutput("recommend_table"),
        hr(),
        # Scatter Plot for feature comparison
        plotOutput("scatter_plot")
      )
    )
  )
}

# Define server function
define_server <- function(input, output, session) {
  
  # Reactive expression for filtered songs based on selected alphabet
  filtered_songs <- reactive({
    clustered_songs[substr(clustered_songs$name, 1, 1) == input$alphabet, ]
  })
  
  # Render song selector UI based on filtered songs
  output$song_selector <- renderUI({
    req(filtered_songs())  # Ensure that the filtered songs are available
    selectInput("song", "Select a Song:", choices = filtered_songs()$name)
  })
  
  # Function to get recommended songs using KNN
  recommend_songs <- function(song_name, top_n = 5) {
    song_info <- clustered_songs[clustered_songs$name == song_name, ]
    if (nrow(song_info) == 0) return(NULL)
    
    song_matrix <- as.matrix(clustered_songs[, features])
    song_vector <- song_matrix[clustered_songs$name == song_name, , drop = FALSE]
    
    # Nearest neighbors calculation
    nn_result <- get.knnx(song_matrix, song_vector, k = top_n + 1)
    recommended_indices <- nn_result$nn.index[-1]  # Exclude self
    
    # Get recommended songs
    recommended_songs <- clustered_songs[recommended_indices, c("name", "artists", "popularity")]
    return(recommended_songs)
  }
  
  # Render recommendations table
  output$recommend_table <- renderTable({
    req(input$song)
    recommend_songs(input$song)
  })
  
  # Render scatter plot for dynamic feature comparisons
  output$scatter_plot <- renderPlot({
    req(input$song, input$scatter_choice)
    
    recs <- recommend_songs(input$song)
    if (is.null(recs) || nrow(recs) == 0) return()
    
    # Feature mapping for scatter plot comparisons
    feature_map <- list(
      "Energy vs Danceability" = c("energy", "danceability"),
      "Valence vs Energy" = c("valence", "energy"),
      "Tempo vs Speechiness" = c("tempo", "speechiness")
    )
    
    selected_features <- feature_map[[input$scatter_choice]]
    x_feature <- selected_features[1]
    y_feature <- selected_features[2]
    
    # Extract selected features for both input song and recommendations
    song_features <- clustered_songs %>% 
      filter(name == input$song) %>% 
      select(name, all_of(x_feature), all_of(y_feature)) %>%
      mutate(category = "Selected Song")
    
    rec_features <- clustered_songs %>%
      filter(name %in% recs$name) %>%
      select(name, all_of(x_feature), all_of(y_feature)) %>%
      mutate(category = "Recommended")
    
    # Combine datasets
    plot_data <- bind_rows(song_features, rec_features)
    
    # Ensure data is present
    if (nrow(plot_data) < 2) return()  
    
    # Scatter Plot
    ggplot(plot_data, aes_string(x = x_feature, y = y_feature, color = "category", label = "name")) +
      geom_point(size = 3) +
      geom_text(vjust = -1, hjust = 0.5) +  # Add song names as labels
      theme_minimal() +
      labs(title = paste(x_feature, "vs", y_feature), x = x_feature, y = y_feature) +
      scale_color_manual(values = c("Selected Song" = "red", "Recommended" = "blue"))
  })
  
  # Function to evaluate Precision@K
  evaluate_precision_at_k <- function(recommendations, input_song_name, threshold = 0.1) {
    input_popularity <- clustered_songs[clustered_songs$name == input_song_name, "popularity"]
    recommended_popularities <- clustered_songs[clustered_songs$name %in% recommendations$name, "popularity"]
    lower_bound <- input_popularity * (1 - threshold)
    upper_bound <- input_popularity * (1 + threshold)
    
    # Count relevant recommendations
    relevant_recommendations <- sum(recommended_popularities >= lower_bound & recommended_popularities <= upper_bound)
    precision <- relevant_recommendations / length(recommended_popularities)
    return(precision)
  }
  
  # Render Precision@K score
  output$precision_k <- renderText({
    req(input$song)
    recs <- recommend_songs(input$song)
    if (is.null(recs)) return("N/A")
    
    precision_score <- evaluate_precision_at_k(recs, input$song)
    paste("Precision@K:", round(precision_score, 4))
  })
}

# Run the Shiny App
shinyApp(ui = define_ui(), server = define_server)
