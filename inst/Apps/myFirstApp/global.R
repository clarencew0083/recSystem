library(shiny)
library(shinythemes)      # Bootswatch color themes for shiny
library(choroplethr)      # Creating Choropleth Maps in R
library(choroplethrMaps)  # Maps used by the choroplethr package

# load the data set from the choroplethrMaps package
data('df_state_demographics')
map_data <- df_state_demographics