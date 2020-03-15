ui <-  navbarPage('RecSystem', id = "inTabset",
                  theme = shinythemes::shinytheme('cerulean'),
                  tabPanel(title = 'Input Data',
  fluidPage(
  sidebarLayout(
    sidebarPanel(
      fileInput("FileInput", "Choose CSV File",
                accept = c(
                  "text/csv",
                  "text/comma-separated-values,text/plain",
                  ".csv")),
      selectInput('select',
                  label = 'Select a Column',
                  choices = c()),
      #tags$hr(),
      #br(),
      div(style="display:inline-block;width:100%;text-align: center;",actionButton("goButton", "Go!"))),
      #actionButton("goButton", "Go!")),
      #checkboxInput("header", "Header", TRUE)),
      mainPanel(width = 12, DT::dataTableOutput("table"))))),
  tabPanel('Data',
             mainPanel(div(style = "font-weight: bold;",textOutput("selected_var")), 
                       actionButton("downloadData", "Download"),
                       tags$hr(),
                       br(),
                       DT::dataTableOutput("contents"))))

