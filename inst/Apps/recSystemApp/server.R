server <- function(input, output, session) {
  datasetInput <- eventReactive(input$FileInput, {
    infile <- input$FileInput
    df <- read.csv(infile$datapath, header = TRUE)
    updateSelectInput(session,"select",choices=colnames(df))
    return(df)
  })
  
  dataCleanInput <- eventReactive(input$goButton, {
    recSystem:::recommend_shiny(datasetInput(), input$select)
  })
  
  dataRecAlg <- eventReactive(input$goButton, {
    recSystem:::recommend_alg(datasetInput(), input$select)
  })
  
  observeEvent(input$goButton, {
    updateNavbarPage(session, "inTabset",selected = "Data")
  })
  
  output$downloadData <- downloadHandler(
    filename = function() {
      paste("data-", Sys.Date(), ".csv", sep="")
    },
    content = function(file) {
      write.csv(dataCleanInput(), file)
    }
  )
  output$selected_var <- renderText({ 
    paste("Recommend algorithm:", dataRecAlg())
  })
  
  output$table = DT::renderDataTable(datasetInput())
  output$contents = DT::renderDataTable(dataCleanInput())
}