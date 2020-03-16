\name{recSystem-package}
\alias{recSystem-package}
\alias{recSystem}
\docType{package}
\title{
  Meta Learning Recommendation System
}
\description{
  This package extracts meta features from a dataset to recommend what machine learning algorithm will perform the best
  without running all the implemented machine learning algorithms.
  The current selection of algorithms is limited to support vector machines, naiive bayes classifier and k nearest neighbors. The metric recall is used to give the recommended classifier. 
  This meta learnening recommendation system utilizes support vector regression with a radial basis function kernel to give  the recommended algorithm for implemnation.
  
  - Additionally, this package cleans the data in the following manner:
    + drop columns that have the exact same input for each row
    + drop rows that have NA's
    + drop object columns that have all unique values
    + one hot encoding for categorical variables
    + normalize continuous columns
    + label encode response

}
\details{
You can launch the recommend function from the console using the recommend function.

You can launch the shiny app using the run_my_app function
  
}
\author{
Clarence Williams, <clarencew3@gmail.com>

Maintainer: Clarence Williams, <clarencew3@gmail.com>
}
\references{
  This optional section can contain literature or other references for
  background information.
}
\keyword{ package }
\seealso{
  Optional links to other man pages
}
\examples{
  \dontrun{
     ## Optional simple examples of the most important functions
     ## run_my_app("recSystemApp")
     ## recSystem::recommend()
  }
}
