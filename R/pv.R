#' @title  Computes present values
#'
#' @description Computes present values given a future value, a rate and a number of years
#' @export
#' @param  FV A \code{numeric} vector of future value of the thing 
#' @param  r A \code{numeric} vector (of length 1) for the rate
#' @param  n The number of years that the rate \code{r} is applied to \code{FV}
#' @examples 
#' \dontrun{
#' # Test for pv
#' pv(FV = 1000, r = .08, n = 5)
#' 
#' # Throws an error
#' fv_l <- list(fv1 = 800, 
#'              fv2 = 900, 
#'                 fv3 = 1100)
#'
#' pv(fv_l, 0.08)
#'
#' 
#' 
#' }
pv <- function(FV, r, n = 5) {
  if(!is.atomic(FV)) {
    stop('FV must be an atomic vector')
  }
  
  if(!is.numeric(FV) | !is.numeric(r) | !is.numeric(n)){
    stop('This function only works for numeric inputs!\n', 
         'You have provided objects of the following classes:\n', 
         'FV: ', class(FV), '\n',
         'r: ', class(r), '\n',
         'n: ', class(n))
  }
  
  if(r < 0 | r > .25) {
    message('The input for r exceeds the normal\n',
            'range for interest rates (0-25%)')
  }
  
  present_value <- FV / (1 + r)^n
  round(present_value, 2)
}
