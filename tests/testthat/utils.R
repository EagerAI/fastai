
skip_if_no_fastai_core <- function() {
  if (!reticulate::py_module_available("fastai"))
    skip("fastai not available for testing")
}


test_succeeds <- function(desc, expr) {
  test_that(desc, {
    skip_if_no_fastai_core()
    expect_error(force(expr), NA)
  })
}







