
skip_if_no_fastai_core <- function() {
  if (!reticulate::py_module_available("fastai"))
    skip("fastai not available for testing")
}

skip_if_32bit <- function() {
  if (Sys.info()[["machine"]]=="x86_32")
    skip("skip 32 bit system!")
}

test_succeeds <- function(desc, expr) {
  test_that(desc, {
    skip_if_no_fastai_core()
    skip_if_32bit()
    expect_error(force(expr), NA)
  })
}

os_win_skip <- function() {
  if (os()=='windows')
    skip("skip windows")
}

test_succeeds_windows <- function(desc, expr) {
  test_that(desc, {
    skip_if_no_fastai_core()
    skip_if_32bit()
    os_win_skip()
    expect_error(force(expr), NA)
  })
}






