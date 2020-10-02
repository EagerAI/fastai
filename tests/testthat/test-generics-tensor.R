context("generic_tensor")

source("utils.R")

test_succeeds('create tensor', {
  r_tensor = tensor(runif(10))
  expect_true(inherits(r_tensor,"torch.Tensor"))
})

test_succeeds('check length', {
  expect_length(r_tensor, 10)
})


test_succeeds('check ==', {
  res = r_tensor == r_tensor
  expect_true(all(res$cpu()$numpy() == TRUE))
})

test_succeeds('check !=', {
  res = r_tensor != r_tensor
  expect_true(all(res$cpu()$numpy() == FALSE))
})

test_succeeds('check pow', {
  r_scalar = 5.23
  scalar = tensor(r_scalar)
  expect_equal(round(as.vector((scalar^scalar)$cpu()$numpy())),
               round(r_scalar ^ r_scalar) )
})

test_succeeds('check !=', {
  r_scalar = 5.23
  scalar = tensor(r_scalar)
  expect_equal(round(as.vector((scalar^scalar)$cpu()$numpy())),
               round(r_scalar ^ r_scalar) )
})








