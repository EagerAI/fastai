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


test_succeeds('check >', {
  res = (scalar > scalar)$cpu()$numpy()[1]
  expect_false(res)
})

test_succeeds('check <', {
  res = (scalar < scalar)$cpu()$numpy()[1]
  expect_false(res)
})

test_succeeds('check >=', {
  res = (scalar >= scalar)$cpu()$numpy()[1]
  expect_true(res)
})

test_succeeds('check <=', {
  res = (scalar <= scalar)$cpu()$numpy()[1]
  expect_true(res)
})


test_succeeds('check max', {
  res = runif(10)
  res_tensor = tensor(res)
  expect_equal(max(res),max(res_tensor)$cpu()$numpy()[1])
})

test_succeeds('check min', {
  res = runif(10)
  res_tensor = tensor(res)
  expect_equal(min(res),min(res_tensor)$cpu()$numpy()[1])
})

test_succeeds('check dim', {
  res = matrix(runif(100),nrow = 5, ncol = 5)
  res_tensor = tensor(res)
  expect_equal(dim(res),dim(res_tensor))
})

# Filter(isGeneric,ls(all.names=TRUE, env = baseenv()))

test_succeeds('check length', {
  expect_equal(length(res), length(res_tensor))
})

test_succeeds('check floor_div %/%', {
  r_vector = runif(10)
  py_tensor = tensor(r_vector)
  expect_equal(r_vector %/% 5, as.vector((py_tensor %/% 5)$cpu()$numpy()))
})


test_succeeds('check floor_div %/%', {
  r_vector = runif(10,5,10)
  py_tensor = tensor(r_vector)
  expect_equal(round(r_vector %% 5, 3),
               round(as.vector((py_tensor %% 5)$cpu()$numpy()), 3))
})


test_succeeds('check &', {
  r_vector = c(5)
  py_tensor = tensor(r_vector)
  expect_equal(r_vector & r_vector, as.vector((py_tensor & py_tensor)$cpu()$numpy()))
})


test_succeeds('check |', {
  r_vector = c(5)
  py_tensor = tensor(r_vector)
  expect_equal(r_vector | r_vector, as.vector((py_tensor | py_tensor)$cpu()$numpy()))
})

test_succeeds('check not !', {
  r_vector = c(5)
  py_tensor = tensor(r_vector)
  expect_equal(!r_vector, as.vector((!py_tensor)$cpu()$numpy()))
})





