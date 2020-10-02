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
  expect_equal(round(max(res), 3),
               round(max(res_tensor)$cpu()$numpy()[1], 3))
})

test_succeeds('check min', {
  res = runif(10)
  res_tensor = tensor(res)
  expect_equal(round(min(res), 3),
               round(min(res_tensor)$cpu()$numpy()[1], 3))
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


test_succeeds('check sort ascending', {
  r_vector = c(4,2,8,5)
  py_tensor = tensor(r_vector)
  expect_equal(sort(r_vector), as.vector(sort(py_tensor)$values$cpu()$numpy()))
})

test_succeeds('check sort descending', {
  r_vector = c(4,2,8,5)
  py_tensor = tensor(r_vector)
  expect_equal(sort(r_vector, decreasing = TRUE), as.vector(sort(py_tensor, decreasing = TRUE)$values$cpu()$numpy()))
})

test_succeeds('check abs', {
  r_vector = c(-10)
  py_tensor = tensor(r_vector)
  expect_equal(abs(r_vector), as.vector(abs(py_tensor)$cpu()$numpy()))
})


test_succeeds('check + add', {
  r_vector = c(4,2,8,5)
  py_tensor = tensor(r_vector)
  expect_equal(r_vector+1, as.vector((py_tensor+1)$cpu()$numpy()))
})


test_succeeds('check - substr', {
  r_vector = c(4,2,8,5)
  py_tensor = tensor(r_vector)
  expect_equal(r_vector-1, as.vector((py_tensor-1)$cpu()$numpy()))
})

test_succeeds('check / divide', {
  r_vector = c(4,2,8,5)
  py_tensor = tensor(r_vector)
  expect_equal(r_vector/2, as.vector((py_tensor/2)$cpu()$numpy()))
})


test_succeeds('check * mul', {
  r_vector = c(4,2,8,5)
  py_tensor = tensor(r_vector)
  expect_equal(r_vector*2, as.vector((py_tensor*2)$cpu()$numpy()))
})


test_succeeds('check exp', {
  r_vector = c(4,2,8,5)
  py_tensor = tensor(r_vector)
  expect_equal(exp(r_vector), as.vector((exp(py_tensor))$cpu()$numpy()))
})


test_succeeds('check expm1', {
  r_vector = c(4,2,8,5)
  py_tensor = tensor(r_vector)
  expect_equal(expm1(r_vector), as.vector((expm1(py_tensor))$cpu()$numpy()))
})

test_succeeds('check log', {
  r_vector = c(4,2,8,5)
  py_tensor = tensor(r_vector)
  expect_equal(log(r_vector), as.vector((log(py_tensor))$cpu()$numpy()))
})

test_succeeds('check log1p', {
  r_vector = c(4,2,8,5)
  py_tensor = tensor(r_vector)
  expect_equal(round(log1p(r_vector),4),
               round(as.vector((log1p(py_tensor))$cpu()$numpy()), 4))
})

test_succeeds('check round', {
  r_vector = runif(20,5,10)
  py_tensor = tensor(r_vector)
  expect_equal(round(r_vector),
               as.vector(round(py_tensor)$cpu()$numpy()))
})

test_succeeds('check sqrt', {
  r_vector = c(4,16)
  py_tensor = tensor(r_vector)
  expect_equal(sqrt(r_vector),
               as.vector(sqrt(py_tensor)$cpu()$numpy()))
})



test_succeeds('check floor', {
  r_vector = runif(20,5,10)
  py_tensor = tensor(r_vector)
  expect_equal(floor(r_vector),
               as.vector(floor(py_tensor)$cpu()$numpy()))
})


test_succeeds('check ceiling', {
  r_vector = runif(20,5,10)
  py_tensor = tensor(r_vector)
  expect_equal(ceiling(r_vector),
               as.vector(ceiling(py_tensor)$cpu()$numpy()))
})

test_succeeds('check cos', {
  r_vector = runif(20,5,10)
  py_tensor = tensor(r_vector)
  expect_vector(round( as.vector(cos(py_tensor)$cpu()$numpy()), 4))
})

test_succeeds('check cosh', {
  r_vector = runif(20,5,10)
  py_tensor = tensor(r_vector)
  expect_vector(round(as.vector(cosh(py_tensor)$cpu()$numpy()), 0))
})

test_succeeds('check sin', {
  r_vector = runif(20,5,10)
  py_tensor = tensor(r_vector)
  expect_equal(round(sin(r_vector),0),
               round(as.vector(sin(py_tensor)$cpu()$numpy()), 0))
})

test_succeeds('check sinh', {
  r_vector = runif(20,5,10)
  py_tensor = tensor(r_vector)
  expect_equal(round(sinh(r_vector),0),
               round(as.vector(sinh(py_tensor)$cpu()$numpy()), 0))
})

test_succeeds('check mean', {
  r_vector = runif(20,5,10)
  py_tensor = tensor(r_vector)
  expect_equal(round(mean(r_vector),3),
               round(as.vector(mean(py_tensor)$cpu()$numpy()),3))
})


















