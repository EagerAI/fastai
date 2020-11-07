
context("modify_tensor")

source("utils.R")

test_succeeds('modify scalar tensor', {
  x3 = tensor(list(list(c(1, 2, 3), c(4, 5, 6), c(7, 8, 9)),list(c(10, 11, 12), c(13, 14, 15), c(16, 17, 18))))
  init = tensor(list(list(c(1, 2, 3), c(4, 5, 6), c(7, 8, 9)),list(c(10, 11, 12), c(13, 14, 15), c(16, 17, 18))))
  value = as.numeric(12)
  x3[0][0][1] %f% value
  expect_equal(init[0][0][1]$cpu()$numpy() + 10, x3[0][0][1]$cpu()$numpy())
  expect_equal(init[0][0][1] + 10, x3[0][0][1])
})


test_succeeds('modify 2 values tensor', {
  x3 = tensor(list(list(c(1, 2, 3), c(4, 5, 6), c(7, 8, 9)),list(c(10, 11, 12), c(13, 14, 15), c(16, 17, 18))))
  init = tensor(list(list(c(1, 2, 3), c(4, 5, 6), c(7, 8, 9)),list(c(10, 11, 12), c(13, 14, 15), c(16, 17, 18))))
  x3[0][0] %f% c(3,9,6)
  expect_equal(init[0][0]$cpu()$numpy() + c(2,7,3), x3[0][0]$cpu()$numpy())
  expect_equal(tensor(c(3,9,6)), x3[0][0])
})






