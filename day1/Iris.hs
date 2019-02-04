-- 1. Install stack (command line interface is marked by $):
--   $ wget -qO- https://get.haskellstack.org/ | sh
-- (alternatively, curl -sSL https://get.haskellstack.org/ | sh)
-- 2. Install open-blas from https://www.openblas.net/ (needed for hmatrix package)
-- 3. Run
--   $ stack --resolver lts-10.6 --install-ghc runghc --package hmatrix-0.18.2.0 Iris.hs

import Numeric.LinearAlgebra as LA

-- New weights
newW (nin, nout) = do
    let k = sqrt (1.0 / fromIntegral nin)
    w <- randn nin nout
    return (cmap (k *) w)

-- Transformations
loss y tgt =
  let diff = y - tgt
  in sumElements $ cmap (^2) diff

sigmoid = cmap f
  where
    f x = recip $ 1.0 + exp (-x)

-- Their gradients
sigmoid' x dY = dY * y * (ones - y)
   where
    y = sigmoid x
    ones = (rows y) >< (cols y) $ repeat 1.0

linear' x dy = cmap (/ m) (tr' x LA.<> dy)
  where
    m = fromIntegral $ rows x

loss' y tgt =
  let diff = y - tgt
  in cmap (* 2) diff

-- Building NN
forward x w1 =
  let h = x LA.<> w1
      y = sigmoid h
  in [h, y]

descend gradF iterN gamma x0 = take iterN (iterate step x0)
  where
    step x = x - gamma * gradF(x)

grad (x, y) w1 = dW1
  where
    [h, y_pred] = forward x w1
    dE = loss' y_pred y
    dY = sigmoid' h dE
    dW1 = linear' x dY

main = do
  dta <- loadMatrix "iris_x.dat"
  tgt <- loadMatrix "iris_y.dat"

  let (nin, nout) = (4, 3)

  w1_rand <- newW (nin, nout)

  let epochs = 500
  let w1 = last $ descend (grad (dta, tgt)) epochs 0.01 w1_rand

      [_, y_pred0] = forward dta w1_rand
      [_, y_pred] = forward dta w1

  putStrLn $ "Initial loss " ++ show (loss y_pred0 tgt)
  putStrLn $ "Loss after training " ++ show (loss y_pred tgt)

  putStrLn "Some predictions by an untrained network:"
  print $ takeRows 5 y_pred0

  putStrLn "Some predictions by a trained network:"
  print $ takeRows 5 y_pred

  putStrLn "Targets"
  print $ takeRows 5 tgt
