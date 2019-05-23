{-# LANGUAGE FlexibleContexts #-}

-- | = Day 1 reimplementation in Massiv

import           Data.Massiv.Array as A

import           Data.Massiv.Array.Numeric ( (|*|) )
-- Benchmark
import           Data.Time

import           NeuralNetwork


linear' :: Matrix Float
        -> Matrix Float
        -> Matrix Float
linear' x dy =
  let trX = compute $ transpose x
      prod = trX |*| dy
      k = recip $ fromIntegral (rows x)
  in k `scale` prod

loss :: Matrix Float
     -> Matrix Float
     -> Float
loss y tgt =
  let diff = compute $ y .- tgt :: Matrix Float
      ar = compute $ diff .^ 2 :: Matrix Float
  in A.sum ar

loss' :: Matrix Float
      -> Matrix Float
      -> Matrix Float
loss' y tgt =
  let diff = compute $ y .- tgt
  in 2.0 `scale` diff

grad :: (Matrix Float, Matrix Float)
     -> Matrix Float
     -> Matrix Float
grad (x, y) w1 =
  let [h, y_pred] = forward x w1
      dE = loss' y_pred y
      dY = sigmoid' h dE
  in linear' x dY

forward :: Matrix Float
        -> Matrix Float
        -> [Matrix Float]
forward x w1 =
  let h = x |*| w1
      y = sigmoid h
  in [h, y]

-- | Gradient descent
descend
  :: Float
  -> Int
  -> Matrix Float
  -> (Matrix Float, Matrix Float)
  -> Matrix Float
descend lr n w0 dta = iterN n step w0
  where
    step w =
      let dW1 = grad dta w
      in compute $ w .- (lr `scale` dW1)

-- | Benchmarking matrix multiplication.
-- Run program with `+RTS -N` flags
bench :: IO ()
bench = do
  a <- rand (-0.5, 0.5) (Sz (2000 :. 2000)) :: IO (Matrix Float)
  b <- rand (-0.5, 0.5) (Sz (2000 :. 2000)) :: IO (Matrix Float)
  -- let a' = setComp Par a
  -- let b' = setComp Par b

  start <- getCurrentTime
  let !c = a |*| b
  end <- getCurrentTime

  let strat = getComp c
  print strat

  print (diffUTCTime end start)

main :: IO ()
main = do
  dta <- "data/iris_x.dat" `readToMatrix` (150, 4)
  tgt <- "data/iris_y.dat" `readToMatrix` (150, 3)

  let (nin, nout) = (4, 3)

  (w0, _) <- genWeights (nin, nout)

  let epochs = 20000
      w1 = descend 0.01 epochs w0 (dta, tgt)

  let [_, y_pred0] = forward dta w0
      [_, y_pred] = forward dta w1

  putStrLn $ "Initial loss " ++ show (loss y_pred0 tgt)
  putStrLn $ "Loss after training " ++ show (loss y_pred tgt)

readToMatrix :: FilePath -> (Int, Int) -> IO (Matrix Float)
readToMatrix path (rows', cols') = do
  x <- (Prelude.map Prelude.read. words) <$> readFile path :: IO [Float]
  return $ A.resize' (Sz (rows' :. cols')) (A.fromList Par x)
