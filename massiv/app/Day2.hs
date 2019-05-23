{-# LANGUAGE FlexibleContexts #-}

-- | = Day 2 reimplementation in Massiv

import           Data.Massiv.Array as A

import           Text.Printf ( printf )

import           NeuralNetwork

-- | Circles dataset
makeCircles
  :: Int -> Float -> Float -> IO (Matrix Float, Matrix Float)
makeCircles m factor noiseLevel = do
  let rand' n = rand (0, 2 * pi) (Sz (n :. 1))
      m1 = m `div` 2
      m2 = m - (m `div` 2)

  r1 <- rand' m1 :: IO (Matrix Float)
  r2 <- rand' m2 :: IO (Matrix Float)
  ns <- rand (-noiseLevel / 2, noiseLevel / 2) (Sz (m :. 2)) :: IO (Matrix Float)

  let outerX = compute $ cosA r1 :: Matrix Float
      outerY = compute $ sinA r1 :: Matrix Float
      innerX = scale factor $ compute $ cosA r2 :: Matrix Float
      innerY = scale factor $ compute $ sinA r2 :: Matrix Float

  -- Merge them all
  let x1 = compute $ append' 1 outerX outerY :: Matrix Float
      x2 = compute $ append' 1 innerX innerY :: Matrix Float
      x = compute $ append' 2 x1 x2 :: Matrix Float
      y1 = (A.replicate Par (Sz2 m1 1) 0) :: Matrix Float
      y2 = (A.replicate Par (Sz2 m2 1) 1) :: Matrix Float
      y = append' 2 y1 y2

  return (compute $ x .+ ns, compute y)


main :: IO ()
main = do
  trainSet <- makeCircles 200 0.6 0.1
  testSet <- makeCircles 100 0.6 0.1

  net <- genNetwork [2, 128, 1] [Relu, Id]

  -- How about correct-by-construction like
  -- net <- genNetwork (I 2 :> FC 128 Relu :> O 1)

  let epochs = 1000
      lr = 0.001  -- Learning rate
      net' = optimize lr epochs net trainSet
      netA = adam adamParams epochs net trainSet

  putStrLn $ printf "Circles problem, 1 hidden layer of 128 neurons, %d epochs" epochs
  putStrLn "---"

  putStrLn $ printf "Training accuracy (gradient descent) %.1f" (net' `accuracy` trainSet)
  putStrLn $ printf "Validation accuracy (gradient descent) %.1f\n" (net' `accuracy` testSet)

  putStrLn $ printf "Training accuracy (Adam) %.1f" (netA `accuracy` trainSet)
  putStrLn $ printf "Validation accuracy (Adam) %.1f\n" (netA `accuracy` testSet)

-- Outputs for the same initial weights:
--
-- Circles problem, 1 hidden layer of 128 neurons, 1000 epochs
-- ---
-- Training accuracy (gradient descent) 72.5
-- Validation accuracy (gradient descent) 73.0
--
-- Training accuracy (Adam) 100.0
-- Validation accuracy (Adam) 100.0
