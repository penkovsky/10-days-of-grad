-- | Circles dataset and gradient descent in a multilayer neural network
--
-- 1. Install stack (command line interface is marked by $):
--   $ wget -qO- https://get.haskellstack.org/ | sh
-- (alternatively, curl -sSL https://get.haskellstack.org/ | sh)
--
-- 2. Install open-blas from https://www.openblas.net/
-- (needed for hmatrix package)
--
-- 3. Compile and run
--
--   $ stack --resolver lts-10.6 --install-ghc ghc --package hmatrix-0.18.2.0 --package hmatrix-morpheus-0.1.1.2 -- -O2 Main.hs
--   $ ./Main

import           Numeric.LinearAlgebra
import           Text.Printf ( printf )

import           NeuralNetwork


-- | Circles dataset
makeCircles
  :: Int -> Double -> Double -> IO (Matrix Double, Matrix Double)
makeCircles m factor noise = do
  let rand' n = (scale (2 * pi)) <$> rand n 1
      m1 = m `div` 2
      m2 = m - (m `div` 2)

  r1 <- rand' m1
  r2 <- rand' m2
  ns <- scale noise <$> randn m 2

  let outerX = cos r1
      outerY = sin r1
      innerX = scale factor $ cos r2
      innerY = scale factor $ sin r2
      -- Merge them all
      x = fromBlocks [[outerX, outerY], [innerX, innerY]]

      -- Labels
      y1 = m1 >< 1 $ repeat 0
      y2 = m2 >< 1 $ repeat 1
      y = y1 === y2

  return (x + ns, y)

-- | Spirals dataset.
-- Note, produces twice more points than m.
makeSpirals
  :: Int -> Double -> IO (Matrix Double, Matrix Double)
makeSpirals m noise = do
  r0 <- (scale (780 * 2*pi / 360). sqrt) <$> rand m 1
  d1x0 <- scale noise <$> rand m 1
  d1y0 <- scale noise <$> rand m 1

  let d1x = d1x0 - cos(r0) * r0
  let d1y = d1y0 + sin(r0) * r0

  let x = (fromBlocks [[d1x, d1y], [-d1x, -d1y]]) / 10.0
  let y1 = m >< 1 $ repeat 0
  let y2 = m >< 1 $ repeat 1
  let y = y1 === y2
  return (x, y)

experiment1 = do
  trainSet <- makeCircles 200 0.6 0.1
  testSet <- makeCircles 100 0.6 0.1

  let (dta, tgt) = trainSet

  (w1_rand, b1_rand) <- genWeights (2, 128)
  (w2_rand, b2_rand) <- genWeights (128, 1)

  let net = [ Layer w1_rand b1_rand Relu
            , Layer w2_rand b2_rand Id ]

  -- -- Alternatively:
  -- net0 <- genNetwork [2, 128, 1] [Relu, Id]

  let epochs = 1000
      lr = 0.001  -- Learning rate
      net' = optimize lr epochs net trainSet
      netA = optimizeAdam adamParams epochs net trainSet

  putStrLn $ printf "Circles problem, 1 hidden layer of 128 neurons, %d epochs" epochs
  putStrLn "---"

  putStrLn $ printf "Training accuracy (gradient descent) %.1f" (net' `accuracy` trainSet)
  putStrLn $ printf "Validation accuracy (gradient descent) %.1f\n" (net' `accuracy` testSet)

  putStrLn $ printf "Training accuracy (Adam) %.1f" (netA `accuracy` trainSet)
  putStrLn $ printf "Validation accuracy (Adam) %.1f\n" (netA `accuracy` testSet)

  putStrLn ""

experiment2 = do
  trainSet <- makeSpirals 200 0.5
  testSet <- makeSpirals 100 0.5
  -- saveMatrix "/tmp/spir.x" "%g" dta
  -- saveMatrix "/tmp/spir.y" "%g" tgt

  let (dta, tgt) = trainSet
      epochs = 700

  putStrLn $ printf "Spirals problem, Adam, %d epochs" epochs
  putStrLn "---"
  putStrLn "1 hidden layer, 128 neurons (513 parameters)"
  net0 <- genNetwork [2, 128, 1] [Relu, Id]
  let net0' = optimizeAdam adamParams epochs net0 trainSet

  putStrLn $ printf "Training accuracy %.1f" (net0' `accuracy` trainSet)
  putStrLn $ printf "Validation accuracy %.1f\n" (net0' `accuracy` testSet)

  putStrLn "1 hidden layer, 512 neurons (2049 parameters)"
  net1 <- genNetwork [2, 512, 1] [Relu, Id]
  let net1' = optimizeAdam adamParams epochs net1 trainSet

  putStrLn $ printf "Training accuracy %.1f" (net1' `accuracy` trainSet)
  putStrLn $ printf "Validation accuracy %.1f\n" (net1' `accuracy` testSet)

  putStrLn "3 hidden layers, 40, 25, and 10 neurons (1416 parameters)"
  net2 <- genNetwork [2, 40, 25, 10, 1] [Relu, Relu, Relu, Id]
  let net2' = optimizeAdam adamParams epochs net2 trainSet

  putStrLn $ printf "Training accuracy %.1f" (net2' `accuracy` trainSet)
  putStrLn $ printf "Validation accuracy %.1f\n" (net2' `accuracy` testSet)

main = experiment1 >> experiment2
