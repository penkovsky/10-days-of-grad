{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DataKinds #-}

import           Data.Massiv.Array hiding ( map, zip, unzip, zipWith, mapM_ )
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest.Vector as A
-- import qualified Data.Massiv.Array.IO as A
-- import           Graphics.ColorSpace
import           Streamly
import qualified Streamly.Prelude as S
import           Text.Printf ( printf )
import           Control.DeepSeq ( force )
import           Control.Monad.Trans.Maybe
import           Data.IDX
import qualified Data.Vector.Unboxed as V
import           Data.List.Split ( chunksOf )

import           NeuralNetwork
import           Shuffle ( shuffleIO )
-- import           Weights

-- TODO: implement algorithm from
-- Accelerating Deep Learning by Focusing on the Biggest Losers

loadMNIST
  :: FilePath -> FilePath -> IO (Maybe [(Volume4 Float, Matrix Float)])
loadMNIST fpI fpL = runMaybeT $ do
    i <- MaybeT $ decodeIDXFile fpI
    l <- MaybeT $ decodeIDXLabelsFile fpL
    d <- MaybeT. return $ force $ labeledIntData l i
    r <- return $ map _conv d
    return r
  where
    _conv :: (Int, V.Vector Int) -> (Volume4 Float, Matrix Float)
    _conv (label, v) = (v1, toOneHot10 label)
      where
        v0 = V.map ((/0.3081). (`subtract` 0.1307). (/ 255). fromIntegral) v

        v1 = A.fromVector' Par (Sz4 1 1 28 28) v0

toOneHot10 :: Int -> Matrix Float
toOneHot10 n = A.makeArrayR U Par (Sz2 1 10) (\(_ :. j) -> if j == n then 1 else 0)

mnistStream
  :: Int -> FilePath -> FilePath
  -> IO (SerialT IO (Volume4 Float, Matrix Float))
mnistStream batchSize fpI fpL = do
  Just dta <- loadMNIST fpI fpL
  dta2 <- shuffleIO dta

  -- Split data into batches
  let (vs, labs) = unzip dta2
      merge4 :: [Volume4 Float] -> Volume4 Float
      -- Dimensions:
      -- Sz (1 :> 1 :> 1 :. 1)
      --     ^    ^    ^    ^
      --     4    |    |    1
      --          3    2
      -- So the first dimension is actually
      -- fourth in this notation
      merge4 = setComp Par. A.compute. A.concat' 4

      merge2 :: [Matrix Float] -> Matrix Float
      merge2 = setComp Par. A.compute. A.concat' 2

      vs' = map merge4 $ chunksOf batchSize vs
      labs' = map merge2 $ chunksOf batchSize labs
      dta' = zip vs' labs'
  return $ S.fromList dta'

data TrainSettings = TrainSettings
  { _printEpochs :: Int  -- Print every N epochs
  , _lr :: Float  -- Learning rate
  , _totalEpochs :: Int  -- Number of training epochs
  }

train
  :: TrainSettings
  -> LeNet Float
  -> (SerialT IO (Volume4 Float, Matrix Float),
      SerialT IO (Volume4 Float, Matrix Float))
  -> IO (LeNet Float)
train TrainSettings { _printEpochs = printEpochs
                    , _lr = lr
                    , _totalEpochs = totalEpochs
                    } net (trainS, testS) = do
  (net', _) <- iterN (totalEpochs `div` printEpochs) (\(net0, j) -> do
    net1 <- sgd lr printEpochs net0 trainS

    tacc <- net1 `avgAccuracy` trainS :: IO Float
    putStr $ printf "%d Training accuracy %.1f" (j :: Int) tacc

    acc <- net1 `avgAccuracy` testS :: IO Float
    putStrLn $ printf "  Validation accuracy %.1f" acc

    return (net1, j + printEpochs)
    ) (net, 1)
  return net'

main :: IO ()
main = do
  trainS <- mnistStream 64 "data/train-images-idx3-ubyte" "data/train-labels-idx1-ubyte"
  testS <- mnistStream 1000 "data/t10k-images-idx3-ubyte" "data/t10k-labels-idx1-ubyte"

  net <- randNetwork

  net' <- train TrainSettings { _printEpochs = 1
                              , _lr = 0.01
                              , _totalEpochs = 10
                              } net (trainS, testS)

  return ()

{-
testGrad =
  let im' = resize' (Sz4 1 1 28 28) im
      im2 = (compute $ append' 4 im' im') :: Volume4 Float
      d0 = A.replicate Par (Sz2 1 3) 1.0 :: Matrix Float
      d1 = computeAs U $ expandWithin Dim1 28 (\e j -> (fromIntegral j + 0.5) + e) d0 :: Volume Float
      d = computeAs U $ expandWithin Dim2 28 (\e j -> (fromIntegral j + 0.25) * 3 + e) d1 :: Volume4 Float
      d2 = (compute $ append' 4 d d) :: Volume4 Float
      a' = conv2d' (Padding (Sz2 2 2) (Sz2 2 2) (Fill 0.0)) w0 d

      w' = conv2d'' (Padding (Sz2 2 2) (Sz2 2 2) (Fill 0.0)) im2 d2
  in (a', w', size d2, size im2)

writeImageY :: FilePath -> Matrix Float -> IO ()
writeImageY f a = do
  let b = compute $ A.map pure a :: A.Image U Y Float
  A.writeImageAuto f b

lenetFeatures :: Volume4 Float -> Volume4 Float
lenetFeatures = conv2d_ (Padding (Sz2 2 2) (Sz2 2 2) (Fill 0.0)) w0
              ~> relu_
              ~> maxpool_
              ~> conv2d_ noPad2 w1
              ~> relu_
              ~> maxpool_

testLeNet :: IO ()
testLeNet = do
  -- By convention, the first dimension is channels
  let im1channel = resize' (Sz (1 :> 28 :. 28)) im
      batch = setComp Par $ computeAs U $ expandWithin Dim4 32 const im1channel
      featureMaps2 = lenetFeatures batch

  print $ size batch
  print $ size featureMaps2
  -- Sz (32 :> 3 :> 5 :. 5)

  -- mapM_ (\(i, result) ->  writeImageY (show i ++ ".png") result) $ zipWith (,) [0..]  (splitChannels featureMaps2)

testConv1D :: IO ()
testConv1D = do
  let a = A.fromList Seq [1,2,5,6,2,-1,3,-2] :: Vector Int
  print a

  -- 1D identity kernel
  let s = makeCorrelationStencilFromKernel (A.fromList Seq [0, 1, 0] :: Vector Int)
  let delayedS = makeCorrelationStencilFromKernel (A.fromList Seq [1, 0, 0] :: Vector Int)
  print $ mapStencil (Fill 0) s a
  print (compute $ mapStencil (Fill 0) delayedS a :: Vector Int)
  let stride = Stride 3
  print (computeWithStride stride $ mapStencil Edge delayedS a :: Vector Int)

main :: IO ()
main = putStrLn "Test 1D convolutions" >> testConv1D
       >> putStrLn "Test 2D convolutions" >> testLeNet
-}

-- zeroPadding (m', n') = compute. applyStencil (Padding (Sz2 0 0) (Sz2 m' n') (Fill 0)) idStencil
--
-- test2Dto1D = do
--   -- 1D convolution equivalent to 2D:
--   -- (1) Append zeros to make both kernel and image square matrices.
--   -- (2) Flatten both.
--   -- (3) Reject zero tails.
--   -- See Multidimensional convolution via a 1D convolution algorithm
--   -- by Naghizadeh and Sacchi.
--
--   -- Here we assume that images are in a smaller 20x20 box, so there is no need for
--   -- additional padding.
--   -- First, prepare the 1D kernel
--   let k_ = w (0, 0) w0
--       -- Zero padding the kernel to a 28x28 matrix:
--       k = zeroPadding (23, 23) k_ :: Matrix Float
--       -- Reshape to a 1D array
--       k' = A.resize' 784 k
--       -- Crop to a new 1D kernel of 28*4 + 5 = 117 first values
--       k2 = compute $ extractFromTo' 0 117 k' :: Vector Float
--
--   -- Second, reshape the image
--   let im' = compute $ A.resize' 784 im :: Vector Float
--
--   -- Third, run the 1D convolution and reshape back
--   let c = compute $ mapStencil Edge (makeCorrelationStencilFromKernel k2) im' :: Vector Float
--       im2 = A.resize' (Sz (28 :. 28)) c
--   writeImageY "0_a.png" im2
--
--   return ()

-- > a = computeAs U $ resize' (Sz2 5 5) (Ix1 11 ... 35)
-- > d = makeArrayR D Seq (Sz2 10 10) (const 0)
-- > insertWindow d (Window (2 :. 3) (size a) ((a !) . liftIndex2 subtract (2 :. 3)) Nothing)
-- Array DW Seq (Sz (10 :. 10))
--   [ [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
--   , [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
--   , [ 0, 0, 0, 11, 12, 13, 14, 15, 0, 0 ]
--   , [ 0, 0, 0, 16, 17, 18, 19, 20, 0, 0 ]
--   , [ 0, 0, 0, 21, 22, 23, 24, 25, 0, 0 ]
--   , [ 0, 0, 0, 26, 27, 28, 29, 30, 0, 0 ]
--   , [ 0, 0, 0, 31, 32, 33, 34, 35, 0, 0 ]
--   , [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
--   , [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
--   , [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
--   ]
