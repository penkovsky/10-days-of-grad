-- {-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
-- {-# LANGUAGE TypeSynonymInstances #-}
-- {-# LANGUAGE FlexibleInstances #-}
-- {-# LANGUAGE TypeOperators #-}

module NeuralNetwork
  ( NeuralNetwork (..)
  , Layer (..)
  , Matrix
  , Vector
  , Activation (..)
  , sigmoid
  , sigmoid'
  , genWeights
  , genNetwork
  , forward'

  -- * Training
  , optimize
  , AdamParameters (..)
  , adamParams
  , adam

  -- * Inference
  , inferBinary
  , accuracy

  -- * Helpers
  , rows
  , cols
  , computeMap
  , rand
  , randn
  , scale
  , iterN
  ) where

import           Control.Monad ( replicateM )
import           Data.List ( foldl' )
import qualified System.Random as R
import           System.Random.MWC ( createSystemRandom )
import           System.Random.MWC.Distributions ( standard )

import           Data.Massiv.Array hiding ( map, zip, zipWith )
import qualified Data.Massiv.Array as A

type MatrixPrim r a = Array r Ix2 a
type Matrix a = Array U Ix2 a
type Vector a = Array U Ix1 a

-- Activation function:
-- * Rectified linear unit (ReLU)
-- * Sigmoid
-- * Identity (no activation)
data Activation = Relu | Sigmoid | Id

-- Neural network layer: weights, biases, and activation
data Layer a = Layer (Matrix a) (Vector a) Activation

type NeuralNetwork a = [Layer a]

-- | Weight and bias gradients
data Gradients a = Gradients (Matrix a) (Vector a)

-- | Lookup activation function by a symbol
getActivation :: Activation -> (Matrix Float -> Matrix Float)
getActivation Id = id
getActivation Sigmoid = sigmoid
getActivation Relu = relu

-- | Lookup activation function derivative by a symbol
getActivation'
  :: Activation
  -> (Matrix Float -> Matrix Float -> Matrix Float)
getActivation' Id = flip const
getActivation' Sigmoid = sigmoid'
getActivation' Relu = relu'

-- | Elementwise sigmoid computation
sigmoid :: Matrix Float -> Matrix Float
sigmoid = computeMap f
  where
    f x = recip $ 1.0 + exp (-x)

-- sigmoid
--   :: (Source r ix e, Floating e)
--   => Array r ix e -> Array D ix e
-- sigmoid = liftArray _sigmoid
--   where
--     _sigmoid x = recip $ 1.0 + exp (-x)
-- {-# INLINE sigmoid #-}

-- | Compute sigmoid gradients
sigmoid' :: Matrix Float
         -> Matrix Float
         -> Matrix Float
sigmoid' x dY =
  let sz = size x
      ones = A.replicate Par sz 1.0 :: Matrix Float
      y = sigmoid x
  in compute $ dY .* y .* (ones .- y)

relu :: Matrix Float -> Matrix Float
relu = computeMap f
  where
    f x = if x < 0
             then 0
             else x

relu' :: Matrix Float
      -> Matrix Float
      -> Matrix Float
relu' x = compute. A.zipWith f x
  where
    f x0 dy0 = if x0 <= 0
                  then 0
                  else dy0

randomishArray
  :: (Mutable r ix e, R.RandomGen a, R.Random e) =>
     (e, e) -> a -> Sz ix -> Array r ix e
randomishArray range g sz = compute $ unfoldlS_ Seq sz rand g
  where
    rand g =
      let (a, g') = R.randomR range g
      in (g', a)

-- | Uniformly-distributed random numbers Array
rand
  :: (R.Random e, Mutable r ix e) =>
     (e, e) -> Sz ix -> IO (Array r ix e)
rand range sz = do
  g <- R.newStdGen
  return $ randomishArray range g sz

-- | Random values from the Normal distribution
randn
  :: (Fractional e, Index ix, Resize r Ix1, Mutable r Ix1 e)
  => Sz ix -> IO (Array r ix e)
randn sz = do
    g <- createSystemRandom
    xs <- _nv g (totalElem sz)
    return $ resize' sz (fromList Seq xs)
  where
    _nv gen n = replicateM n (realToFrac <$> standard gen)
    {-# INLINE _nv #-}

rows :: Matrix Float -> Int
rows m =
  let (r :. _) = unSz $ size m
  in r

cols :: Matrix Float -> Int
cols m =
  let (_ :. c) = unSz $ size m
  in c

-- Returns a delayed Array. Useful for fusion
_scale c = A.map (* c)

scale :: Index sz => Float -> Array U sz Float -> Array U sz Float
scale konst = computeMap (* konst)

computeMap f = A.compute. A.map f

linearW' :: Matrix Float
        -> Matrix Float
        -> Matrix Float
linearW' x dy =
  let trX = compute $ transpose x
      prod = trX |*| dy
      k = recip $ fromIntegral (rows x)
  in k `scale` prod

linearX' :: Matrix Float
        -> Matrix Float
        -> Matrix Float
linearX' w dy = compute $ dy `multiplyTransposed` w

-- | Bias gradient
bias' :: Matrix Float -> Vector Float
bias' dY = k `scale` r
  where
    rCols = cols dY
    -- Sum elements in each column and return a new matrix
    -- r = matrix rCols $ map sumElements (toColumns dY)
    r = A.fromList Par $ map (A.sum. (dY <!)) [0..rCols - 1]
    k = recip $ fromIntegral $ rows dY

-- | Forward pass in a neural network:
-- exploit Haskell lazyness to never compute the
-- gradients.
forward'
  :: NeuralNetwork Float -> Matrix Float -> Matrix Float
forward' net dta = fst $ pass net (dta, undefined)

-- | Both forward and backward neural network passes
-- TODO: define only the forward pass and use `backprop` package
-- for automatic differentiation
-- TODO: use mini-batches by default (starting with 3 Dim Arrays)
pass
  :: NeuralNetwork Float
  -- ^ `NeuralNetwork` `Layer`s: weights and activations
  -> (Matrix Float, Matrix Float)
  -- ^ Data set
  -> (Matrix Float, [Gradients Float])
  -- ^ NN computation from forward pass and weights gradients
pass net (x, tgt) = (pred, grads)
  where
    (_, pred, grads) = _pass x net

    _pass inp [] = (loss', pred, [])
      where
        pred = sigmoid inp
        -- Gradient of cross-entropy loss
        -- after sigmoid activation.
        loss' = compute $ pred .- tgt

    _pass inp (Layer w b sact:layers) = (dX, pred, Gradients dW dB:t)
      where
        bBroadcasted = expandWithin Dim2 (rows inp) (\e _ -> e) b
        lin = compute $ (inp |*| w) .+ bBroadcasted
        y = getActivation sact lin

        (dZ, pred, t) = _pass y layers

        dY = getActivation' sact lin dZ
        dW = linearW' inp dY
        dB = bias' dY
        dX = linearX' w dY

optimize
  :: Float
  -- ^ Learning rate
  -> Int
  -- ^ No of iterations
  -> NeuralNetwork Float
  -- ^ Neural network
  -> (Matrix Float, Matrix Float)
  -- ^ Dataset
  -> NeuralNetwork Float
  -- ^ Updated neural network
optimize lr n net0 dataSet = iterN n step net0
  where
    step net = zipWith f net dW
      where
        (_, dW) = pass net dataSet

    f :: Layer Float
      -> Gradients Float
      -> Layer Float
    f (Layer w b act) (Gradients dW dB) =
      -- Layer (compute $ w .- (A.map (lr *) dW)) (compute $ b .- (A.map (lr *) dB)) act
      Layer (compute $ w .- lr `_scale` dW) (compute $ b .- lr `_scale` dB) act

-- | Strict left fold
iterN :: Int -> (a -> a) -> a -> a
iterN n f x0 = foldl' (\x _ -> f x) x0 [1..n]

data AdamParameters = AdamParameters { _beta1 :: Float
                                     , _beta2 :: Float
                                     , _epsilon :: Float
                                     , _lr :: Float
                                     }

-- | Adam optimizer parameters
adamParams = AdamParameters { _beta1 = 0.9
                            , _beta2 = 0.999
                            , _epsilon = 1e-8
                            , _lr = 0.001  -- ^ Learning rate
                            }

-- | Adam optimization
adam
  :: AdamParameters
     -- ^ Adam parameters
     -> Int
     -- ^ No of iterations
     -> NeuralNetwork Float
     -- ^ Neural network layers
     -> (Matrix Float, Matrix Float)
     -- ^ Dataset
     -> NeuralNetwork Float
adam p iterN w0 dataSet = w
  where
    s0 = map zf w0
    v0 = map zf w0
    zf (Layer a b _) = (zerosLike a, zerosLike b)
    zerosLike m = A.replicate Par (size m) 0.0
    (w, _, _) = _adam p iterN (w0, s0, v0) dataSet

_adam
  :: AdamParameters
     -> Int
     -> ([Layer Float], [(Matrix Float, Vector Float)], [(Matrix Float, Vector Float)])
     -> (Matrix Float, Matrix Float)
     -> ([Layer Float], [(Matrix Float, Vector Float)], [(Matrix Float, Vector Float)])
_adam p@AdamParameters { _lr = lr
                       , _beta1 = beta1
                       , _beta2 = beta2
                       , _epsilon = epsilon
                       } n (w0, s0, v0) dataSet = iterN n step (w0, s0, v0)
  where
    step (w, s, v) = (wN, sN, vN)
      where
        (_, dW) = pass w dataSet

        sN = zipWith f2 s dW
        vN = zipWith f3 v dW
        wN = Prelude.zipWith3 f w vN sN

        f :: Layer Float
          -> (Matrix Float, Vector Float)
          -> (Matrix Float, Vector Float)
          -> Layer Float
        f (Layer w_ b_ sf) (vW, vB) (sW, sB) =
           Layer (compute $ w_ .- lr `_scale` vW ./ ((A.map sqrt sW) `addC` epsilon))
                 (compute $ b_ .- lr `_scale` vB ./ ((A.map sqrt sB) `addC` epsilon))
                 sf

        addC m c = A.map (c +) m

        f2 :: (Matrix Float, Vector Float)
           -> Gradients Float
           -> (Matrix Float, Vector Float)
        f2 (sW, sB) (Gradients dW dB) =
          ( compute $ beta2 `_scale` sW .+ (1 - beta2) `_scale` (dW.^2)
          , compute $ beta2 `_scale` sB .+ (1 - beta2) `_scale` (dB.^2))

        f3 :: (Matrix Float, Vector Float)
           -> Gradients Float
           -> (Matrix Float, Vector Float)
        f3 (vW, vB) (Gradients dW dB) =
          ( compute $ beta1 `_scale` vW .+ (1 - beta1) `_scale` dW
          , compute $ beta1 `_scale` vB .+ (1 - beta1) `_scale` dB)

-- | Generate random weights and biases
genWeights
  :: (Int, Int)
  -> IO (Matrix Float, Vector Float)
genWeights (nin, nout) = do
  w <- setComp Par <$> _genWeights (nin, nout)
  b <- setComp Par <$> _genBiases nout
  return (w, b)
    where
      _genWeights (nin, nout) = scale k <$> randn sz
        where
          sz = Sz (nin :. nout)
          k = sqrt (1.0 / fromIntegral nin)

      _genBiases n = randn (Sz n)

-- | Generate a neural network with random weights
genNetwork
  :: [Int] -> [Activation] -> IO (NeuralNetwork Float)
genNetwork nodes activations = do
    weights <- Prelude.mapM genWeights nodes'
    return (zipWith (\(w, b) a -> Layer w b a) weights activations)
  where
    nodes' = zip nodes (tail nodes)

-- | Perform a binary classification
inferBinary
  :: NeuralNetwork Float -> Matrix Float -> Matrix Float
inferBinary net dta =
  let pred = forward' net dta
  -- Thresholding the NN output
  in compute $ A.map (\a -> if a < 0.5 then 0 else 1) pred

-- | Binary classification accuracy in percent
accuracy
  :: [Layer Float]
  -- ^ Neural network
  -> (Matrix Float, Matrix Float)
  -- ^ Dataset
  -> Float
accuracy net (dta, tgt) = 100 * (1 - e / m)
  where
    pred = net `inferBinary` dta
    e = A.sum $ abs (tgt .- pred)
    m = fromIntegral $ rows tgt
