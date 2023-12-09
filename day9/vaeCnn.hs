{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

import Control.Monad ( forM_, forM, when, (<=<) )
import Control.Monad.Cont ( ContT (..) )
import Data.List (foldl')
import System.Environment ( getArgs )
import GHC.Generics
import Pipes hiding ( (~>) )
import qualified Pipes.Prelude as P
import Text.Printf ( printf )
import Torch
import Torch.Serialize
import Torch.Typed.Vision ( initMnist, MnistData )
import qualified Torch.Vision as V
import Torch.Lens ( HasTypes (..)
                  , over
                  , types )
import Prelude hiding ( exp )
import qualified Torch.Optim.CppOptim as Cpp
import Data.Default.Class
import Control.Exception
import System.CPUTime

latent_size = 2

encode :: VAE -> Tensor -> (Tensor, Tensor)
encode VAE {..} x0 =
  let enc_ =
          reshape [-1, 1, 28, 28]  -- Reshape [batch_size x 784] into images of [batch_size x 1 x 28 x 28]
          ~> conv2dForward c1 (2, 2) (0, 0)  -- Stride 2, padding 0
          ~> relu
          ~> conv2dForward c2 (2, 2) (0, 0)
          ~> relu
          ~> conv2dForward c3 (2, 2) (0, 0)
          ~> relu
          ~> flatten (Dim 1) (Dim (-1))

      x1 = enc_ x0
      mu = linear lMu x1
      logSigma = linear lSigma x1
   in (mu, logSigma)

decode :: VAE -> Tensor -> Tensor
decode VAE {..} =
         linear l
         ~> relu
         ~> reshape [-1, 1024, 1, 1]  -- unsqueeze(-1).unsqueeze(-1)
         ~> convTranspose2dForward t1 (2, 2) (0, 0)  -- Stride 2, padding 0
         ~> relu
         ~> convTranspose2dForward t2 (2, 2) (0, 0)
         ~> relu
         ~> convTranspose2dForward t3 (2, 2) (0, 0)
         ~> sigmoid
         ~> reshape [-1, 784]  -- Reshape back

data VAESpec = VAESpec
  {
    -- Encoder trainable parameters
    conv1 :: Conv2dSpec,
    conv2 :: Conv2dSpec,
    conv3 :: Conv2dSpec,
    fcMu :: LinearSpec,
    fcSigma :: LinearSpec,

    -- Decoder trainable parameters
    fc :: LinearSpec,
    deconv1 :: ConvTranspose2dSpec,
    deconv2 :: ConvTranspose2dSpec,
    deconv3 :: ConvTranspose2dSpec
  }
  deriving (Show, Eq)

myConfig =
  VAESpec
    (Conv2dSpec 1 32 4 4)    -- 1 -> 32 channels; 4 x 4 kernel
    (Conv2dSpec 32 64 4 4)   -- 32 -> 64 channels; 4 x 4 kernel
    (Conv2dSpec 64 128 3 3)  -- 64 -> 128 channels; 3 x 3 kernel
    (LinearSpec (2 * 2 * 128) latent_size)
    (LinearSpec (2 * 2 * 128) latent_size)
    (LinearSpec latent_size 1024)
    (ConvTranspose2dSpec 1024 256 4 4)
    (ConvTranspose2dSpec 256 128 6 6)
    (ConvTranspose2dSpec 128 1 6 6)

data VAE = VAE
  { c1 :: Conv2d,
    c2 :: Conv2d,
    c3 :: Conv2d,
    lMu :: Linear,
    lSigma :: Linear,
    l :: Linear,
    t1 :: ConvTranspose2d,
    t2 :: ConvTranspose2d,
    t3 :: ConvTranspose2d
  }
  deriving (Generic, Show, Parameterized)

instance Randomizable VAESpec VAE where
  sample VAESpec {..} =
    VAE
      <$> sample conv1
      <*> sample conv2
      <*> sample conv3
      <*> sample fcMu
      <*> sample fcSigma
      <*> sample fc
      <*> sample deconv1
      <*> sample deconv2
      <*> sample deconv3

vaeForward :: VAE -> Bool -> Tensor -> IO (Tensor, Tensor, Tensor)
vaeForward net@(VAE {..}) _ x0 = do
  let (mu, logSigma) = encode net x0
      sigma = exp (0.5 * logSigma)

  eps <- toLocalModel' <$> randnLikeIO sigma
  -- eps <- toLocalModel' <$> randLikeIO' sigma

  let z = (eps `mul` sigma) `add` mu
      reconstruction = decode net z

  return (reconstruction, mu, logSigma)

(~>) :: (a -> b) -> (b -> c) -> a -> c
f ~> g = g. f

toLocalModel :: forall a. HasTypes a Tensor => Device -> DType -> a -> a
toLocalModel device' dtype' = over (types @Tensor @a) (toDevice device')

fromLocalModel :: forall a. HasTypes a Tensor => a -> a
fromLocalModel = over (types @Tensor @a) (toDevice (Device CPU 0))

-- On GPU
toLocalModel' :: forall a. HasTypes a Tensor => a -> a
toLocalModel' = toLocalModel (Device CUDA 0) Float

-- On CPU
-- toLocalModel' = toLocalModel (Device CPU 0) Float

-- | Binary cross-entropy loss (adding 1e-10 to avoid log(0))
bceLoss target x =
  let r = -sumAll (x * Torch.log(1e-10 + target) + (1 - x) * Torch.log(1e-10 + 1 - target))
  in r

-- | VAE loss: Reconstruction Error + KL Divergence
vaeLoss :: Float -> Tensor -> Tensor -> Tensor -> Tensor -> Tensor
vaeLoss beta recon_x x mu logSigma = reconLoss + asTensor beta * kld
  where
    reconLoss = bceLoss recon_x x
    kld = -0.5 * sumAll (1 + logSigma - pow (2 :: Int) mu - exp logSigma)

trainLoop
  :: Optimizer o
  => Float -> (VAE, o) -> LearningRate -> ListT IO (Tensor, Tensor) -> IO (VAE, o)
trainLoop beta (model0, opt0) lr = P.foldM step begin done. enumerateData
  where
    step :: Optimizer o => (VAE, o) -> ((Tensor, Tensor), Int) -> IO (VAE, o)
    step (model, opt) args = do
      let ((x, _), iter) = toLocalModel' args
          -- Rescale pixel values [0, 255] -> [0, 1.0].
          -- This is important as the sigmoid activation in decoder can
          -- reach values only between 0 and 1.
          x' = x / 255.0
      (recon_x, mu, logSigma) <- vaeForward model False x'
      let loss = vaeLoss beta recon_x x' mu logSigma
      -- Print loss every 100 batches
      when (iter `mod` 100 == 0) $ do
        putStrLn
          $ printf "Batch: %d | Loss: %.2f" iter (asValue loss :: Float)
      runStep model opt loss lr
    done = pure
    begin = pure (model0, opt0)

train :: Float -> V.MNIST IO -> Int -> VAE -> IO VAE
train beta trainMnist epochs net0 = do
    optimizer <- Cpp.initOptimizer adamOpt net0

    (net', _) <- foldLoop (net0, optimizer) epochs $ \(net', optState) _ ->
      runContT (streamFromMap dsetOpt trainMnist)
      $ trainLoop beta (net', optState) 0.0 . fst

    return net'
  where
    dsetOpt = datasetOpts workers
    workers = 2
    -- Adam optimizer parameters
    adamOpt =
        def
          { Cpp.adamLr = learningRate,
            Cpp.adamBetas = (0.9, 0.999),
            Cpp.adamEps = 1e-8,
            Cpp.adamWeightDecay = 0,
            Cpp.adamAmsgrad = False
          } ::
          Cpp.AdamOptions

save' :: VAE -> FilePath -> IO ()
save' net = save (map toDependent. flattenParameters $ net)

load' :: FilePath -> IO VAE
load' fpath = do
  params <- mapM makeIndependent <=< load $ fpath
  net0 <- sample myConfig
  return $ replaceParameters net0 params

test1 = do
    vae <- toLocalModel' <$> sample myConfig
    dta0 <- toLocalModel' <$> randIO' [8, 784]

    (recon, _, _) <- vaeForward vae False dta0
    print $ shape recon

testLatentSpace :: FilePath -> V.MNIST IO -> VAE -> IO ()
testLatentSpace fn testStream net = do
      runContT (streamFromMap (datasetOpts 2) testStream) $ recordPoints fn net. fst

recordPoints :: FilePath -> VAE -> ListT IO (Tensor, Tensor) -> IO ()
recordPoints logname net = P.foldM step begin done. enumerateData
  where
    step :: () -> ((Tensor, Tensor), Int) -> IO ()
    step () args = do
      let ((input, labels), i) = toLocalModel' args
          (encMu, _) = encode net input
          batchSize = head $ shape encMu

      let s = toStr $ Torch.cat (Dim 1) [reshape [-1, 1] labels, encMu]
      appendFile logname s

      return ()

    done () = pure ()
    begin = pure ()

toStr :: Tensor -> String
toStr dec =
    let a = asValue dec :: [[Float]]
        b = map (unwords. map show) a
     in unlines b

time :: IO t -> IO t
time a = do
    start <- getCPUTime
    v <- a
    end   <- getCPUTime
    let diff = fromIntegral (end - start) / (10^12)
    printf "Computation time: %0.3f sec\n" (diff :: Double)
    return v

learningRate :: Double
learningRate = 1e-3

main = do
    (trainData, testData) <- initMnist "data"
    net0 <- toLocalModel' <$> sample myConfig

    beta_: _ <- getArgs
    putStrLn $ "beta = " ++ beta_

    let beta = read beta_
        trainMnistStream = V.MNIST { batchSize = 128, mnistData = trainData }
        testMnistStream = V.MNIST { batchSize = 128, mnistData = testData }
        epochs = 20
        cpt = printf "VAE-CNN-Aug2022-beta_%s.ht" beta_
        logname = printf "VAE-CNN-beta_%s.log" beta_

    putStrLn "Starting training..."
    net' <- time $ train beta trainMnistStream epochs net0
    putStrLn "Done"

    -- Saving the trained model
    save' net' cpt

    -- Restoring the model
    net <- load' cpt

    -- Test data distribution in the latent space
    putStrLn $ "Saving test dataset distribution to " ++ logname
    _ <- testLatentSpace logname testMnistStream net

    let xs = [-3,-2.7..3::Float]

        -- 2D latent space as a Cartesian product
        zs = [ [x,y] | x<-xs, y<-xs ]

        decoded = Torch.cat (Dim 0) $
                    map (decode net. toLocalModel'. asTensor. (:[])) zs

    writeFile (printf "latent_space_beta_%s.txt" beta_) (toStr decoded)

    putStrLn "Done"
