{- Greek letters:

  ϕ - policy (actor) parameters (neural network weights)
  θ - value (critic) network parameters
  γ - discount factor

  Latin letters:

  s - state
  o - observations
  a - action
  r - reward
  s' - next state
-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

import GHC.Generics
import Text.Printf ( printf )
import Torch
import Torch.Serialize
import Torch.Internal.Managed.Type.Context ( manual_seed_L )
import qualified Torch.Optim.CppOptim as Cpp
import qualified Torch.Distributions.Categorical as Categorical
import qualified Torch.Distributions.Distribution as D
import Prelude hiding ( exp
                      , tanh )
import Data.Default.Class
import Control.Monad.State ( foldM )
import qualified Data.List as L
import Control.Monad.Extra ( unfoldM )

import qualified CartPole as Env
import Lib ( Action (..)
           , Space (..)
           , Reward
           )


ppo cfg@Config {..} seed (agent, trainer) i = do
  -- Rollout
  let s0 = Env.reset (seed + i)
  (!obs, !acs, !rs, !logprobs, !dones, !vs) <- rollout maxSteps agent s0

  -- Episode length without the auxiliary final step
  let numSteps = length obs - 1

  -- Sum the rewards
  putStrLn $ "Episode " ++ show i ++ " - Score " ++ show (sum $ L.take numSteps rs)

  -- Advantages
  let advan = L.take numSteps $  -- Ignore the final step
         advantages γ γ' rs dones vs

  -- Optimize
  -- Convert to tensors
  let obs_t = asTensor $ L.take numSteps obs
      acs_t = asTensor $ L.take numSteps acs
      val_t = asTensor $ L.take numSteps vs
      logprobs_t = asTensor $ L.take numSteps logprobs
      advantages_t = asTensor advan
      returns_t = advantages_t + val_t

  let optLoop = optimize cfg obs_t acs_t val_t logprobs_t advantages_t returns_t
  (agent', trainer') <- foldM (\at _ -> optLoop at) (agent, trainer) [1..updateEpochs]
  pure (agent', trainer')

-- Hyperparameters
conf :: Config
conf = Config
    { lr = 2.5e-4
    , clipC = 0.2
    , entC = 0.01
    , vfC = 0.5

    -- Discount factor
    , γ = 0.99

    -- GAE lambda
    , γ' = 0.95

    -- K epochs to update the policy
    , updateEpochs = 4

    -- Number of steps per policy rollout
    , maxSteps = 500
    }

rollout :: Int -> Agent -> Env.State
  -> IO ([Observation], [[Int]], [Reward], [Float], [Bool], [Float])
rollout _maxSteps agent s0 = L.unzip6 <$> unfoldM f (_maxSteps, agent, s0)
  where
    -- Finished max number of steps
    -- and one extra step to be able to calculate the last return
    f (-1, _, _) = pure Nothing

    f (remaining, _agent@Agent{..}, _s) = do
      if Env.isDone _s
        -- The environment is done
        then do
           pure Nothing
        else do
          let ob = Env.observations _s
          (ac@(Action ac_), logprob, _) <- getAction ϕ ob
          let v = value θ ob
              (r, s') = Env.step ac _s
          pure $ Just ((ob, ac_, r, logprob, Env.isDone _s, v), (remaining - 1, _agent, s'))

-- | Get action, logProb, and entropy.
-- Convenience wrapper to use native types rather than tensors.
getAction
  :: Phi  -- ^ Policy weights
  -> [Float]  -- ^ Observations
  -> IO (Action Int, Float, Float)  -- ^ Discrete actions, logProb, entropy
getAction ϕ obs = do
  let obs_ = unsqueeze (Dim 0) $ asTensor obs
  (ac, logprob, entropy) <- evaluate ϕ obs_ Nothing

  -- Return a single discrete action
  return (Action [asValue ac], asValue logprob, asValue entropy)

-- | Get action, logProb, and entropy tensors
evaluate
  :: Phi  -- ^ Policy weights
  -> Tensor  -- ^ Observations
  -> Maybe Tensor  -- ^ Action
  -> IO (Tensor, Tensor, Tensor)
evaluate ϕ obs a = do
      let probs = policy ϕ obs
          dist = Categorical.fromProbs probs
      action <- _getAct a dist
      let logProb = D.logProb dist action
          entropy = D.entropy dist
      pure (action, logProb, entropy)
  where
      _getAct :: Maybe Tensor -> Categorical.Categorical -> IO Tensor
      -- Sample from the categorical distribution:
      -- get a tensor of integer values (one sample per observation).
      -- The list argument is desired tensor shape:
      -- e.g. [1] will create a singleton tensor.
      _getAct Nothing dist = D.sample dist [head $ shape obs]
      _getAct (Just a') _ = pure a'

-- | Generalized advantage estimator (GAE).
--
-- Equation (11) from http://arxiv.org/abs/1707.06347
advantages
  :: Float  -- ^ Discount factor
  -> Float  -- ^ GAE lambda
  -> [Reward] -> [Bool] -> [Float] -> [Float]
advantages γ γ' rs dones vs = f $ L.zip4 rs ((L.drop 1 dones) ++ [undefined]) vs ((L.drop 1 vs) ++ [0.0])
-- vs are current values (denoted by v) and
-- (L.drop 1 vs) are future values (denoted by v')
  where
    -- Not necessary to reverse the list if using lazy evaluation
    f :: [(Float, Bool, Float, Float)] -> [Float]
    -- End of list to be reached: same as terminal (auxiliary value)
    f ((r, _, v, _):[]) = [r - v]

    -- Next state terminal
    f ((r, True, v, _):xs) = (r - v) : f xs

    -- Next state non-terminal
    f ((r, False, v, v'):xs) =
      let a = f xs
          delta = r + γ * v' - v
       in delta + γ * γ' * (head a) : a

optimize :: Config
         -> Tensor -> Tensor -> Tensor -> Tensor -> Tensor -> Tensor
         -> (Agent, Trainer)
         -> IO (Agent, Trainer)
optimize Config {..} obs_t acs_t val_t logprobs_t advantages_t returns_t (Agent {..}, Trainer {..}) = do
  (_, newlogprobs, entropy) <- evaluate ϕ obs_t (Just acs_t)

  let newvalues = critic θ obs_t
      logratio = newlogprobs - logprobs_t
      -- Normalized advantages
      advNorm = (advantages_t - mean advantages_t) / (std advantages_t + 1e-8)

      pg = pgLoss clipC logratio advNorm

      -- vLoss = mean (newvalues - returns_t)^2
      vLoss = clippedValueLoss clipC val_t newvalues returns_t

      entropyLoss = mean entropy

      loss = pg + vfC `mulScalar` vLoss - entC `mulScalar` entropyLoss

  ((θ', ϕ'), opt') <- runStep (θ, ϕ) opt loss (asTensor lr)

  pure (Agent θ' ϕ', Trainer opt')

pgLoss clipC logratio advNorm =
  let ratio = exp logratio
      ratio' = clamp (1 - clipC) (1 + clipC) ratio

      pgLoss1 = -advNorm * ratio
      pgLoss2 = -advNorm * ratio'
   in mean $ max' pgLoss1 pgLoss2

clippedValueLoss clipC val newval ret =
  let lossUnclipped = (newval - ret)^2
      clipped = val + (clamp (-clipC) clipC (newval - val))
      lossClipped = (clipped - ret)^2
      lossMax = max' lossUnclipped lossClipped
   in mean lossMax


-- | Training parameters config
data Config = Config
    {
    -- Learning rate
    lr :: Tensor

    -- Surrogate clipping coefficient
    , clipC :: Float

    , entC :: Float
    , vfC :: Float

    -- Discount factor
    , γ :: Float

    -- GAE lambda
    , γ' :: Float

    -- K epochs to update the policy
    , updateEpochs :: Int

    -- Number of steps per policy rollout
    , maxSteps :: Int
    }

-- Policy (Actor) Network type
data Phi = Phi
  { pl1 :: Linear
  , pl2 :: Linear
  , pl3 :: Linear
  }
  deriving (Generic, Show, Parameterized)

-- Value (Critic) Network type: Three fully-connected layers
data Theta = Theta
  { l1 :: Linear
  , l2 :: Linear
  , l3 :: Linear
  }
  deriving (Generic, Show, Parameterized)

-- | Forward pass in a Policy Network
policy :: Phi -> Tensor -> Tensor
policy Phi {..} state =
  let x = (   linear pl1 ~> tanh
           ~> linear pl2 ~> tanh
           ~> linear pl3 ~> softmax (Dim 1)) state
   in x

-- | Forward pass in a Critic Network
critic :: Theta -> Tensor -> Tensor
critic Theta {..} state =
  let net = linear l1 ~> tanh
            ~> linear l2 ~> tanh
            ~> linear l3
   in net state

-- | Get value: Convenience wrapper around `critic` function.
value :: Theta -> [Float] -> Float
value θ ob =
  let ob_ = unsqueeze (Dim 0) $ asTensor ob
   in asValue $ critic θ ob_

-- Trainable parameters
data Agent = Agent
  { θ :: Theta  -- Value network
  , ϕ :: Phi  -- Policy net
  }
  deriving (Generic, Show)

newtype Trainer = Trainer
  { opt :: Adam
  }
  deriving Generic

-- | A new, untrained agent
mkAgent :: Int -> Int -> IO Agent
mkAgent obsDim actDim = do
  let hiddenDim = 64
  θ <- sampleTheta obsDim hiddenDim
  ϕ <- samplePhi obsDim actDim hiddenDim
  pure $ Agent θ ϕ

-- | Initial Network weights
sampleTheta :: Int -> Int -> IO Theta
sampleTheta obsDim hiddenDim =
  Theta <$> sample (LinearSpec obsDim hiddenDim)
     <*> sample (LinearSpec hiddenDim hiddenDim)
     <*> sample (LinearSpec hiddenDim 1)

-- | Sample random Policy Network weights
samplePhi :: Int -> Int -> Int -> IO Phi
samplePhi obsDim actDim hiddenDim =
  Phi <$> sample (LinearSpec obsDim hiddenDim)
     <*> sample (LinearSpec hiddenDim hiddenDim)
     <*> sample (LinearSpec hiddenDim actDim)

mkTrainer :: Agent -> Trainer
mkTrainer Agent {..} =
  let par = (flattenParameters θ) ++ (flattenParameters ϕ)
      opt = mkAdam 0 0.9 0.999 par
   in Trainer opt

actionDim :: Int
actionDim =
  let Discrete dim = Env.actionSpace
   in dim

obsDim :: Int
obsDim =
  let Continuous dim _ = Env.observationSpace
   in dim

type Observation = [Float]

saveAgent :: FilePath -> Agent -> IO ()
saveAgent path Agent {..} = do
  saveParams θ (path ++ "/value.pt")
  saveParams ϕ (path ++ "/policy.pt")

loadAgent :: FilePath -> IO Agent
loadAgent path = do
  Agent {..} <- mkAgent obsDim actionDim
  θ' <- loadParams θ (path ++ "/value.pt")
  ϕ' <- loadParams ϕ (path ++ "/policy.pt")
  return (Agent θ' ϕ')

-- Iterations in the outermost loop
numEpisodes = 360

main :: IO ()
main = do
  putStrLn $ "Num episodes " ++ show numEpisodes

  -- Seed Torch for reproducibility.
  -- Feel free to remove.
  manual_seed_L 10

  -- Initialize
  agent <- mkAgent obsDim actionDim
  let trainer = mkTrainer agent
      seed = 42

  (agent', trainer') <- foldM (\at i -> (ppo conf) seed at i) (agent, trainer) [1..numEpisodes]

  putStrLn "Saving agent"
  saveAgent "." agent'

  putStrLn "Loading agent"
  agent2 <- loadAgent "."

  (_, _, !rs, _, _, _) <- rollout 1000 agent2 (Env.reset 1000000)
  let numSteps = length rs - 1
  putStrLn $ "Test Score " ++ show (sum $ L.take numSteps rs)

  return ()

max' :: Tensor -> Tensor -> Tensor
max' a b =
  let c = cat (Dim 1) [unsqueeze (Dim 1) a, unsqueeze (Dim 1) b]
   in fst $ maxDim (Dim 1) RemoveDim c

-- Composition operator (reverse)
(~>) :: (a -> b) -> (b -> c) -> a -> c
f ~> g = g. f

{- See also
  https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
  https://github.com/openai/baselines/tree/master/baselines/ppo2
  https://github.com/quantumiracle/Popular-RL-Algorithms/
-}
