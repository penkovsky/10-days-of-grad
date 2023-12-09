Reinforce
====

> {-# LANGUAGE BangPatterns #-}
> {-# LANGUAGE DeriveAnyClass #-}
> {-# LANGUAGE DeriveGeneric #-}
> {-# LANGUAGE FlexibleContexts #-}
> {-# LANGUAGE MultiParamTypeClasses #-}
> {-# LANGUAGE RecordWildCards #-}
> {-# LANGUAGE ScopedTypeVariables #-}
> {-# LANGUAGE TypeApplications #-}
> 
> import GHC.Generics
> import Torch
> import Torch.Serialize
> import Torch.Internal.Managed.Type.Context ( manual_seed_L )
> import qualified Torch.Distributions.Categorical as Categorical
> import qualified Torch.Distributions.Distribution as D
> import Prelude hiding ( exp
>                       , tanh )
> import Control.Monad.State ( foldM )
> import qualified Data.List as L
> import Control.Monad.Extra ( unfoldM )
> 
> import qualified CartPole as Env
> import Lib ( Action (..)
>            , Space (..)
>            , Reward
>            )

Reinforce algorithm:

1. Initialize policy parameters $\phi$.
2. Generate trajectories $S_0, A_0, R_1,...,S_{T-1}, A_{T-1}, R_T$ by interacting with environment.
3. Estimate returns.
4. Optimize agent parameters $\phi$.
5. Repeat steps 2-4.

> numEpisodes = 400

> main :: IO ()
> main = do
>   putStrLn $ "Num episodes " ++ show numEpisodes
> 
>   -- Seed Torch for reproducibility.
>   -- Feel free to remove.
>   manual_seed_L 10
>   let seed = 42  -- Environment seed
>
>   -- Step 1: Initialize policy parameters
>   agent <- mkAgent obsDim actionDim
>   -- We also initialize the optimizer
>   let trainer = mkTrainer agent
>
>   -- Repeat steps 2-4 for the number of episodes (trajectories)
>   (agent', trainer') <- foldM (\at i -> (reinforce conf) seed at i) (agent, trainer) [1..numEpisodes]
>
>   return ()

> reinforce cfg@Config {..} seed (agent, trainer) i = do
>   -- Step 2: Trajectories generation (rollout)
>   let s0 = Env.reset (seed + i)
>   (_, _, !rs, !logprobs_t) <- rollout maxSteps agent s0
>   let logprobs' = cat (Dim 0) logprobs_t
>   putStrLn $ "Episode " ++ show i ++ " - Score " ++ show (sum rs)
>
>   -- Step 3: Estimating returns
>   let returns_t = asTensor $ returns γ rs
>       returnsNorm = (returns_t - mean returns_t) / (std returns_t + 1e-8)
>
>   -- Step4: Optimize
>   optimize cfg logprobs' returnsNorm (agent, trainer)

Default hyperparameters

> conf :: Config
> conf = Config
>     { lr = 0.01  -- Learning rate
>
>     , γ = 1.0  -- Discount factor
>
>     , maxSteps = 500  -- Number of steps per policy rollout
>     }

A rollout: generated trajectory by interacting with the environment

> rollout :: Int -> Agent -> Env.State
>   -> IO ([Observation], [[Int]], [Reward], [Tensor])
> rollout _maxSteps agent s0 = L.unzip4 <$> unfoldM f (_maxSteps, agent, s0)
>   where
>     -- Finished max number of steps
>     f (0, _, _) = pure Nothing
> 
>     f (_maxSteps, _agent@Agent{..}, _s) = do
>       if Env.isDone _s
>         -- The environment is done
>         then do
>            pure Nothing
>         else do
>           let ob = Env.observations _s
>           (ac@(Action ac_), logprob) <- getAction ϕ ob
>           let (r, s') = Env.step ac _s
> 
>           pure $ Just ((ob, ac_, r, logprob), (_maxSteps - 1, _agent, s'))

Get action and log probability.

> getAction
>   :: Phi  -- ^ Policy weights
>   -> [Float]  -- ^ Observations
>   -> IO (Action Int, Tensor)  -- ^ Discrete actions and logProb
> getAction ϕ obs = do
>   let obs_ = unsqueeze (Dim 0) $ asTensor obs
>   (ac, logprob) <- evaluate ϕ obs_ Nothing
> 
>   -- Return a single discrete action
>   return (Action [asValue ac], logprob)

Evaluate the policy $\phi$:
If no action provided, sample a new action from the learned distribution.
Get log probabilities for the action.

> evaluate
>   :: Phi  -- ^ Policy weights
>   -> Tensor  -- ^ Observations
>   -> Maybe Tensor  -- ^ Action
>   -> IO (Tensor, Tensor)
> evaluate ϕ obs a = do
>       let probs = policy ϕ obs
>           dist = Categorical.fromProbs probs
>       action <- _getAct a dist
>       let logProb = D.logProb dist action
>       pure (action, logProb)
>   where
>       _getAct :: Maybe Tensor -> Categorical.Categorical -> IO Tensor
>       -- Sample from the categorical distribution:
>       -- get a tensor of integer values (one sample per observation).
>       -- The list argument is a tensor shape:
>       -- e.g. [1] will create a singleton tensor.
>       _getAct Nothing dist = D.sample dist [head $ shape obs]
>       _getAct (Just a') _ = pure a'

Returns estimation

> returns :: Float -> [Reward] -> [Float]
> returns γ rs = f rs
>   where
>     f :: [Float] -> [Float]
>     -- End of list
>     f (r:[]) = [r]
> 
>     f (r:xs) =
>       let y = f xs
>        -- Discounting a future return
>        in r + γ * (head y) : y

Optimizing parameters:

1. Computing the loss.
2. Running a gradient step.

> optimize :: Config
>          -> Tensor -> Tensor
>          -> (Agent, Trainer)
>          -> IO (Agent, Trainer)
> optimize Config {..} logprobs_t returns_t (Agent {..}, Trainer {..}) = do
>   let loss = Torch.sumAll $ -logprobs_t * returns_t
>   (ϕ', opt') <- runStep ϕ opt loss (asTensor lr)
>   pure (Agent ϕ', Trainer opt')

Hyperparameters (config) data type

> data Config = Config
>     {
>     -- Learning rate
>     lr :: Tensor
> 
>     -- Discount factor
>     , γ :: Float
>
>     -- Number of steps per policy rollout
>     , maxSteps :: Int
>     }

Finally, define our policy network.
Here we have three fully-connected layers. That is two hidden layers.

> data Phi = Phi
>   { pl1 :: Linear
>   , pl2 :: Linear
>   , pl3 :: Linear
>   }
>   deriving (Generic, Show, Parameterized)

Forward pass in a Policy Network

> policy :: Phi -> Tensor -> Tensor
> policy Phi {..} state =
>   let x = (   linear pl1 ~> tanh
>            ~> linear pl2 ~> tanh
>            ~> linear pl3 ~> softmax (Dim 1)) state
>    in x

Wrapper: an agent is simply a policy network

> data Agent = Agent
>   { ϕ :: Phi
>   }
>   deriving (Generic, Show)

Trainer contains a single optimizer

> newtype Trainer = Trainer
>   { opt :: Adam
>   }
>   deriving Generic

A new, untrained agent with random weights

> mkAgent :: Int -> Int -> IO Agent
> mkAgent obsDim actDim = do
>   let hiddenDim = 16
>   ϕ <- samplePhi obsDim actDim hiddenDim
>   pure $ Agent ϕ

Parameters $\phi \in \Phi$ initialization

> samplePhi :: Int -> Int -> Int -> IO Phi
> samplePhi obsDim actDim hiddenDim =
>   Phi <$> sample (LinearSpec obsDim hiddenDim)
>      <*> sample (LinearSpec hiddenDim hiddenDim)
>      <*> sample (LinearSpec hiddenDim actDim)

Initializing the trainer

> mkTrainer :: Agent -> Trainer
> mkTrainer Agent {..} =
>   let par = flattenParameters ϕ
>       opt = mkAdam 0 0.9 0.999 par
>    in Trainer opt

Action dimensionality (total number of actions for discrete environment).

> actionDim :: Int
> actionDim =
>   let Discrete dim = Env.actionSpace
>    in dim

Number of observed dimensions.

> obsDim :: Int
> obsDim =
>   let Continuous dim _ = Env.observationSpace
>    in dim

> type Observation = [Float]

Composition operator (reverse)

> (~>) :: (a -> b) -> (b -> c) -> a -> c
> f ~> g = g. f

See also:

* https://github.com/udacity/deep-reinforcement-learning/blob/master/reinforce/REINFORCE.ipynb
* https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
