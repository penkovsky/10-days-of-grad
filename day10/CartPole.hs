{-
Based on
  - http://incompleteideas.net/sutton/book/code/pole.c
  - https://github.com/sentenai/reinforce/blob/master/reinforce-environments/src/Environments/CartPole.hs

Discrete action space: either go left or go right.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

Continuous observation space.

-}

module CartPole
  ( actionSpace
  , observationSpace
  , State
  , observations
  , isDone
  , reset
  , reset'
  , step
  , set
  ) where

import Control.Monad.Primitive
import qualified Control.Monad.State as S
import System.Random

import Lib


-- | Left (0) or right (1) - two elements in this space
actionSpace :: Space
actionSpace = Discrete 2

-- | Observations: Position, velocity, angle, angle velocity
observationSpace :: Space
observationSpace = Continuous 4 a
    where
      a = [ (-2 * xThreshold, 2 * xThreshold)
          , (-1e+38, 1e+38)  -- Should be sufficient
          , (-2 * thetaThresholdRadians, 2 * thetaThresholdRadians)
          , (-1e+38, 1e+38)
          ]

newtype State = State ( [Float]  -- Cartpole state
                      , Bool  -- Episode terminated
                      , StdGen
                      )
  deriving Show

-- | Reset the environment to a random state: Reuse the generator.
reset' (State (_, _, g0)) =
  let rnd = uniformR (-0.05, 0.05)
      (x0, g1) = rnd g0
      (x1, g2) = rnd g1
      (x2, g3) = rnd g2
      (x3, g4) = rnd g3
   in State ([x0, x1, x2, x3], False, g4)

-- | Reset the environment to a random state
reset :: Int -> State
reset seed =
  let g0 = mkStdGen seed
      rnd = uniformR (-0.05, 0.05)
      (x0, g1) = rnd g0
      (x1, g2) = rnd g1
      (x2, g3) = rnd g2
      (x3, g4) = rnd g3
   in State ([x0, x1, x2, x3], False, g4)

set :: [Float] -> State
set s =
  let g0 = mkStdGen 0
   in State (s, False, g0)

observations :: State -> [Float]
observations (State (s, _, _)) = s

isDone :: State -> Bool
isDone (State (_, done, _)) = done

-- | Physical parameters
data CartPoleConf = CartPoleConf
  { gravity    :: Float
  , masscart   :: Float
  , masspole   :: Float
  , poleLength :: Float -- ^ Half the pole's length
  , forceMag   :: Float
  , tau        :: Float -- ^ Sampling time = seconds between state updates
  }

defaultConf :: CartPoleConf
defaultConf = CartPoleConf
  { gravity = 9.8
  , masscart = 1.0
  , masspole = 0.1
  , poleLength = 0.5
  , forceMag = 10.0
  , tau = 0.02
  }

polemassLength :: CartPoleConf -> Float
polemassLength s = masspole s * poleLength s

totalMass :: CartPoleConf -> Float
totalMass s = masspole s + masscart s

-- | Angle at which the episode fails
thetaThresholdRadians :: Float
thetaThresholdRadians = 12 * 2 * pi / 360

xThreshold :: Float
xThreshold = 2.4

hasFallen :: [Float] -> Bool
hasFallen s =
  let position: _: angle: _ = s
   in position < (-1 * xThreshold)
      || position > xThreshold
      ||    angle < (-1 * thetaThresholdRadians)
      ||    angle > thetaThresholdRadians

-- | A transition for given state and action.
step :: Action Int -> State -> (Reward, State)
step = step' defaultConf

step' :: CartPoleConf -> Action Int -> State -> (Reward, State)
step' conf (Action (a:_)) (State (s, done, g0)) =
  let [x, xDot, theta, thetaDot] = s

      force    = (if a == 0 then -1 else 1) * forceMag conf
      costheta = cos theta
      sintheta = sin theta

      temp     = (force + polemassLength conf * (thetaDot ** 2) * sintheta) / totalMass conf
      thetaacc = (gravity conf * sintheta - costheta * temp)
                 / (poleLength conf * (4 / 3 - masspole conf * (costheta ** 2) / totalMass conf))
      xacc     = temp - polemassLength conf * thetaacc * costheta / totalMass conf

      s' =  [ x        + tau conf * xDot
            , xDot     + tau conf * xacc
            , theta    + tau conf * thetaDot
            , thetaDot + tau conf * thetaacc
            ]

      r = if done then 0 else 1

      done' = (hasFallen s') || done

   in (r, State (s', done', g0))
