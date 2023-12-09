module Lib
  ( Action (..)
  , Space (..)
  , Reward
  ) where

import Control.Applicative


newtype Action a = Action [a]
  deriving Show

{- | Space description.

Discrete:
  * The number of elements N. Assume starting from 0, e.g. if N = 3 then the
  space is {0, 1, 2}.

Continuous:
  * Dimensionality (for convenience)
  * Lower and upper bounds
-}
data Space = Discrete Int
           | Continuous Int [(Float, Float)]
  deriving Show

type Reward = Float

