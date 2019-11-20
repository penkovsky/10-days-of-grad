import           Criterion.Main
import           Data.Massiv.Array hiding ( map, zip, unzip, zipWith, mapM_ )
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest.Vector as A
import           Data.List ( foldl' )

import           Weights ( im, w0, w1 )
import           NeuralNetwork ( conv2d_ )

-- TODO: benchmark Array P instead of Array U on larger arrays

-- | 2D convolution on individual image
conv2dl'
       :: Padding Ix3 Float  -- ^ Padding
       -> Array U Ix4 Float  -- ^ Weights
       -> Array U Ix3 Float  -- ^ Input features
       -> Array U Ix3 Float  -- ^ Output features
conv2dl' padding w x = compute res
  where
    (Sz (cout :> cin :> _ :. _)) = size w
    base = computeAs U $ applyStencil padding (makeCorrelationStencilFromKernel $ w !> 0) x
    res = foldl' (\prev ch -> let stencil = makeCorrelationStencilFromKernel $ w !> ch
                                  conv = computeAs U $ applyStencil padding stencil x
                              in computeAs U $ append' 3 prev conv) base [1..cout - 1]

main :: IO ()
main = do
  let pad2 = Padding (Sz2 2 2) (Sz2 2 2) (Fill 0.0) :: Padding Ix2 Float
      nopad2 = Padding (Sz2 0 0) (Sz2 0 0) (Fill 0.0) :: Padding Ix2 Float
      pad3 = Padding (Sz3 0 2 2) (Sz3 0 2 2) (Fill 0.0) :: Padding Ix3 Float
      nopad3 = Padding (Sz3 0 0 0) (Sz3 0 0 0) (Fill 0.0) :: Padding Ix3 Float
      fm0 = computeAs U. conv2dl' pad3 w0. resize' (Sz (1 :> 28 :. 28)) $ im
      fm' = computeAs U. conv2d_ pad2 w0. resize' (Sz (1 :> 1 :> 28 :. 28)) $ im
      fm3 = computeAs U. conv2d_ pad2 w0. computeAs U. expandWithin Dim4 bs1 const. resize' (Sz (1 :> 28 :. 28)) $ im
      bs1 = 64 :: Sz1
  defaultMain
    [ bgroup
        "Conv2d: no batches (3D)"
        [ bgroup
            "1 chan -> 3 chan, with padding"
            [ env
                (return (setComp Seq $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Seq".
                 whnf (computeAs U. conv2dl' pad3 w0))
            , env
                (return (setComp Par $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Par".
                 whnf (computeAs U. conv2dl' pad3 w0))
            ]
        , bgroup
            "3 chan -> 3 chan, no padding"
            [ env
                (return (setComp Seq $ computeAs U fm0))
                (bench "Seq".
                 whnf (computeAs U. conv2dl' nopad3 w1))
            , env
                (return (setComp Par $ computeAs U fm0))
                (bench "Par".
                 whnf (computeAs U. conv2dl' nopad3 w1))
            ]
        ]

    , bgroup
        "Conv2d: batch size 1 (4D)"
        [ bgroup
            "1 chan -> 3 chan, with padding"
            [ env
                (return (setComp Seq $ resize' (Sz (1 :> 1 :> 28 :. 28)) im))
                (bench "Seq".
                 whnf (computeAs U. conv2d_ pad2 w0))
            , env
                (return (setComp Par $ resize' (Sz (1 :> 1 :> 28 :. 28)) im))
                (bench "Par".
                 whnf (computeAs U. conv2d_ pad2 w0))
            ]
        , bgroup
            "3 chan -> 3 chan, no padding"
            [ env
                (return (setComp Seq $ computeAs U fm'))
                (bench "Seq".
                 whnf (computeAs U. conv2d_ nopad2 w1))
            , env
                (return (setComp Par $ computeAs U fm'))
                (bench "Par".
                 whnf (computeAs U. conv2d_ nopad2 w1))
            ]
        ]

    , bgroup
        ("Conv2d: batch size " ++ show bs1 ++ " (4D)")  -- Increased batch size to see if parallelization is useful
        [ bgroup
            "1 chan -> 3 chan, with padding"
            [ env
                (return (setComp Seq $ computeAs U $ expandWithin Dim4 bs1 const $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Seq".
                 whnf (computeAs U. conv2d_ pad2 w0))
            , env
                (return (setComp Par $ computeAs U $ expandWithin Dim4 bs1 const $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Par".
                 whnf (computeAs U. conv2d_ pad2 w0))
            ]
        , bgroup
            "3 chan -> 3 chan, no padding"
            [ env
                (return (setComp Seq $ computeAs U fm3))
                (bench "Seq".
                 whnf (computeAs U. conv2d_ nopad2 w1))
            , env
                (return (setComp Par $ computeAs U fm3))
                (bench "Par".
                 whnf (computeAs U. conv2d_ nopad2 w1))
            ]
        ]

    ]

-- First test (1 -> 3 channels):
-- 280.8 us (single image) -> 15.94/64=249us (batch of 64)
--
-- Second test (3 -> 3 channels):
-- mean 512.6 us (single im) -> 25.16ms/64=393us (batch of 64)

{-

Dell XPS Laptop (6 cores only):

benchmarking Conv2d: no batches (3D)/1 chan -> 3 chan, with padding/Seq
time                 281.3 μs   (280.0 μs .. 283.1 μs)
                     1.000 R²   (0.999 R² .. 1.000 R²)
mean                 280.8 μs   (279.9 μs .. 282.4 μs)
std dev              3.947 μs   (2.927 μs .. 5.228 μs)

benchmarking Conv2d: no batches (3D)/1 chan -> 3 chan, with padding/Par
time                 489.6 μs   (484.2 μs .. 495.4 μs)
                     0.998 R²   (0.996 R² .. 0.999 R²)
mean                 508.5 μs   (499.9 μs .. 520.8 μs)
std dev              39.22 μs   (21.73 μs .. 59.48 μs)
variance introduced by outliers: 65% (severely inflated)

benchmarking Conv2d: no batches (3D)/3 chan -> 3 chan, no padding/Seq
time                 515.5 μs   (512.4 μs .. 519.2 μs)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 512.6 μs   (510.8 μs .. 514.6 μs)
std dev              6.679 μs   (5.791 μs .. 7.931 μs)

benchmarking Conv2d: no batches (3D)/3 chan -> 3 chan, no padding/Par
time                 724.2 μs   (714.2 μs .. 735.3 μs)
                     0.997 R²   (0.995 R² .. 0.998 R²)
mean                 750.0 μs   (742.8 μs .. 757.8 μs)
std dev              26.00 μs   (22.58 μs .. 30.43 μs)
variance introduced by outliers: 25% (moderately inflated)

benchmarking Conv2d: batch size 1 (4D)/1 chan -> 3 chan, with padding/Seq
time                 712.2 μs   (710.6 μs .. 714.2 μs)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 716.4 μs   (715.0 μs .. 718.5 μs)
std dev              5.481 μs   (4.651 μs .. 6.995 μs)

benchmarking Conv2d: batch size 1 (4D)/1 chan -> 3 chan, with padding/Par
time                 897.4 μs   (880.8 μs .. 913.4 μs)
                     0.998 R²   (0.997 R² .. 0.999 R²)
mean                 914.4 μs   (906.1 μs .. 925.3 μs)
std dev              33.71 μs   (26.76 μs .. 44.71 μs)
variance introduced by outliers: 27% (moderately inflated)

benchmarking Conv2d: batch size 1 (4D)/3 chan -> 3 chan, no padding/Seq
time                 1.144 ms   (1.137 ms .. 1.153 ms)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 1.155 ms   (1.152 ms .. 1.159 ms)
std dev              12.73 μs   (10.42 μs .. 16.88 μs)

benchmarking Conv2d: batch size 1 (4D)/3 chan -> 3 chan, no padding/Par
time                 1.419 ms   (1.375 ms .. 1.477 ms)
                     0.994 R²   (0.991 R² .. 0.996 R²)
mean                 1.435 ms   (1.414 ms .. 1.459 ms)
std dev              81.70 μs   (70.75 μs .. 97.37 μs)
variance introduced by outliers: 44% (moderately inflated)

benchmarking Conv2d: batch size Sz1 64 (4D)/1 chan -> 3 chan, with padding/Seq
time                 41.46 ms   (40.82 ms .. 42.30 ms)
                     0.999 R²   (0.998 R² .. 1.000 R²)
mean                 42.07 ms   (41.76 ms .. 42.37 ms)
std dev              626.3 μs   (516.3 μs .. 729.3 μs)

benchmarking Conv2d: batch size Sz1 64 (4D)/1 chan -> 3 chan, with padding/Par
time                 19.07 ms   (18.41 ms .. 19.98 ms)
                     0.987 R²   (0.978 R² .. 0.995 R²)
mean                 15.94 ms   (14.97 ms .. 16.73 ms)
std dev              2.067 ms   (1.829 ms .. 2.303 ms)
variance introduced by outliers: 59% (severely inflated)

benchmarking Conv2d: batch size Sz1 64 (4D)/3 chan -> 3 chan, no padding/Seq
time                 65.34 ms   (64.09 ms .. 67.39 ms)
                     0.999 R²   (0.997 R² .. 1.000 R²)
mean                 70.75 ms   (68.34 ms .. 75.19 ms)
std dev              5.408 ms   (1.756 ms .. 7.841 ms)
variance introduced by outliers: 25% (moderately inflated)

benchmarking Conv2d: batch size Sz1 64 (4D)/3 chan -> 3 chan, no padding/Par
time                 28.22 ms   (27.36 ms .. 29.87 ms)
                     0.994 R²   (0.990 R² .. 0.997 R²)
mean                 25.16 ms   (23.76 ms .. 26.48 ms)
std dev              3.221 ms   (2.329 ms .. 3.838 ms)
variance introduced by outliers: 56% (severely inflated)



Dell Precision 5820 desktop (6 cores only):

benchmarking Conv2d: no batches (3D)/1 chan -> 3 chan, with padding/Seq
time                 288.9 μs   (287.9 μs .. 290.3 μs)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 288.8 μs   (288.2 μs .. 289.8 μs)
std dev              2.822 μs   (2.062 μs .. 4.315 μs)

benchmarking Conv2d: no batches (3D)/1 chan -> 3 chan, with padding/Par
time                 773.9 μs   (664.1 μs .. 887.4 μs)
                     0.886 R²   (0.839 R² .. 0.932 R²)
mean                 989.8 μs   (926.3 μs .. 1.063 ms)
std dev              251.6 μs   (219.6 μs .. 296.3 μs)
variance introduced by outliers: 96% (severely inflated)

benchmarking Conv2d: no batches (3D)/3 chan -> 3 chan, no padding/Seq
time                 496.5 μs   (495.7 μs .. 497.2 μs)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 496.8 μs   (496.4 μs .. 497.4 μs)
std dev              1.759 μs   (1.034 μs .. 2.673 μs)

benchmarking Conv2d: no batches (3D)/3 chan -> 3 chan, no padding/Par
time                 910.7 μs   (860.5 μs .. 992.6 μs)
                     0.945 R²   (0.910 R² .. 0.977 R²)
mean                 935.6 μs   (889.0 μs .. 996.5 μs)
std dev              172.7 μs   (123.0 μs .. 238.5 μs)
variance introduced by outliers: 91% (severely inflated)

benchmarking Conv2d: batch size 1 (4D)/1 chan -> 3 chan, with padding/Seq
time                 659.2 μs   (653.5 μs .. 668.8 μs)
                     0.999 R²   (0.998 R² .. 1.000 R²)
mean                 659.0 μs   (656.4 μs .. 665.9 μs)
std dev              12.02 μs   (4.109 μs .. 21.43 μs)

benchmarking Conv2d: batch size 1 (4D)/1 chan -> 3 chan, with padding/Par
time                 1.065 ms   (1.022 ms .. 1.140 ms)
                     0.970 R²   (0.948 R² .. 0.988 R²)
mean                 1.073 ms   (1.039 ms .. 1.120 ms)
std dev              142.9 μs   (101.5 μs .. 187.6 μs)
variance introduced by outliers: 83% (severely inflated)

benchmarking Conv2d: batch size 1 (4D)/3 chan -> 3 chan, no padding/Seq
time                 1.046 ms   (1.044 ms .. 1.048 ms)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 1.046 ms   (1.044 ms .. 1.047 ms)
std dev              4.230 μs   (3.378 μs .. 5.493 μs)

benchmarking Conv2d: batch size 1 (4D)/3 chan -> 3 chan, no padding/Par
time                 1.442 ms   (1.368 ms .. 1.574 ms)
                     0.941 R²   (0.885 R² .. 0.990 R²)
mean                 1.527 ms   (1.459 ms .. 1.640 ms)
std dev              281.9 μs   (198.0 μs .. 419.3 μs)
variance introduced by outliers: 90% (severely inflated)

benchmarking Conv2d: batch size Sz1 64 (4D)/1 chan -> 3 chan, with padding/Seq
time                 42.91 ms   (42.20 ms .. 43.74 ms)
                     0.999 R²   (0.998 R² .. 1.000 R²)
mean                 42.48 ms   (42.30 ms .. 42.83 ms)
std dev              501.4 μs   (179.3 μs .. 870.8 μs)

benchmarking Conv2d: batch size Sz1 64 (4D)/1 chan -> 3 chan, with padding/Par
time                 8.757 ms   (7.805 ms .. 9.608 ms)
                     0.958 R²   (0.942 R² .. 0.987 R²)
mean                 8.578 ms   (8.261 ms .. 8.899 ms)
std dev              1.031 ms   (864.6 μs .. 1.184 ms)
variance introduced by outliers: 65% (severely inflated)

benchmarking Conv2d: batch size Sz1 64 (4D)/3 chan -> 3 chan, no padding/Seq
time                 67.12 ms   (66.96 ms .. 67.25 ms)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 67.32 ms   (67.22 ms .. 67.58 ms)
std dev              244.1 μs   (129.4 μs .. 339.3 μs)

benchmarking Conv2d: batch size Sz1 64 (4D)/3 chan -> 3 chan, no padding/Par
time                 12.43 ms   (11.90 ms .. 13.37 ms)
                     0.965 R²   (0.939 R² .. 0.990 R²)
mean                 13.54 ms   (13.06 ms .. 14.11 ms)
std dev              1.397 ms   (1.082 ms .. 1.717 ms)
variance introduced by outliers: 52% (severely inflated)

-}
