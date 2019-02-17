Haskell Guide To Neural Networks
====

Bogdan Penkovsky

How to run this guide iteractively:

< $ stack --install-ghc ghci --resolver lts-11.9 \
< --package backprop-0.2.2.0 --package random \
< haskell-guide-to-neural-networks.lhs
< > main

Now that we have seen how do neural networks work, we realize that understanding
of the gradients flow is essential for survival.  Therefore, we will start with
the elementary building blocks, functions and will check how do the gradients
behave on the microlevel.  However, as neural networks become more complicated,
calculation of gradients by hand becomes a murky business. Yet, fear not young
_padawan_, there is a way out! I am very excited that today finally we will get
acquainted with automatic differentiation, an essential tool in your deep
learning arsenal.  This post was largely inspired by [Hacker's guide to Neural
Networks](http://karpathy.github.io/neuralnets/). For comparison, see also
[Python version](https://github.com/urwithajit9/HG_NeuralNetwork).


Before jumping ahead, you may also want to check the previous posts:

* [Day 1: Learning Neural Networks The Hard Way](http://penkovsky.com/post/neural-networks)
* [Day 2: Why My Network Does Not Work?](http://penkovsky.com/post/neural-networks-2)







We start the project with some pragmas and imports. Feel free
to ignore them for now.

> {-# LANGUAGE FlexibleContexts #-}
> {-# LANGUAGE BangPatterns #-}

> import Control.Monad ( foldM
>                      , (<=<)
>                      , replicateM_
>                      )
> import Numeric.Backprop as BP
> import System.Random ( randomIO )








Why Random Local Search Fails
------

Following the Karpathy's [guide](http://karpathy.github.io/neuralnets/), we will
also consider a simple multiplication circuit.  Well, Haskell is not JavaScript,
so the definition is pretty straightforward:

> forwardMultiplyGate = (*)

Or we could have written

< forwardMultiplyGate x y = x * y

to make the function look more intuitively $f(x,y) = x \cdot y$. Anyway,

< forwardMultiplyGate (-2) 3

returns -6. Exciting.

Now, the question: is it possible to change the input $(x,y)$ slightly in order
to increase the output?  One way would be to perform local random search.

> _search tweakAmount (x, y, bestOut) = do
>   x_try <- (x + ). (tweakAmount *) <$> randomDouble
>   y_try <- (y + ). (tweakAmount *) <$> randomDouble
>   let out = forwardMultiplyGate x_try y_try
>   return $ if out > bestOut
>                then (x_try, y_try, out)
>                else (x, y, bestOut)

Not surprisingly, the function above represents a single iteration of a
"for"-loop. What is does, it randomly selects points around initial $x$ and $y$
and checks if the output has increased.  If yes, then it updates the best known
inputs and the maximal output. To iterate, we can use `foldM :: (b -> a -> IO b)
-> b -> [a] -> IO b` since we anticipate some interaction with "external
world" in the form of random numbers generation:

> localSearch tweakAmount (x0, y0, out0) =
>   foldM (searchStep tweakAmount) (x0, y0, out0) [1..100]

What the code essentially tells us is that we, seeding the algorithm with some
initial values of `x0`, `y0`, and `out0`, iterate over the list starting from 1
and ending with 100.  The core of the algorithm is `searchStep`:

> searchStep ta xyz _ = _search ta xyz

which is a convenience function ignoring the iteration number that glues those
two pieces together. Now, we would like to have a random number generator within
the range of [-1; 1). From the
[documentation](http://hackage.haskell.org/package/random-1.1/docs/System-Random.html),
we know that `randomIO` produces a number between 0 and 1.  Therefore, we scale
the value by multiplying by 2 and subtracting 1:

> randomDouble :: IO Double
> randomDouble = subtract 1. (*2) <$> randomIO

The `<$>` function is a synonym to `fmap`.  What it essentially does is
attaching the pure function `subtract 1. (*2)` which has type `Double ->
Double`, to the "external world" action `randomIO`, which has type `IO Double`
(yes, IO = input/output).

A hack for a numerical minus infinity:

> inf_ = -1.0 / 0

Now, we run `localSearch 0.01 (-2, 3, inf_)` several times:

```
(-1.7887454910045664,2.910160042416705,-5.205535653974539)
(-1.7912166830200635,2.89808308735154,-5.19109477484237)
(-1.8216809458018006,2.8372869694452523,-5.168631610010152)
```

In fact, we see that the outputs have increased from -6 to about -5.2.
But the improvement is only about $0.8/100 = 0.008$ units per iteration.
That is an extremely inefficient method. The problem with random
search is that each time it attempts to change the inputs in random
directions. If the algorithm makes a mistake, it has to discard
the result and start again from the previously known best position.
Wouldn't it be nice if instead each iteration would improve the result
at least by a little bit?


Backpropagation
=======

Instead of random search in random direction, we can use the precise direction
and amount to change the input so that the output would improve. And that tells
us the [gradient](https://en.wikipedia.org/wiki/Gradient).  In his
[article](https://idontgetoutmuch.wordpress.com/2013/10/13/backpropogation-is-just-steepest-descent-with-automatic-differentiation-2/),
Dominic Steinitz explains the differences between the numerical, symbolic, and
automatic differentiation. Here, we will directly explain the last approach,
invaluable for neural networks training.

The idea is that first we will explicitly define the gradient for our elementary
operators. Then, we will exploit the [chain
rule](https://en.wikipedia.org/wiki/Chain_rule) when combining those into neural
networks or whatever we like. That strategy will provide the necessary
gradients. Let us illustrate the concept.

Below we define both multiplication operator and its gradient using the chain
rule, i.e. $d/dt x(t) y(t) = x(t) y'(t) + x'(t) y(t)$:

> (x, x') *. (y, y') = (x * y, x * y' + x' * y)

The same can be done with addition, subtraction, division, exponent:

> (x, x') +. (y, y') = (x + y, x' + y')

> (x, x') /. (y, y') = (x / y, (y * x' - x * y') / y^2)
> x -. y = x +. (negate1 y)

> negate1 (x, x') = (negate x, negate x')

> exp1 (x, x') = (exp x, x' * exp x)

We also have `constOp` for constants:

> constOp :: Double -> (Double, Double)
> constOp x = (x, 0.0)

Finally, we can define our favourite sigmoid $\sigma(x)$
using the operators above:

> sigmoid1 x = constOp 1 /. (constOp 1 +. exp1 (negate1 x))

Now, let us compute a neuron $f(x, y) = \sigma(a * x + b * y + c)$, where $x$
and $y$ are inputs and $a$, $b$, and $c$ are parameters

> neuron1 [a, b, c, x, y] = sigmoid1 ((a *. x) +. (b *. y) +. c)

Now, we can obtain the gradient of `a`:

> abcxy1 :: [(Double, Double)]
> abcxy1 = [(1, 1), (2, 0), (-3, 0), (-1, 0), (3, 0)]

< neuron1 abcxy1
< (0.8807970779778823,-0.1049935854035065)

Here, the first number is the result of the neuron's output
and the second one is the gradient of `a`.

In a similar way, we can obtain the rest of the gradients:

< neuron1 [(1, 0), (2, 1), (-3, 0), (-1, 0), (3, 0)]
< (0.8807970779778823,0.3149807562105195)

< neuron1 [(1, 0), (2, 0), (-3, 1), (-1, 0), (3, 0)]
< (0.8807970779778823,0.1049935854035065)

< neuron1 [(1, 0), (2, 0), (-3, 0), (-1, 1), (3, 0)]
< (0.8807970779778823,0.1049935854035065)

< neuron1 [(1, 0), (2, 0), (-3, 0), (-1, 0), (3, 1)]
< (0.8807970779778823,0.209987170807013)

Introducing backprop library
=======

The [backprop library](https://backprop.jle.im/) specifically designed for
[differentiable
programming](https://www.quora.com/What-is-Differentiable-Programming).  It
provides combinators to reduce our mental overhead.  In addition, the most
useful operations such as arithmetics and trigonometry, have already been
defined in the _backprop_ library. See also
[hmatrix-backprop](http://hackage.haskell.org/package/hmatrix-backprop) for
linear algebra.  So all you need for differentiable programming now is to define
some functions:

> neuron
>   :: Reifies s W
>   => [BVar s Double] -> BVar s Double
> neuron [a, b, c, x, y] = sigmoid (a * x + b * y + c)

> sigmoid x = 1 / (1 + exp (-x))

Here `BVar s` wrapper signifies that our function is
differentiable. Now, the forward pass is:

> forwardNeuron = BP.evalBP (neuron. BP.sequenceVar)

We use `sequenceVar` isomorphism to convert a `BVar` of a list into a list of
`BVar`s, as required by our `neuron` equation. And the backward pass is

> backwardNeuron = BP.gradBP (neuron. BP.sequenceVar)

< abcxy0 :: [Double]
< abcxy0 = [1, 2, (-3), (-1), 3]

< forwardNeuron abcxy0
< -- 0.8807970779778823

< backwardNeuron abcxy0
< -- [-0.1049935854035065,0.3149807562105195,0.1049935854035065,0.1049935854035065,0.209987170807013]

Note that all gradients are in one list, the type of the first `neuron`
argument.  Below we just run the equations we have defined above.

> main = do
>   putStrLn "> forwardMultiplyGate (-2) 3"
>   print $ forwardMultiplyGate (-2) 3
>   putStrLn ""

>   putStrLn "> localSearch 0.01 (-2, 3, inf_) (3 times)"
>   replicateM_ 3 $ (print <=< localSearch 0.01) (-2, 3, inf_)
>   putStrLn ""


>   putStrLn "Testing automatic differentiation:"
>   putStrLn "gradients for parameters a, b, c, x, and y:\n"

>   putStrLn "> neuron1 [(1, 0), (2, 1), (-3, 0), (-1, 0), (3, 0)]"
>   print $ neuron1 [(1, 0), (2, 1), (-3, 0), (-1, 0), (3, 0)]

>   putStrLn "> neuron1 [(1, 0), (2, 1), (-3, 0), (-1, 0), (3, 0)]"
>   print $ neuron1 [(1, 0), (2, 1), (-3, 0), (-1, 0), (3, 0)]

>   putStrLn "> [(1, 0), (2, 0), (-3, 1), (-1, 0), (3, 0)]"
>   print $ neuron1 [(1, 0), (2, 0), (-3, 1), (-1, 0), (3, 0)]

>   putStrLn "> neuron1 [(1, 0), (2, 0), (-3, 0), (-1, 1), (3, 0)]"
>   print $ neuron1 [(1, 0), (2, 0), (-3, 0), (-1, 1), (3, 0)]

>   putStrLn "> neuron1 [(1, 0), (2, 0), (-3, 0), (-1, 0), (3, 1)]"
>   print $ neuron1 [(1, 0), (2, 0), (-3, 0), (-1, 0), (3, 1)]
>   putStrLn ""


>   putStrLn "Testing backprop library"
>   putStrLn "> forwardNeuron [1, 2, (-3), (-1), 3]"
>   print $ forwardNeuron [1, 2, (-3), (-1), 3]
>   putStrLn ""

>   putStrLn "> backwardNeuron [1, 2, (-3), (-1), 3]"
>   print $ backwardNeuron [1, 2, (-3), (-1), 3]


Further reading
=======

* [Visual guide to neural networks](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)
* [Backprop documentation](https://backprop.jle.im/01-getting-started.html)
* [Article on backpropagation by Dominic Steinitz](https://idontgetoutmuch.wordpress.com/2013/10/13/backpropogation-is-just-steepest-descent-with-automatic-differentiation-2/)

% pandoc -f markdown+lhs -t html haskell-guide-to-neural-networks.lhs > guide.html
% pandoc -f markdown+lhs -t latex haskell-guide-to-neural-networks.lhs -o guide.pdf
