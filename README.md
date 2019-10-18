# Steady-state Kalman smoother and EM algorithm 

Kalman smoother and EM algorithm with
1. Batched EM, in case there are multiple measurement sequences
2. Steady-state optimizations in the forward and backward passes of the E step

For reference, see [Ghahramani & Hinton (1996)](http://mlg.eng.cam.ac.uk/zoubin/papers/tr-96-2.pdf) or [Yu, Shenoy & Sahani (2005)](http://www.gatsby.ucl.ac.uk/~byron/derive_ks.pdf).

To install, clone the FastKF repository, `cd` into the FastKF directory and run

`pip install -e .`

Requirements:
1. numpy
2. sklearn
