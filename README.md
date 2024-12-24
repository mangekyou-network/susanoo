# Susanoo ZKML prediction model for trading on Solana

## Weights/biases scaling:
- Circom only accepts integers as signals, but Tensorflow weights and biases are floating-point numbers.
- In order to simulate a neural network in Circom, weights must be scaled up by `10**m` times. The larger `m` is, the higher the precision.
- Subsequently, biases (if any) must be scaled up by `10**2m` times or even more to maintain the correct output of the network.

Deeper network would have to sacrifice precision, due to the limitation that Circom works under a finite field of modulo `p` which is around 254 bits. As `log(2**254)~76`, we need to make sure total scaling do not aggregate to exceed `10**76` (or even less) times. On average, a network with `l` layers should be scaled by less than or equal to `10**(76//l)` times.


Ps: This is a work in progress. More details will be added later. Check out [susanoo-sdk](https://github.com/mangekyou-network/susanoo-sdk) for more updates.