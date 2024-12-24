pragma circom 2.0.0;

// Polynomial activation function f(n,x) = x^2 + n*x
// where n is a scaling factor to adjust for floating point weights
template Poly(n) {
    signal input in;
    signal output out;
    
    // Compute x^2 + n*x
    out <== in * in + n * in;
} 