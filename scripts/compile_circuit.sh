#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Compile circuit
echo "Compiling circuit..."
circom2 circuits/model1_price_test.circom --r1cs --wasm --sym -o build

# Check if compilation was successful
if [ ! -f build/model1_price_test.r1cs ]; then
    echo "Circuit compilation failed!"
    exit 1
fi

# Generate witness
echo "Generating witness..."
node build/model1_price_test_js/generate_witness.js build/model1_price_test_js/model1_price_test.wasm models/model1_price_input.json build/witness.wtns

# Check if witness generation was successful
if [ ! -f build/witness.wtns ]; then
    echo "Witness generation failed!"
    exit 1
fi

# Start a new powers of tau ceremony
echo "Starting powers of tau ceremony..."
snarkjs powersoftau new bn128 12 build/pot12_0000.ptau -v

# Contribute to the ceremony
echo "Contributing to the ceremony..."
echo "Please enter some random text for entropy (press Enter when done):"
read entropy_input
snarkjs powersoftau contribute build/pot12_0000.ptau build/pot12_0001.ptau --name="First contribution" -v

# Phase 2
echo "Preparing phase 2..."
snarkjs powersoftau prepare phase2 build/pot12_0001.ptau build/pot12_final.ptau -v

# Generate zkey
echo "Generating zkey..."
snarkjs groth16 setup build/model1_price_test.r1cs build/pot12_final.ptau build/model1_price_test_0000.zkey

# Contribute to phase 2
echo "Contributing to phase 2..."
echo "Please enter more random text for phase 2 entropy (press Enter when done):"
read phase2_entropy
snarkjs zkey contribute build/model1_price_test_0000.zkey build/model1_price_test_final.zkey --name="1st Contributor" -v

# Export verification key
echo "Exporting verification key..."
snarkjs zkey export verificationkey build/model1_price_test_final.zkey build/verification_key.json

# Generate proof
echo "Generating proof..."
snarkjs groth16 prove build/model1_price_test_final.zkey build/witness.wtns build/proof.json build/public.json

# Export for Solana using groth16-solana
echo "Exporting for Solana..."
node ../groth16-solana/parse_vk_to_rust.js build/verification_key.json src/

echo "✅ Circuit compilation and proof generation complete!" 