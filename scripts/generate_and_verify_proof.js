const snarkjs = require("snarkjs");
const { Connection, PublicKey, Keypair, Transaction, TransactionInstruction, ComputeBudgetProgram, sendAndConfirmTransaction } = require('@solana/web3.js');
const fs = require('fs');
const { buildBn128, utils } = require("ffjavascript");
const { unstringifyBigInts } = utils;

function g1Uncompressed(curve, p1Raw) {
    let p1 = curve.G1.fromObject(p1Raw);
    let buff = new Uint8Array(64);
    curve.G1.toRprUncompressed(buff, 0, p1);
    return Buffer.from(buff);
}

function g2Uncompressed(curve, p2Raw) {
    let p2 = curve.G2.fromObject(p2Raw);
    let buff = new Uint8Array(128);
    curve.G2.toRprUncompressed(buff, 0, p2);
    return Buffer.from(buff);
}

function to32ByteBuffer(bigInt) {
    const hexString = bigInt.toString(16).padStart(64, '0');
    return Buffer.from(hexString, "hex");
}

async function generateAndFormatProof() {
    // Load input data
    const input = require('../models/model1_price_input.json');

    // Generate witness
    console.log('Generating witness...');
    try {
        const { proof, publicSignals } = await snarkjs.groth16.fullProve(
            input,
            "build/model1_price_test_js/model1_price_test.wasm",
            "build/model1_price_test_final.zkey"
        );

        // Save proof and public signals for debugging
        fs.writeFileSync('proof.json', JSON.stringify(proof, null, 2));
        fs.writeFileSync('public.json', JSON.stringify(publicSignals, null, 2));

        console.log('Generated proof structure:', JSON.stringify(proof, null, 2));
        console.log('Public signals:', publicSignals);

        // Process the proof using ffjavascript
        const curve = await buildBn128();
        const proofProc = unstringifyBigInts(proof);
        const publicSignalsProc = unstringifyBigInts(publicSignals);

        // Convert to uncompressed points
        let pi_a = g1Uncompressed(curve, proofProc.pi_a);
        const pi_b = g2Uncompressed(curve, proofProc.pi_b);
        const pi_c = g1Uncompressed(curve, proofProc.pi_c);

        // Negate pi_a point
        const p1 = curve.G1.fromRprUncompressed(pi_a, 0);
        const p1Neg = curve.G1.neg(p1);
        const pi_a_neg = new Uint8Array(64);
        curve.G1.toRprUncompressed(pi_a_neg, 0, p1Neg);
        pi_a = Buffer.from(pi_a_neg);

        // Convert public signals
        const publicSignalsBuffer = to32ByteBuffer(BigInt(publicSignalsProc[0]));

        // Log the formatted components
        console.log('Formatted proof components:');
        console.log('pi_a (negated):', Array.from(pi_a));
        console.log('pi_b (first 64 bytes):', Array.from(pi_b).slice(0, 64));
        console.log('pi_b (last 64 bytes):', Array.from(pi_b).slice(64, 128));
        console.log('pi_c:', Array.from(pi_c));
        console.log('public signals:', Array.from(publicSignalsBuffer));

        return { pi_a, pi_b, pi_c, publicSignalsBuffer };
    } catch (error) {
        console.error('Error generating proof:', error);
        throw error;
    }
}

async function verifyProof(programId, proofData) {
    // Connect to devnet
    const connection = new Connection('https://api.devnet.solana.com', 'confirmed');

    // Load keypair from file
    let payer;
    try {
        const secretKeyString = fs.readFileSync(process.env.KEYPAIR_PATH || '~/.config/solana/id.json', 'utf8');
        const secretKey = Uint8Array.from(JSON.parse(secretKeyString));
        payer = Keypair.fromSecretKey(secretKey);
        console.log('Loaded keypair:', payer.publicKey.toString());
    } catch (error) {
        console.error('Error loading keypair:', error);
        process.exit(1);
    }

    // Create the serialized proof data
    const serializedData = Buffer.concat([
        proofData.pi_a,
        proofData.pi_b,
        proofData.pi_c,
        proofData.publicSignalsBuffer
    ]);

    console.log('Serialized proof data length:', serializedData.length);
    console.log('Serialized proof data:', serializedData.toString('hex'));

    // Create transaction
    const transaction = new Transaction();

    // Add compute unit limit and price
    transaction.add(ComputeBudgetProgram.setComputeUnitLimit({ units: 1_400_000 }));
    transaction.add(ComputeBudgetProgram.setComputeUnitPrice({ microLamports: 2 }));

    // Create the instruction
    const instruction = new TransactionInstruction({
        keys: [
            {
                pubkey: payer.publicKey,
                isSigner: true,
                isWritable: true
            }
        ],
        programId: new PublicKey(programId),
        data: serializedData
    });

    // Add the instruction to the transaction
    transaction.add(instruction);

    // Sign and send transaction
    console.log('Sending transaction...');
    try {
        const signature = await sendAndConfirmTransaction(
            connection,
            transaction,
            [payer],
            {
                skipPreflight: true,
                commitment: 'confirmed'
            }
        );

        console.log('Transaction signature:', signature);

        // Get transaction logs
        const txLogs = await connection.getTransaction(signature, {
            commitment: 'confirmed',
            maxSupportedTransactionVersion: 0
        });
        console.log('Transaction logs:', txLogs?.meta?.logMessages);

    } catch (error) {
        console.error('❌ Error:', error);
        if (error.logs) {
            console.log('Detailed error logs:', error.logs);
        }
    }
}

async function main() {
    // Get program ID from command line
    const programId = process.argv[2];
    if (!programId) {
        console.error('Please provide the program ID as an argument');
        process.exit(1);
    }

    try {
        console.log('Generating and formatting proof...');
        const proofData = await generateAndFormatProof();

        console.log('Verifying proof on Solana...');
        await verifyProof(programId, proofData);
    } catch (error) {
        console.error('Error:', error);
        process.exit(1);
    }
}

main().catch(console.error); 