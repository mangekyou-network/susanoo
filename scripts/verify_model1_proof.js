const { Connection, PublicKey, Keypair, Transaction, TransactionInstruction } = require('@solana/web3.js');
const fs = require('fs');

// Load model1's proof and inputs
const model1Input = require('../../models/model1_input.json');

async function verifyProof(programId) {
    // Connect to devnet
    const connection = new Connection('https://api.devnet.solana.com', 'confirmed');

    // Create a payer account
    const payer = Keypair.generate();

    // Request airdrop for testing
    console.log('Requesting airdrop...');
    const airdropSignature = await connection.requestAirdrop(
        payer.publicKey,
        1000000000 // 1 SOL
    );
    await connection.confirmTransaction(airdropSignature);

    // Format the proof data from model1_input.json
    const proofData = Buffer.concat([
        // Proof A
        Buffer.from(model1Input.Dense32weights[0][0].slice(2), 'hex'), // Remove '0x' prefix
        Buffer.from(model1Input.Dense32weights[0][1].slice(2), 'hex'),

        // Proof B (flatten the 2D array)
        ...model1Input.Dense32weights[1].map(x => Buffer.from(x.slice(2), 'hex')),

        // Proof C
        Buffer.from(model1Input.Dense32out[0].slice(2), 'hex'),
        Buffer.from(model1Input.Dense32out[1].slice(2), 'hex'),

        // Public inputs
        Buffer.from(model1Input.in[0].slice(2), 'hex')
    ]);

    // Create the instruction
    const instruction = new TransactionInstruction({
        keys: [],
        programId: new PublicKey(programId),
        data: proofData
    });

    // Create and send transaction
    const transaction = new Transaction().add(instruction);
    transaction.feePayer = payer.publicKey;

    console.log('Getting recent blockhash...');
    const { blockhash } = await connection.getLatestBlockhash();
    transaction.recentBlockhash = blockhash;

    // Sign and send transaction
    console.log('Sending transaction...');
    try {
        const signature = await connection.sendTransaction(transaction, [payer], {
            skipPreflight: false,
            preflightCommitment: 'confirmed'
        });

        console.log('Waiting for confirmation...');
        const confirmation = await connection.confirmTransaction(signature);

        if (confirmation.value.err) {
            console.error('❌ Transaction failed:', confirmation.value.err);
        } else {
            console.log('✅ Proof verified successfully!');
            console.log('Transaction signature:', signature);
        }

        // Get transaction logs
        const txLogs = await connection.getTransaction(signature, {
            commitment: 'confirmed',
            maxSupportedTransactionVersion: 0
        });
        console.log('Transaction logs:', txLogs?.meta?.logMessages);

    } catch (error) {
        console.error('❌ Error:', error);
    }
}

// Get program ID from command line
const programId = process.argv[2];
if (!programId) {
    console.error('Please provide the program ID as an argument');
    process.exit(1);
}

verifyProof(programId).catch(console.error); 