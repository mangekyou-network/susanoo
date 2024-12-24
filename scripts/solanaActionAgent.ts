import { SolanaAgentKit } from 'solana-agent-kit';
import * as ort from 'onnxruntime-web';
import * as dotenv from 'dotenv';
import {
    getBalance,
    getPriceFromPyth,
    swapSolToUsdc,
    swapUsdcToSol,
    TOKENS
} from './solanaHelpers';
import { fetchSolData } from './fetchSolData';

dotenv.config();

// Load environment variables
const PRIVATE_KEY = process.env.SOLANA_PRIVATE_KEY;
const RPC_URL = process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// Simple min-max scaling function
function minMaxScale(data: number[][], featureRange: [number, number] = [0, 1]): number[][] {
    const numFeatures = data[0].length;
    const mins: number[] = new Array(numFeatures).fill(Infinity);
    const maxs: number[] = new Array(numFeatures).fill(-Infinity);

    // Find min and max for each feature
    for (const row of data) {
        for (let i = 0; i < numFeatures; i++) {
            mins[i] = Math.min(mins[i], row[i]);
            maxs[i] = Math.max(maxs[i], row[i]);
        }
    }

    // Scale the data
    return data.map(row =>
        row.map((val, i) => {
            const scaled = (val - mins[i]) / (maxs[i] - mins[i]);
            return scaled * (featureRange[1] - featureRange[0]) + featureRange[0];
        })
    );
}

async function createAgent(): Promise<SolanaAgentKit> {
    if (!PRIVATE_KEY || !OPENAI_API_KEY) {
        throw new Error('Missing required environment variables: SOLANA_PRIVATE_KEY or OPENAI_API_KEY');
    }

    // Create agent with OpenAI API key
    const agent = new SolanaAgentKit(
        PRIVATE_KEY,
        RPC_URL,
        OPENAI_API_KEY
    );

    return agent;
}

async function loadModel(): Promise<ort.InferenceSession> {
    try {
        // Create inference session
        const session = await ort.InferenceSession.create('models/sol_price_model.onnx');
        return session;
    } catch (error) {
        console.error('Failed to load ONNX model:', error);
        throw error;
    }
}

async function executeTradingStrategy() {
    console.log('Starting the trading strategy');

    try {
        // Create agent
        const agent = await createAgent();

        // Fetch latest data and prepare features
        const { features, latestFeatures } = await fetchSolData();

        // Scale features
        const allFeatures = [...features, ...latestFeatures];
        const scaledFeatures = minMaxScale(allFeatures);
        const latestScaled = [scaledFeatures[scaledFeatures.length - 1]];

        // Load model
        const session = await loadModel();

        // Create input tensor
        const inputTensor = new ort.Tensor('float32', new Float32Array(latestScaled.flat()), [1, latestScaled[0].length]);

        // Run inference
        const outputMap = await session.run({ input: inputTensor });
        const predictedPrice = outputMap.output.data[0] as number;

        console.log(`Predicted SOL price: $${predictedPrice.toFixed(2)}`);

        // Get current SOL price from Pyth oracle
        const currentSolPrice = await getPriceFromPyth(agent, 'SOL/USD');
        console.log(`Current SOL price from Pyth: $${currentSolPrice.toFixed(2)}`);

        // Execute trading logic
        if (predictedPrice > currentSolPrice * 1.01) {
            // Check USDC balance
            const usdcBalance = await getBalance(agent, TOKENS.USDC);
            console.log(`Your USDC balance: ${usdcBalance?.toFixed(2)}`);

            if (usdcBalance && usdcBalance > 0.2) {
                console.log('Buying SOL with 0.1 USDC...');
                const txHash = await swapUsdcToSol(agent, 0.1);
                if (txHash) {
                    console.log(`Transaction hash: ${txHash}`);
                } else {
                    console.log('Transaction failed');
                }
            } else {
                console.log('Insufficient USDC balance to buy SOL');
            }
        } else if (predictedPrice < currentSolPrice * 0.99) {
            // Check SOL balance
            const solBalance = await getBalance(agent, TOKENS.SOL);
            console.log(`SOL balance: ${solBalance?.toFixed(9)}`);

            if (solBalance && solBalance > 0) {
                console.log('Selling SOL for USDC...');
                const txHash = await swapSolToUsdc(agent, solBalance);
                if (txHash) {
                    console.log(`Transaction hash: ${txHash}`);
                } else {
                    console.log('Transaction failed');
                }
            } else {
                console.log('No SOL balance to sell');
            }
        } else {
            console.log('No trading action needed at this time.');
        }

        // Print additional technical indicators
        console.log('\nTechnical Indicators:');
        console.log(`Volume Ratio: ${latestFeatures[0][1].toFixed(2)}`);
        console.log(`Price Momentum: ${latestFeatures[0][2].toFixed(2)}`);

    } catch (error) {
        console.error('Error during trading strategy execution:', error);
    }

    console.log('-------Agent Execution Finished---------');
}

// Run the trading strategy
executeTradingStrategy().catch(console.error); 