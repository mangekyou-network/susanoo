import { SolanaAgentKit, pythFetchPrice } from "solana-agent-kit";
import { PublicKey } from '@solana/web3.js';
import * as dotenv from 'dotenv';

dotenv.config();

// Constants
export const DEFAULT_SLIPPAGE_BPS = 100; // 1% default slippage

// Token addresses (mainnet-beta)
export const TOKENS = {
    SOL: new PublicKey("So11111111111111111111111111111111111111112"),
    USDC: new PublicKey("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
} as const;

export async function getBalance(
    agent: SolanaAgentKit,
    tokenMint?: PublicKey
): Promise<number | null> {
    try {
        const balance = await agent.getBalance(tokenMint);
        return balance;
    } catch (error) {
        console.error('Error getting balance:', error);
        return 0;
    }
}

export async function getPriceFromPyth(
    agent: SolanaAgentKit,
    symbol: string
): Promise<number> {
    try {
        const price = await pythFetchPrice(symbol);
        return price;
    } catch (error) {
        console.error('Error getting price from Pyth:', error);
        return 0;
    }
}

export async function swapTokens(
    agent: SolanaAgentKit,
    outputMint: PublicKey,
    inputAmount: number,
    inputMint: PublicKey,
    slippageBps: number = DEFAULT_SLIPPAGE_BPS
): Promise<string | null> {
    try {
        const signature = await agent.trade(
            outputMint,
            inputAmount,
            inputMint,
            slippageBps
        );
        return signature;
    } catch (error) {
        console.error('Error in swap_tokens:', error);
        return null;
    }
}

export async function swapSolToUsdc(
    agent: SolanaAgentKit,
    amount: number
): Promise<string | null> {
    return swapTokens(
        agent,
        TOKENS.USDC,
        amount,
        TOKENS.SOL
    );
}

export async function swapUsdcToSol(
    agent: SolanaAgentKit,
    amount: number
): Promise<string | null> {
    return swapTokens(
        agent,
        TOKENS.SOL,
        amount,
        TOKENS.USDC
    );
} 