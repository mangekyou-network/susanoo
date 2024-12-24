import os
import requests
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction, VersionedTransaction
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.rpc.commitment import Confirmed
from spl.token.instructions import get_associated_token_address
import json
from base64 import b64decode, b64encode
from typing import Optional, Dict, Any
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Constants from solana-agent-kit
JUPITER_API_URL = "https://quote-api.jup.ag/v6"
PYTH_API_URL = "https://hermes.pyth.network/api/latest_price_feeds"
LAMPORTS_PER_SOL = 1_000_000_000
DEFAULT_SLIPPAGE_BPS = 300  # 3% default slippage

# Token addresses (mainnet-beta)
TOKENS = {
    "SOL": PublicKey("So11111111111111111111111111111111111111112"),
    "USDC": PublicKey("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
}

async def get_balance(client: AsyncClient, wallet: Keypair, token_address: Optional[str] = None) -> float:
    """
    Get the balance of SOL or SPL token for a wallet, using solana-agent-kit's implementation
    """
    try:
        if token_address is None:  # Get SOL balance
            balance = await client.get_balance(wallet.public_key, commitment=Confirmed)
            return float(balance['result']['value']) / LAMPORTS_PER_SOL
        else:  # Get SPL token balance
            token_pubkey = PublicKey(token_address)
            associated_token_address = get_associated_token_address(wallet.public_key, token_pubkey)
            balance = await client.get_token_account_balance(associated_token_address)
            return float(balance['result']['value']['uiAmount'])
    except Exception as e:
        print(f"Error getting balance: {str(e)}")
        return 0.0

async def get_price_from_pyth(symbol: str) -> float:
    """
    Get price from Pyth oracle, using solana-agent-kit's implementation
    """
    try:
        # Following pyth_fetch_price.ts implementation
        response = requests.get(
            f"{PYTH_API_URL}?ids[]={get_pyth_price_feed_id(symbol)}"
        )
        price_feed = response.json()[0]
        
        # Extract price and confidence interval
        price = float(price_feed['price']['price'])
        confidence = float(price_feed['price']['conf'])
        
        print(f"Price confidence: ±${confidence:.4f}")
        return price
    except Exception as e:
        print(f"Error getting price from Pyth: {str(e)}")
        return 0.0

def get_pyth_price_feed_id(symbol: str) -> str:
    """Get Pyth price feed ID for a given symbol"""
    # Common Pyth price feed IDs
    PRICE_FEEDS = {
        "SOL/USD": "H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG",
        "USDC/USD": "Gnt27xtC473ZT2Mw5u8wZ68Z3gULkSTb5DuxJy7eJotD"
    }
    return PRICE_FEEDS.get(symbol, "")

async def get_jupiter_quote(
    input_mint: PublicKey,
    output_mint: PublicKey,
    amount: float,
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS
) -> Dict[str, Any]:
    """Get Jupiter quote following solana-agent-kit implementation"""
    quote_url = f"{JUPITER_API_URL}/quote"
    params = {
        "inputMint": str(input_mint),
        "outputMint": str(output_mint),
        "amount": int(amount * LAMPORTS_PER_SOL),
        "slippageBps": slippage_bps,
        "onlyDirectRoutes": True,
        "maxAccounts": 20
    }
    
    response = requests.get(quote_url, params=params)
    return response.json()

async def swap_tokens(
    client: AsyncClient,
    wallet: Keypair,
    input_token: PublicKey,
    output_token: PublicKey,
    amount: float,
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS
) -> Optional[str]:
    """
    Swap tokens using Jupiter Exchange, following solana-agent-kit's implementation
    """
    try:
        # Get quote
        quote_response = await get_jupiter_quote(input_token, output_token, amount, slippage_bps)
        
        # Get swap transaction
        swap_url = f"{JUPITER_API_URL}/swap"
        swap_data = {
            "quoteResponse": quote_response,
            "userPublicKey": str(wallet.public_key),
            "wrapAndUnwrapSol": True,
            "dynamicComputeUnitLimit": True,
            "prioritizationFeeLamports": "auto"
        }
        
        swap_response = requests.post(swap_url, json=swap_data).json()
        
        # Deserialize and sign transaction
        tx_data = b64decode(swap_response['swapTransaction'])
        transaction = VersionedTransaction.deserialize(tx_data)
        transaction.sign([wallet])
        
        # Send transaction
        encoded_tx = b64encode(transaction.serialize()).decode('utf-8')
        result = await client.send_raw_transaction(
            encoded_tx,
            opts={"skip_preflight": True, "max_retries": 3}
        )
        
        return result['result']
    except Exception as e:
        print(f"Error in swap_tokens: {str(e)}")
        return None

async def swap_sol_to_usdc(client: AsyncClient, wallet: Keypair, amount: float) -> Optional[str]:
    """
    Swap SOL to USDC using Jupiter
    """
    return await swap_tokens(
        client,
        wallet,
        TOKENS["SOL"],
        TOKENS["USDC"],
        amount
    )

async def swap_usdc_to_sol(client: AsyncClient, wallet: Keypair, amount: float) -> Optional[str]:
    """
    Swap USDC to SOL using Jupiter
    """
    return await swap_tokens(
        client,
        wallet,
        TOKENS["USDC"],
        TOKENS["SOL"],
        amount
    )