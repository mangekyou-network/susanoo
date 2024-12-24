import argparse
import logging
import os
import asyncio
from datetime import date, timedelta
import base58
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import MinMaxScaler
from solana.rpc.async_api import AsyncClient
from solana.keypair import Keypair

from solana_helpers import (
    get_balance,
    get_price_from_pyth,
    swap_sol_to_usdc,
    swap_usdc_to_sol,
    TOKENS
)

# Import our custom model
from models.model1_price_training import Model1, fetch_sol_data

load_dotenv(find_dotenv())

# Load environment variables
PRIVATE_KEY = os.environ.get("SOLANA_PRIVATE_KEY")
RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_logger")

def create_client():
    """Create Solana client and wallet"""
    # Create Solana client
    client = AsyncClient(RPC_URL)
    
    # Create wallet from private key
    private_key_bytes = base58.b58decode(PRIVATE_KEY)
    wallet = Keypair.from_secret_key(private_key_bytes)
    
    return client, wallet

def load_trained_model():
    """Load and return the trained model"""
    model = Model1()
    try:
        model.load_state_dict(torch.load('models/sol_price_model.pth'))
        model.eval()
        return model
    except:
        logger.error("Failed to load trained model. Please run model1_price_training.py first.")
        raise

async def execute_trading_strategy():
    """Execute the trading strategy based on the prediction"""
    logger.info("Starting the trading strategy")
    
    # Create client and wallet
    client, wallet = create_client()

    try:
        # Fetch latest data and prepare features
        features, _, _ = fetch_sol_data(start_date=(date.today() - timedelta(days=60)).strftime("%Y-%m-%d"))
        latest_features = features[-1:]  # Get the most recent data point
        
        # Scale features
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = feature_scaler.fit_transform(features)
        latest_scaled = feature_scaler.transform(latest_features)
        
        # Load and use our trained model
        model = load_trained_model()
        
        # Make prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(latest_scaled)
            prediction = model(input_tensor)
            predicted_price = prediction.item()
        
        logger.info(f"Predicted SOL price: ${predicted_price:.2f}")

        # Get current SOL price from Pyth oracle
        current_sol_price = await get_price_from_pyth("SOL/USD")
        logger.info(f"Current SOL price from Pyth: ${current_sol_price:.2f}")

        # Execute trading logic
        if predicted_price > current_sol_price * 1.01:
            # Check USDC balance
            usdc_balance = await get_balance(client, wallet, TOKENS["USDC"])
            logger.info(f"Your USDC balance: {usdc_balance:.2f}")
            
            if usdc_balance > 0.2:
                logger.info("Buying SOL with 0.1 USDC...")
                tx_hash = await swap_usdc_to_sol(client, wallet, 0.1)
                if tx_hash:
                    logger.info(f"Transaction hash: {tx_hash}")
                else:
                    logger.info("Transaction failed")
            else:
                logger.info("Insufficient USDC balance to buy SOL")
            
        elif predicted_price < current_sol_price * 0.99:
            # Check SOL balance
            sol_balance = await get_balance(client, wallet)
            logger.info(f"SOL balance: {sol_balance:.9f}")
            
            if sol_balance > 0:
                logger.info("Selling SOL for USDC...")
                tx_hash = await swap_sol_to_usdc(client, wallet, sol_balance)
                if tx_hash:
                    logger.info(f"Transaction hash: {tx_hash}")
                else:
                    logger.info("Transaction failed")
            else:
                logger.info("No SOL balance to sell")
        else:
            logger.info("No trading action needed at this time.")

        # Print additional technical indicators
        logger.info("\nTechnical Indicators:")
        logger.info(f"Volume Ratio: {latest_features[0][1]:.2f}")
        logger.info(f"Price Momentum: {latest_features[0][2]:.2f}")

    except Exception as e:
        logger.error(f"Error during trading strategy execution: {str(e)}")
    finally:
        await client.close()
    
    logger.info("-------Agent Execution Finished---------")

if __name__ == "__main__":
    asyncio.run(execute_trading_strategy()) 