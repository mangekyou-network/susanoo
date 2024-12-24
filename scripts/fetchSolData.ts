import axios from 'axios';

interface SolData {
    Close: number;
    Volume: number;
    Date: string;
}

interface TechnicalIndicators {
    sma7: number;
    sma21: number;
    volumeMA5: number;
    volumeRatio: number;
    priceMomentum: number;
}

interface ProcessedData {
    features: number[][];
    latestFeatures: number[][];
    rawData: SolData[];
}

function calculateSMA(data: number[], window: number): number[] {
    const sma = [];
    for (let i = 0; i < data.length; i++) {
        if (i < window - 1) {
            sma.push(NaN);
            continue;
        }
        const sum = data.slice(i - window + 1, i + 1).reduce((a, b) => a + b, 0);
        sma.push(sum / window);
    }
    return sma;
}

function calculateTechnicalIndicators(data: SolData[]): TechnicalIndicators[] {
    const closes = data.map(d => d.Close);
    const volumes = data.map(d => d.Volume);

    const sma7 = calculateSMA(closes, 7);
    const sma21 = calculateSMA(closes, 21);
    const volumeMA5 = calculateSMA(volumes, 5);

    return data.map((d, i) => ({
        sma7: sma7[i],
        sma21: sma21[i],
        volumeMA5: volumeMA5[i],
        volumeRatio: volumeMA5[i] ? d.Volume / volumeMA5[i] : 1,
        priceMomentum: sma21[i] ? d.Close / sma21[i] : 1
    }));
}

export async function fetchSolData(startDate: string = '2023-06-01'): Promise<ProcessedData> {
    try {
        // Fetch SOL/USD data from CoinGecko
        const response = await axios.get(
            `https://api.coingecko.com/api/v3/coins/solana/market_chart/range`, {
            params: {
                vs_currency: 'usd',
                from: new Date(startDate).getTime() / 1000,
                to: Date.now() / 1000
            }
        }
        );

        // Process the data
        const prices = response.data.prices;
        const volumes = response.data.total_volumes;

        const solData: SolData[] = prices.map((price: number[], i: number) => ({
            Date: new Date(price[0]).toISOString(),
            Close: price[1],
            Volume: volumes[i][1]
        }));

        // Calculate technical indicators
        const indicators = calculateTechnicalIndicators(solData);

        // Create features array
        const features = solData.slice(21, -1).map((d, i) => {
            const idx = i + 21;
            return [
                d.Close,
                indicators[idx].volumeRatio,
                indicators[idx].priceMomentum
            ];
        });

        // Get latest features
        const latestFeatures = [[
            solData[solData.length - 1].Close,
            indicators[indicators.length - 1].volumeRatio,
            indicators[indicators.length - 1].priceMomentum
        ]];

        return {
            features,
            latestFeatures,
            rawData: solData
        };
    } catch (error) {
        console.error('Error fetching SOL data:', error);
        throw error;
    }
} 