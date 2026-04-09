***References:***

- `kalshi-fee-schedule.pdf` (in root directory)
- https://docs.kalshi.com/getting_started/orderbook_responses
- https://help.kalshi.com/en/articles/13823828-the-orderbook
- https://jdsemrau.substack.com/p/automated-market-making-on-kalshi
- https://www.switchmarkets.com/learn/price-slippage
- https://defirate.com/prediction-markets/fees/
- https://news.kalshi.com/p/how-to-read-probabilities


***Conversation with Gemini about Kalshi API (how "rate-limit tiers" might play a role in our position limit) + reference links:***

No, accessing and using the Kalshi API is free of charge. There are no monthly subscription fees or separate "access costs" for standard users to view market data or place trades via the API. [1, 2, 3, 4, 5]
Instead of a flat fee, your costs are tied to your activity on the platform: [6]
## 1. No Membership or Access Fees
Kalshi does not charge membership fees. All users get access to REST, WebSocket, and FIX protocols. [1, 7]

* Public Data: You can even access some [public endpoints](https://docs.kalshi.com/getting_started/quick_start_market_data) for market data without needing an API key.
* Sandbox: There is a [Demo environment](https://docs.kalshi.com/welcome) available for free to test your trading bots without using real money. [8, 9, 10]

## 2. Standard Trading Fees
While the API itself is free, every trade you place through it is subject to the platform’s standard [Fee Schedule](https://kalshi.com/fee-schedule). [3, 11]

* Taker Fees: Applied when your order matches immediately with an existing order. The formula is generally $\text{round up}(0.07 \times \text{Contracts} \times P \times (1-P))$.
* Maker Fees: Often lower than taker fees, these apply when your order sits on the order book before being filled. [12]

## 3. Rate Limit Tiers (Performance-Based)
Kalshi uses a tiered system for API [rate limits](https://docs.kalshi.com/getting_started/rate_limits). Rather than paying cash for more speed, you "earn" higher limits through trading volume: [13, 14]

* Basic: Free upon signup (20 read / 10 write requests per second).
* Premier: Requires 3.75% of exchange traded volume in a given month.
* Prime: Requires 7.5% of exchange traded volume in a given month. [15, 16, 17, 18]

## 4. Non-API Costs

* Deposits/Withdrawals: ACH and bank transfers are free, but [debit card deposits](https://help.kalshi.com/en/articles/13823805-fees) incur a 2% fee.
* Third-Party Tools: Some third-party wrappers or services like [RapidAPI](https://rapidapi.com/kalshi-kalshi-default/api/kalshi-trading-api/pricing) may charge their own platform fees if you choose to use them instead of Kalshi's direct API. [3, 19, 20, 21]

Are you planning to build a trading bot, or do you just need to pull historical market data for analysis?

[1] [https://docs.kalshi.com](https://docs.kalshi.com/welcome)
[2] [https://www.quicknode.com](https://www.quicknode.com/builders-guide/tools/kalshi-api-by-kalshi-inc?category=trading-tools#:~:text=Want%20to%20stay%20updated?%20Our%20newsletter%20is,resources%2C%20Quicknode%20updates%2C%20Web3%20insights%2C%20and%20more.)
[3] [https://rapidapi.com](https://rapidapi.com/kalshi-kalshi-default/api/kalshi-trading-api/pricing)
[4] [https://news.kalshi.com](https://news.kalshi.com/p/apiv2-is-here)
[5] [https://zuplo.com](https://zuplo.com/learning-center/kalshi-api)
[6] [https://deadspin.com](https://deadspin.com/prediction-markets/kalshi/fees/)
[7] [https://kalshi.com](https://kalshi.com/docs/kalshi-fee-schedule.pdf)
[8] [https://docs.kalshi.com](https://docs.kalshi.com/welcome)
[9] [https://docs.kalshi.com](https://docs.kalshi.com/getting_started/quick_start_market_data)
[10] [https://docs.kalshi.com](https://docs.kalshi.com/getting_started/making_your_first_request)
[11] [https://kalshi.com](https://kalshi.com/fee-schedule)
[12] [https://kalshi.com](https://kalshi.com/docs/kalshi-fee-schedule.pdf)
[13] [https://docs.kalshi.com](https://docs.kalshi.com/getting_started/rate_limits)
[14] [https://rapidapi.com](https://rapidapi.com/kalshi-kalshi-default/api/kalshi-trading-api/pricing)
[15] [https://docs.kalshi.com](https://docs.kalshi.com/getting_started/rate_limits)
[16] [https://www.cftc.gov](https://www.cftc.gov/sites/default/files/filings/orgrules/22/09/rule091222kexdcm003.pdf)
[17] [https://docs.kalshi.com](https://docs.kalshi.com/getting_started/rate_limits#:~:text=Qualification%20for%20tiers:%20*%20Basic:%20Completing%20signup.,exchange%20traded%20volume%20in%20a%20given%20month.)
[18] [https://docs.kalshi.com](https://docs.kalshi.com/getting_started/rate_limits#:~:text=Rate%20Limits%20and%20Tiers%20*%20Basic:%20Completing,exchange%20traded%20volume%20in%20a%20given%20month.)
[19] [https://help.kalshi.com](https://help.kalshi.com/en/articles/13823805-fees)
[20] [https://help.kalshi.com](https://help.kalshi.com/en/articles/13823805-fees)
[21] [https://rapidapi.com](https://rapidapi.com/kalshi-kalshi-default/api/kalshi-trading-api/pricing)
