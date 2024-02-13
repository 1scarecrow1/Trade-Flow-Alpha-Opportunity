### Can trade flow predict returns?

In this notebook, we assess trade flow as a means of generating profit opportunity in 3 crypto token markets. Specifically, we compute trade flow in τ-second intervals and regress T-second forward returns on the trade flow to establish a measure of return predictability. Based on this measure, we perform a backtest to identify trade opportunities and assess how a trade may have performed.

Trade flow is a central idea in this analysis. It is a running tally of signed trade sizes where the sign is defined as 1 if the trade was seller-initiated and -1 if it was buyer-initiated. At any moment, we examine all reported trades within the last time period of length τ. 
  
This can be normalized or transformed in many ways, with the goal of achieving more predictive forms. The essential idea behind flow as a quantitative metric is that, in circumstances when many sellers are willing to cross that market-making bid-offer spread to complete their transactions, there is likely to be new information driving their choices. We do not know exactly what it is, but we certainly want to adapt to it. 
