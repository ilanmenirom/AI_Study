import torch
import torch.nn as nn


class SharpeLoss(nn.Module):
    def __init__(
        self,
        target_volatility=0.1,
        cost_rate=0.001,
        eps=1e-14,
    ) -> None:
        """
        Computes the Sharpe ratio loss for portfolio optimization.
        If [target_volatility] is specified, it scales the weights to match the target volatility.
        If [cost_rate] is specified, it applies transaction costs based on the change in weights.
        """

        super(SharpeLoss, self).__init__()
        self.target_volatility = target_volatility
        self.cost_rate = cost_rate
        self.eps = eps

    def forward(self, weights, returns, volatility_estimate=None):
        # Ensure weights are non-negative and sum to 1 for each day
        # weights shape: [batch_size, T, num_stocks]
        # returns shape: [batch_size, T, num_stocks]

        # TODO: ensure this is not done in the model
        #  (I actually think it should be done in the model)
        weights = torch.softmax(weights, dim=-1)

        # Re-weight returns by the volatility estimate if required
        if self.target_volatility and (volatility_estimate is not None):
            weights = self.target_volatility * weights / (volatility_estimate + 1e-14)

        # Calculate portfolio returns for each timestep
        # Sum across stocks dimension: [batch_size, T]
        portfolio_returns = torch.sum(weights * returns, dim=-1)

        # Calculate transaction costs
        if self.cost_rate:
            # TODO: fix transaction costs of first day
            transaction_costs = self.cost_rate * torch.sum(
                torch.abs(weights[:, 1:, :] - weights[:, :-1, :]), dim=-1
            )
            transaction_costs = torch.cat(
                (transaction_costs[:, 0][:, None], transaction_costs), axis=1
            )

            # Subtract transaction costs from portfolio returns
            portfolio_returns -= transaction_costs

        # Calculate mean and std of portfolio returns across time dimension
        mean_returns = torch.mean(portfolio_returns, dim=1)  # [batch_size]
        std_returns = torch.std(portfolio_returns, dim=1)  # [batch_size]

        # Add epsilon to std to avoid division by zero
        if self.eps > min(std_returns) * 0.001:
            raise ValueError("Epsilon is too large, consider reducing it")

        std_returns = std_returns + self.eps

        # Calculate Sharpe ratio for each batch
        sharpe_ratio = mean_returns / std_returns  # [batch_size]

        # Return negative mean Sharpe ratio as we want to maximize it
        return -torch.mean(sharpe_ratio)
