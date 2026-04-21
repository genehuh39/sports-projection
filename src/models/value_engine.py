class ValueEngine:
    """Compares model probabilities against market prices."""

    @staticmethod
    def american_to_decimal(american_odds: float) -> float:
        """Converts American odds (e.g. -150, +200) to decimal odds."""
        if american_odds == 0:
            return 1.0
        if american_odds > 0:
            return (american_odds / 100) + 1
        return (100 / abs(american_odds)) + 1

    @staticmethod
    def decimal_to_implied_prob(decimal_odds: float) -> float:
        """Converts decimal odds to implied probability."""
        if decimal_odds == 0:
            return 0.0
        return 1.0 / decimal_odds

    def american_to_implied_prob(self, american_odds: float) -> float:
        """Converts American odds directly to implied probability."""
        return self.decimal_to_implied_prob(self.american_to_decimal(american_odds))

    @staticmethod
    def calculate_edge(model_win_prob: float, implied_prob: float) -> float:
        """Returns model edge over the market implied probability."""
        return model_win_prob - implied_prob

    def calculate_expected_value(self, model_win_prob: float, american_odds: float) -> float:
        """Calculates expected profit/loss per 1 unit staked."""
        decimal_odds = self.american_to_decimal(american_odds)
        profit = decimal_odds - 1.0
        loss = 1.0
        return (model_win_prob * profit) - ((1 - model_win_prob) * loss)

    def get_bet_recommendation(self, ev: float) -> str:
        """Returns a recommendation based on EV."""
        if ev > 0.05:
            return "STRONG VALUE"
        if ev > 0:
            return "SLIGHT VALUE"
        return "NO VALUE"


if __name__ == "__main__":
    ve = ValueEngine()
    ev_val = ve.calculate_expected_value(0.75, 150)
    print(f"EV for 75% win at +150: {ev_val:.2%}")
    print(f"Implied probability at +150: {ve.american_to_implied_prob(150):.2%}")
    print(f"Recommendation: {ve.get_bet_recommendation(ev_val)}")
