import numpy as np
from scipy.stats import poisson

class ProbabilityEngine:
    """
    Handles converting score projections into win/loss probabilities.
    """

    @staticmethod
    def calculate_win_probability(home_expected_score: float, away_expected_score: float) -> float:
        """
        Uses a simplified Poisson-based approach to estimate the probability 
        of the home team winning.
        """
        # In a more complex model, we'd use a multivariate distribution 
        # or simulations (Monte Carlo). For now, a Poisson-based approach:
        # We'll simulate 10,000 games with these expected scores.
        
        size = 10000
        home_scores = np.random.poisson(home_expected_score, size)
        away_scores = np.random.poisson(away_expected_score, size)
        
        wins = np.sum(home_scores > away_scores)
        draws = np.sum(home_scores == away_scores) # In many sports, we handle draws separately
        
        # Simplified: win probability for home team
        return float(wins / size)

    @staticmethod
    def calculate_over_under_probability(home_expected: float, away_expected: float) -> float:
        """Calculates probability of total score being over a certain line."""
        # This is more complex in reality, but for now, 
        # let's just return a placeholder or basic logic.
        return 0.5 # Placeholder

if __name__ == "__main__":
    # Test the probability engine
    p_engine = ProbabilityEngine()
    prob = p_engine.calculate_win_probability(110, 100)
    print(f"Home Win Probability: {prob:.2%}")
