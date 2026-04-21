from typing import List, Dict
import numpy as np
from src.models.schemas_v2 import PlayerProjection

class PlayerPropEngine:
    """
    Specialized engine for generating and evaluating individual player prop bets.
    """

    def __init__(self, simulations_per_player: int = 5000):
        self.sims = simulations_per_player

    def simulate_prop(self, player_stats: dict, prop_type: str, minutes_available: float) -> Dict[str, float]:
        """
        Simulates a specific player prop outcome (e.g., 'points', 'rebounds') 
        using a Monte Carlo approach.
        """
        # Base rate from historical stats
        stats_map = {
            'points': player_stats.get('ppg', 20.0),
            'rebounds': player_stats.get('rpg', 5.0),
            'assists': player_stats.get('apg', 5.0)
        }
        
        base_rate = stats_map.get(prop_type, 10.0)
        scale_factor = minutes_available / player_stats.get('min', 36)
        expected_val = base_rate * scale_factor

        # Simulate 'n' outcomes for this specific prop
        simulated_outcomes = np.random.poisson(expected_val, self.sims)
        
        # Calculate metrics from simulation
        return {
            "expected_value": float(np.mean(simulated_outcomes)),
            "prob_over": float(np.sum(simulated_outcomes > expected_val) / self.sims), # Placeholder logic
            "all_outcomes": simulated_outcomes
        }

    def evaluate_line(self, player_stats: dict, prop_type: str, line: float, minutes: float) -> dict:
        """
        Compares the simulation results against a specific betting line.
        Example: Line is 24.5, we calculate the probability of 'Over'.
        """
        # Run simulation
        base_rate = {
            'points': player_stats.get('ppg', 20.0),
            'rebounds': player_stats.get('rpg', 5.0),
            'assists': player_stats.get('apg', 5.0)
        }.get(prop_type, 10.0)
        
        scale_factor = minutes / player_stats.get('min', 36)
        expected_val = base_rate * scale_factor
        
        # The core of the bet: How often does the player go OVER the line?
        simulated_outcomes = np.random.poisson(expected_val, self.sims)
        prob_over = float(np.sum(simulated_outcomes > line) / self.sims)
        
        # Calculate EV (Assuming a $100 bet on 'Over')
        # If the line is 24.5, and we win if they score 25+
        # We use decimal odds to calculate EV. Let's assume standard -110 juice for the line.
        decimal_odds = 1.91 # Standard -110 odds
        profit = decimal_odds - 1.0
        ev = (prob_over * profit) - ((1 - prob_over) * 1.0)

        return {
            "prop": prop_type,
            "line": line,
            "prob_over": prob_over,
            "expected_value": ev,
            "recommendation": "BUY OVER" if ev > 0.05 else ("BUY UNDER" if ev < -0.05 else "PASS")
        }

if __name__ == "__main__":
    # Test Scenario: A star player with 25 PPG vs a line of 22.5
    engine = PlayerPropEngine(simulations_per_player=10000)
    star_player = {"id": "p1", "name": "Star", "ppg": 25.0, "rpg": 5.0, "apg": 5.0, "min": 36}
    
    line = 22.5
    result = engine.evaluate_line(star_player, "points", line, 36)
    
    print(f"Prop: {result['prop']} | Line: {result['line']}")
    print(f"Probability of Over: {result['prob_over']:.2%}")
    print(f"Expected Value (EV): {result['expected_value']:.2%}")
    print(f"Action: {result['recommendation']}")
