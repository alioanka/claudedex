import json
from collections import defaultdict
import re

def analyze_main_log(log_file):
    """
    Analyzes the main TradingBot log file to extract insights.
    """
    stats = defaultdict(int)
    safety_check_failures = defaultdict(int)
    cooldowns = defaultdict(int)
    volatility = []

    print(f"Analyzing log file: {log_file}")

    with open(log_file, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                message = log_entry.get("message", "")

                if "OPPORTUNITY FOUND" in message:
                    stats['opportunities_found'] += 1
                elif "Safety checks failed" in message:
                    stats['safety_checks_failed'] += 1
                    # Example: "   \u274c Safety checks failed for 819T1na5X5"
                    token_match = re.search(r'failed for (\w+)', message)
                    if token_match:
                        # Could be improved to parse the specific failure reason from other log lines
                        # For now, just count failures per token symbol if available
                        token_symbol = token_match.group(1)
                        safety_check_failures[token_symbol] += 1
                elif "HONEYPOT DETECTED" in message:
                    stats['honeypots_detected'] += 1
                elif "EXCESSIVE VOLATILITY" in message:
                    stats['excessive_volatility_warnings'] += 1
                    # Example: "   \u274c EXCESSIVE VOLATILITY: +65.9% in 5min"
                    vol_match = re.search(r'([+-]?\d+\.\d+)%', message)
                    if vol_match:
                        volatility.append(float(vol_match.group(1)))
                elif "COOLDOWN ACTIVE" in message:
                    stats['cooldown_active_warnings'] += 1
                    # Example: "\u2744\ufe0f COOLDOWN ACTIVE for SAFESPACE: closed 3.0min ago (reason: stop_loss), 57.0min remaining"
                    reason_match = re.search(r'reason: (\w+)', message)
                    if reason_match:
                        reason = reason_match.group(1)
                        cooldowns[reason] += 1
                elif "No pairs found on" in message:
                    stats['no_pairs_found'] += 1
                elif "Missing score components" in message:
                    stats['missing_score_components'] += 1
                elif "RUGSCREENER FAILED" in message:
                    stats['rugscreener_failed'] += 1
                elif "RugCheck unavailable" in message:
                    stats['rugcheck_unavailable'] += 1
                elif "AI Strategy Signal" in message:
                    stats['ai_strategy_signals'] += 1
                elif "Momentum Signal" in message:
                    stats['momentum_strategy_signals'] += 1
                elif "TREND Signal" in message:
                    stats['trend_strategy_signals'] += 1

            except json.JSONDecodeError:
                pass  # Ignore non-JSON lines

    print("\n--- Main Log Analysis Results ---")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("\n--- Cooldown Reasons ---")
    for reason, count in cooldowns.items():
        print(f"{reason.replace('_', ' ').title()}: {count}")

    if volatility:
        print("\n--- Volatility Analysis ---")
        print(f"Average Volatility Warning: {sum(volatility) / len(volatility):.2f}%")
        print(f"Max Volatility Warning: {max(volatility):.2f}%")


if __name__ == "__main__":
    analyze_main_log("TradingBot.log")
    analyze_main_log("TradingBot.log.1")
    analyze_main_log("TradingBot.log.2")
