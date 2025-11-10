import json

def analyze_trades(log_file):
    wins = 0
    losses = 0
    total_pnl = 0.0
    trade_entries = {}
    trade_exits = []

    with open(log_file, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                if log_entry.get("type") == "TRADE_ENTRY":
                    trade_entries[log_entry["trade_id"]] = log_entry
                elif log_entry.get("type") == "TRADE_EXIT":
                    pnl = log_entry.get("pnl_usd", 0.0)
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    total_pnl += pnl
                    trade_exits.append(log_entry)

            except json.JSONDecodeError:
                # a few lines are not json, but trade entry/exit summaries, like:
                # 🟢 TRADE_ENTRY | SOLANA | SAFESPACE (BAsWv39p...) | Entry: $0.0003 | Amount: 21483.79 | Size: $7.22
                # I can ignore these for now as the json line below it has the same info.
                pass


    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    print(f"Total trades: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total P&L: ${total_pnl:.2f}")

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
    }

if __name__ == "__main__":
    analyze_trades("logs/TradingBot_trades.log")
