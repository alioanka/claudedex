# Key Files and Methods Report

## 1. Project Structure

The project is well-organized into the following key directories:

-   `config/`: Manages all configuration files and settings.
-   `core/`: Contains the central logic of the trading bot, including the trading engine, decision maker, and risk manager.
-   `dashboard/`: The Node.js-based web interface for monitoring the bot.
-   `data/`: Handles data collection, storage, and management.
-   `ml/`: Includes machine learning models for market analysis and prediction.
-   `trading/`: Contains the trading strategies, executors, and order management systems.

## 2. Core Components

### `main.py`

This is the main entry point of the application. It initializes the bot, loads the configuration, and starts the trading engine. The `TradingBotApplication` class is the central orchestrator, managing the bot's lifecycle and ensuring all components are properly initialized.

### `core/engine.py`

The `TradingBotEngine` class is the heart of the bot, responsible for:

-   **Managing the main trading loop.**
-   **Discovering new trading opportunities** by monitoring data from DexScreener.
-   **Analyzing opportunities** using a sophisticated scoring algorithm that considers liquidity, volume, momentum, and risk.
-   **Executing trades** in both simulated and live modes, with support for multiple chains.
-   **Monitoring active positions** and executing exit strategies based on predefined conditions.

### `config/config_manager.py`

The `ConfigManager` class provides a centralized system for managing all configuration settings. It supports:

-   **Schema validation** using Pydantic to ensure data integrity.
-   **Hot-reloading** of configuration files, allowing for dynamic updates without restarting the bot.
-   **Environment variable overrides** for easy deployment and customization.

## 3. Trading Logic

### `trading/strategies/base_strategy.py`

This file defines the `BaseStrategy` abstract class, which serves as the foundation for all trading strategies. It provides a clear interface for creating new strategies and includes built-in support for:

-   **Performance tracking.**
-   **Risk management.**
-   **Backtesting.**

### `core/decision_maker.py`

The `DecisionMaker` class is responsible for making the final trading decisions. It uses the opportunity score and other data points to determine whether to enter or exit a trade.

### `core/risk_manager.py`

The `RiskManager` class is responsible for managing risk at both the individual trade and portfolio levels. It includes features such as:

-   **Dynamic position sizing.**
-   **Stop-loss and take-profit orders.**
-   **Portfolio-level risk controls.**

## 4. Data Collectors

### `data/collectors/dexscreener.py`

This module is responsible for collecting data from the DexScreener API. It provides a reliable and efficient way to get real-time data on new pairs, prices, and volume.

### `data/collectors/chain_data.py`

This module is responsible for collecting on-chain data, such as token information and holder distributions. This data is used to enrich the opportunity scoring algorithm and provide a more complete picture of the market.

## 5. Dashboard

The `dashboard/` directory contains a Node.js-based web application that provides a user-friendly interface for monitoring the bot. The dashboard allows you to:

-   **View the bot's current status and performance.**
-   **Monitor active positions.**
-   **Review trade history.**
-   **Manage configuration settings.**
