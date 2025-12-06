"""
Auto ML Model Trainer for ClaudeDex Trading Bot

This module provides automatic training and retraining of ML models
for the AI trading strategy. It can be run:
1. Periodically (e.g., daily/weekly via cron)
2. On-demand via API call
3. After accumulating enough new trade data

Usage:
    # Run training from command line
    python -m ml.training.auto_trainer --train

    # Schedule periodic training (add to crontab):
    # 0 2 * * 0 cd /path/to/claudedex && python -m ml.training.auto_trainer --train
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# Configure logging
logging.basicConfig(level=logging.INFO)


class AutoMLTrainer:
    """
    Automatic ML Model Trainer for Trading Strategies

    Trains models on historical trade data to improve AI strategy predictions.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Training configuration
        self.min_trades_for_training = self.config.get('min_trades', 100)
        self.training_lookback_days = self.config.get('lookback_days', 30)
        self.model_save_dir = Path(self.config.get('model_dir', './models/ai_strategy'))
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # Database connection
        self.db_pool = None

        # Training metrics
        self.training_metrics = {
            'last_trained': None,
            'trades_used': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

    async def initialize(self):
        """Initialize database connection"""
        try:
            import asyncpg

            db_url = os.getenv('DATABASE_URL') or os.getenv('DB_URL')
            if db_url:
                self.db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
                logger.info("Database connection established for training")
            else:
                logger.warning("No DATABASE_URL found, will use synthetic data")

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")

    async def fetch_training_data(self) -> pd.DataFrame:
        """Fetch historical trade data for training"""
        if not self.db_pool:
            logger.warning("No database connection, generating synthetic data")
            return self._generate_synthetic_data()

        try:
            async with self.db_pool.acquire() as conn:
                # Fetch closed trades with their features
                cutoff_date = datetime.utcnow() - timedelta(days=self.training_lookback_days)

                rows = await conn.fetch("""
                    SELECT
                        t.trade_id,
                        t.symbol,
                        t.chain,
                        t.entry_price,
                        t.exit_price,
                        t.profit_loss,
                        t.side,
                        t.strategy,
                        t.timestamp as entry_time,
                        t.exit_timestamp as exit_time,
                        t.metadata
                    FROM trades t
                    WHERE t.status = 'closed'
                    AND t.timestamp >= $1
                    ORDER BY t.timestamp DESC
                    LIMIT 10000
                """, cutoff_date)

                if not rows:
                    logger.warning("No historical trades found, generating synthetic data")
                    return self._generate_synthetic_data()

                # Convert to DataFrame
                data = []
                for row in rows:
                    trade = dict(row)

                    # Parse metadata for additional features
                    metadata = {}
                    if trade.get('metadata'):
                        try:
                            if isinstance(trade['metadata'], str):
                                metadata = json.loads(trade['metadata'])
                            else:
                                metadata = trade['metadata']
                        except:
                            pass

                    data.append({
                        'trade_id': trade['trade_id'],
                        'symbol': trade['symbol'],
                        'chain': trade['chain'],
                        'entry_price': float(trade['entry_price'] or 0),
                        'exit_price': float(trade['exit_price'] or 0),
                        'pnl': float(trade['profit_loss'] or 0),
                        'is_win': float(trade['profit_loss'] or 0) > 0,
                        'side': trade['side'],
                        'strategy': trade['strategy'],
                        'entry_time': trade['entry_time'],
                        'exit_time': trade['exit_time'],
                        # Features from metadata
                        'pump_probability': metadata.get('pump_probability', 0.5),
                        'volume_24h': metadata.get('volume_24h', 0),
                        'liquidity': metadata.get('liquidity', 0),
                        'price_change_24h': metadata.get('price_change_24h', 0),
                        'holder_count': metadata.get('holder_count', 0),
                        'volatility': metadata.get('volatility', 0)
                    })

                df = pd.DataFrame(data)
                logger.info(f"Fetched {len(df)} trades for training")
                return df

        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate BALANCED synthetic training data when real data is insufficient"""
        logger.info(f"Generating {n_samples} synthetic training samples (50% balanced)")

        np.random.seed(42)

        # Generate balanced classes - 50% wins, 50% losses
        half = n_samples // 2

        # Generate WINNING trades with favorable characteristics
        wins_data = {
            'pump_probability': np.random.beta(5, 2, half),  # Higher pump prob for wins
            'volume_24h': np.random.exponential(200000, half),  # Higher volume
            'liquidity': np.random.exponential(100000, half),  # Higher liquidity
            'price_change_24h': np.random.normal(10, 15, half),  # Positive price change
            'holder_count': np.random.exponential(2000, half).astype(int),
            'volatility': np.abs(np.random.normal(0.08, 0.03, half)),  # Lower volatility
            'is_win': np.ones(half, dtype=int),
            'pnl': np.abs(np.random.normal(15, 8, half))
        }

        # Generate LOSING trades with unfavorable characteristics
        losses_data = {
            'pump_probability': np.random.beta(2, 5, half),  # Lower pump prob for losses
            'volume_24h': np.random.exponential(50000, half),  # Lower volume
            'liquidity': np.random.exponential(20000, half),  # Lower liquidity
            'price_change_24h': np.random.normal(-5, 20, half),  # Negative/neutral price change
            'holder_count': np.random.exponential(500, half).astype(int),
            'volatility': np.abs(np.random.normal(0.15, 0.08, half)),  # Higher volatility
            'is_win': np.zeros(half, dtype=int),
            'pnl': -np.abs(np.random.normal(10, 5, half))
        }

        # Combine and shuffle
        wins_df = pd.DataFrame(wins_data)
        losses_df = pd.DataFrame(losses_data)
        df = pd.concat([wins_df, losses_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

        # Add trade IDs and metadata
        df['trade_id'] = [f'synthetic_{i}' for i in range(len(df))]
        df['symbol'] = [f'TOKEN{i % 100}' for i in range(len(df))]
        df['chain'] = np.random.choice(['ETHEREUM', 'BSC', 'BASE', 'SOLANA'], len(df))
        df['entry_price'] = np.random.uniform(0.0001, 100, len(df))

        logger.info(f"Synthetic data: {df['is_win'].sum()} wins, {(df['is_win'] == 0).sum()} losses")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        # Feature columns
        feature_cols = [
            'pump_probability',
            'volume_24h',
            'liquidity',
            'price_change_24h',
            'holder_count',
            'volatility'
        ]

        # Fill missing values
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)

        X = df[feature_cols].values
        y = df['is_win'].astype(int).values

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save scaler for inference
        import joblib
        scaler_path = self.model_save_dir / 'feature_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved feature scaler to {scaler_path}")

        return X_scaled, y

    async def train_models(self) -> Dict:
        """Train all ML models"""
        logger.info("Starting ML model training...")

        # Fetch training data
        df = await self.fetch_training_data()

        if len(df) < self.min_trades_for_training:
            logger.warning(f"Insufficient data: {len(df)} trades (need {self.min_trades_for_training})")
            # Still train on synthetic data for testing

        # Prepare features
        X, y = self.prepare_features(df)

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        results = {}

        # Train XGBoost
        try:
            results['xgboost'] = await self._train_xgboost(X_train, y_train, X_test, y_test)
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            results['xgboost'] = {'error': str(e)}

        # Train LightGBM
        try:
            results['lightgbm'] = await self._train_lightgbm(X_train, y_train, X_test, y_test)
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            results['lightgbm'] = {'error': str(e)}

        # Train Random Forest (ensemble backup)
        try:
            results['random_forest'] = await self._train_random_forest(X_train, y_train, X_test, y_test)
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            results['random_forest'] = {'error': str(e)}

        # Update training metrics
        self.training_metrics['last_trained'] = datetime.utcnow().isoformat()
        self.training_metrics['trades_used'] = len(df)

        # Calculate overall metrics from all successful models
        successful_models = [r for r in results.values() if 'accuracy' in r]
        if successful_models:
            self.training_metrics['accuracy'] = np.mean([m['accuracy'] for m in successful_models])
            self.training_metrics['precision'] = np.mean([m['precision'] for m in successful_models])
            self.training_metrics['recall'] = np.mean([m['recall'] for m in successful_models])
            self.training_metrics['f1_score'] = np.mean([m['f1_score'] for m in successful_models])

        # Save training report
        await self._save_training_report(results)

        logger.info("=" * 60)
        logger.info("🎯 Training Complete! Final Metrics:")
        logger.info(f"   Accuracy:  {self.training_metrics.get('accuracy', 0):.4f}")
        logger.info(f"   Precision: {self.training_metrics.get('precision', 0):.4f}")
        logger.info(f"   Recall:    {self.training_metrics.get('recall', 0):.4f}")
        logger.info(f"   F1 Score:  {self.training_metrics.get('f1_score', 0):.4f}")
        logger.info("=" * 60)
        return results

    async def _train_xgboost(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train XGBoost model with class imbalance handling"""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not installed, skipping")
            return {'error': 'XGBoost not installed'}

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Calculate scale_pos_weight to handle class imbalance
        # This gives more weight to the minority class (winning trades)
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(f"XGBoost: Training with scale_pos_weight={scale_pos_weight:.2f} (neg={n_neg}, pos={n_pos})")

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,  # Handle class imbalance
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric='auc'  # AUC is better for imbalanced data
        )

        model.fit(X_train, y_train)

        # Get probability predictions for threshold tuning
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Use a lower threshold for predicting positive class (default is 0.5)
        # This helps catch more winning trades at the cost of some false positives
        threshold = 0.4
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'threshold': threshold,
            'scale_pos_weight': scale_pos_weight
        }

        # Save model
        model_path = self.model_save_dir / 'xgboost_model.json'
        model.save_model(str(model_path))
        logger.info(f"Saved XGBoost model to {model_path}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")

        return metrics

    async def _train_lightgbm(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train LightGBM model with class imbalance handling"""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not installed, skipping")
            return {'error': 'LightGBM not installed'}

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Calculate class weights for imbalance
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(f"LightGBM: Training with scale_pos_weight={scale_pos_weight:.2f}")

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            objective='binary',
            scale_pos_weight=scale_pos_weight,  # Handle class imbalance
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )

        model.fit(X_train, y_train)

        # Get probability predictions and use threshold
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        threshold = 0.4
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'threshold': threshold
        }

        # Save model
        model_path = self.model_save_dir / 'lightgbm_model.txt'
        model.booster_.save_model(str(model_path))
        logger.info(f"Saved LightGBM model: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")

        return metrics

    async def _train_random_forest(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Random Forest model with class imbalance handling"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import joblib

        logger.info("Random Forest: Training with class_weight='balanced'")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle class imbalance automatically
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Use probability predictions with threshold
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        threshold = 0.4
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'threshold': threshold
        }

        # Save model
        model_path = self.model_save_dir / 'random_forest_model.joblib'
        joblib.dump(model, model_path)
        logger.info(f"Saved Random Forest: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")

        return metrics

    async def _save_training_report(self, results: Dict):
        """Save training report to file"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': self.training_metrics,
            'model_results': results,
            'config': {
                'min_trades': self.min_trades_for_training,
                'lookback_days': self.training_lookback_days
            }
        }

        report_path = self.model_save_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Saved training report to {report_path}")

    async def close(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()


async def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Auto ML Model Trainer')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--min-trades', type=int, default=100, help='Minimum trades for training')
    parser.add_argument('--lookback-days', type=int, default=30, help='Days of data to use')
    args = parser.parse_args()

    if not args.train:
        parser.print_help()
        return

    config = {
        'min_trades': args.min_trades,
        'lookback_days': args.lookback_days
    }

    trainer = AutoMLTrainer(config)

    try:
        await trainer.initialize()
        results = await trainer.train_models()

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Trades used: {trainer.training_metrics['trades_used']}")
        print(f"Overall accuracy: {trainer.training_metrics['accuracy']:.2%}")
        print(f"Overall F1 score: {trainer.training_metrics['f1_score']:.2%}")
        print("\nModel Results:")
        for model_name, metrics in results.items():
            if 'accuracy' in metrics:
                print(f"  {model_name}: accuracy={metrics['accuracy']:.2%}, f1={metrics['f1_score']:.2%}")
            else:
                print(f"  {model_name}: {metrics.get('error', 'unknown error')}")
        print("="*60)

    finally:
        await trainer.close()


if __name__ == '__main__':
    asyncio.run(main())
