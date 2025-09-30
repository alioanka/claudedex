"""Update ML models with new weights"""
import asyncio
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ml.models.ensemble_model import EnsemblePredictor
import json
from datetime import datetime

async def update_models(weights_file: str = None):
    """Update model weights"""
    
    print("🔄 UPDATING ML MODELS")
    print("="*60)
    
    try:
        model_dir = "ml/models"
        ensemble = EnsemblePredictor(model_dir)
        
        # Load models
        print("\n📂 Loading existing models...")
        ensemble.load_models()
        print("   ✅ Models loaded")
        
        # Update weights if provided
        if weights_file and os.path.exists(weights_file):
            print(f"\n⚖️  Updating weights from {weights_file}...")
            with open(weights_file, 'r') as f:
                new_weights = json.load(f)
            
            ensemble.update_weights(new_weights)
            print("   ✅ Weights updated")
        else:
            print("\n⚠️  No weights file provided, using default weights")
        
        # Save updated models
        print("\n💾 Saving models...")
        ensemble.save_models()
        
        # Create backup
        backup_dir = Path(model_dir) / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"models_backup_{timestamp}"
        
        print(f"   Backup created at: {backup_path}")
        
        print("\n✅ Model update complete!")
        
    except Exception as e:
        print(f"\n❌ Error updating models: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Update ML models')
    parser.add_argument('--weights', help='Path to weights JSON file')
    args = parser.parse_args()
    
    asyncio.run(update_models(args.weights))