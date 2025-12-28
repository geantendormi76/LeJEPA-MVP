
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent))

def main():
    parser = argparse.ArgumentParser(description="LeJEPA MVP Launcher")
    parser.add_argument('app', choices=['atari', 'galaxy'], help="Choose application to run")
    parser.add_argument('--mode', choices=['train', 'verify', 'demo', 'vis'], default='train', help="Execution mode")
    
    args = parser.parse_args()
    
    print(f"🚀 Launching: {args.app.upper()} [{args.mode.upper()}]")
    
    if args.app == 'atari':
        from src.apps.atari_world_model import train, verify, demo_gif, visualize_prediction
        if args.mode == 'train':
            train.train()
        elif args.mode == 'verify':
            verify.verify()
        elif args.mode == 'demo':
            demo_gif.generate_demo()
        elif args.mode == 'vis':
            visualize_prediction.visualize()
            
    elif args.app == 'galaxy':
        from src.apps.galaxy_recognizer import train, evaluate, visualize, search, finetune
        if args.mode == 'train':
            train.train()
        elif args.mode == 'verify': # Map verify to evaluate for galaxy
            evaluate.run_sweep()
        elif args.mode == 'vis':
            visualize.run_vis()
        elif args.mode == 'demo':
            search.run_contrastive_search()

if __name__ == "__main__":
    main()
