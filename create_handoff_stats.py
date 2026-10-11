import json 
from pathlib import Path 

handoff_root = Path("robotwin-handoff") 
stats_in = json.loads(Path("outputs/robotwin-clean-stats.json").read_text()) 

# 1. Create reference action mean/std statistics file 
ref_dir = handoff_root / "reference" 
ref_dir.mkdir(parents=True, exist_ok=True) 
ref_stats = { 
    "schema": "egoscale-robotwin-lerobot-mean-std-v1", 
    "action": {
        "mean": stats_in["action"]["mean"], 
        "std": stats_in["action"]["std"], 
        }, 
        } 
(ref_dir / "mean-std-eef16-h50-stage1grip-train95-v2.json").write_text(json.dumps(ref_stats, indent=2)) 