import pandas as pd
import torch
import copy
import os
import time

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def display_elapsed_time(time_started, msg="Elapsed"):
    elapsed = time.time() - time_started
    print(msg, '{:.0f}m {:.0f}s'.format(
        elapsed // 60, elapsed % 60))

class CheckpointManager():
    def __init__(self, max_items=10):
        self.checkpoints = []
        self.min_loss = float("inf")
        self.max_items = max_items
        self.stats = []
    
    def add(self, model, batch_loss, val_auc, val_loss):
        self.stats.append({
            "batch_loss": batch_loss,
            "val_auc": val_auc,
            "val_loss": val_loss
        })
        self._add_model(model, val_auc)
    
    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        stats_df = pd.DataFrame(self.stats)
        stats_df.to_csv(os.path.join(output_dir, 'stats.csv'))
        for i, chk in enumerate(self.checkpoints):
            torch.save(chk["model"], os.path.join(output_dir, "checkpoint_{}.pth".format(i)))
    
    def _add_model(self, model, val_auc):
        if not self._should_add(val_auc):
            return
        checkpoint = {
            "model": copy.deepcopy(model.state_dict()),
            "auc": val_auc
        }
        self.checkpoints.append(checkpoint)
        self._sort_and_prune()
    
    def _should_add(self, auc):
        if len(self.checkpoints) < self.max_items:
            return True
        if auc < self.min_loss:
            return True
        return False
    
    def _sort_and_prune(self):
        self.checkpoints = sorted(self.checkpoints, key=lambda c: -c["auc"])
        self.checkpoints = self.checkpoints[:self.max_items]
        