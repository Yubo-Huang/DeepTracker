import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from Network.LSTM_reg import LSTM_Reg, LSTM_Reg_Attn
from dataset import HDF5Dataset

# =========================================================
# Configuration
# =========================================================
SAVE_DIR = "model/lstm2atten" 
TEST_DATA_PATH = "selected_test_data_reg_horizon_1000.h5" 
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Step 1: Find the best model checkpoint
# =========================================================
def find_best_model(save_dir):
    model_files = [f for f in os.listdir(save_dir) if f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError(f"No saved models found in {save_dir}")
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(save_dir, f)), reverse=True)
    best_model = os.path.join(save_dir, model_files[0])
    print(f"🔍 Found checkpoint: {best_model}")
    return best_model

# =========================================================
# Step 2: Load model
# =========================================================
def load_model(checkpoint_path, device):
    model = LSTM_Reg_Attn(
        input_size=8,
        hidden_size=256,
        num_layers=2,
        num_out=2
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("✅ Model weights loaded successfully.")
    return model

# =========================================================
# Step 3: Evaluate on test set (Corrected for Degrees)
# =========================================================
def evaluate(model, loader, device, save_csv=True):
    total_loss = 0.0
    total_count = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y1, y2 in loader:
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            y = torch.stack((y1, y2), dim=1)

            preds = model(x)
            loss = F.mse_loss(preds, y, reduction="sum")
            total_loss += loss.item()
            total_count += x.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / total_count
    print(f"📊 Average Test Loss (MSE): {avg_loss:.6f}")

    # 1. Combine batches
    preds_np = np.concatenate(all_preds, axis=0)
    y_np = np.concatenate(all_targets, axis=0)

    # 2. Extract and Scale Coordinates
    # These are in DEGREES based on your requirement
    true_angle_deg = y_np[:, 0] * 10
    true_dist      = y_np[:, 1] * 10
    
    pred_angle_deg = preds_np[:, 0] * 10
    pred_dist      = preds_np[:, 1] * 10

    # 3. Transform Polar to Cartesian
    # CRITICAL CHANGE: Convert degrees to radians for numpy trig functions
    true_x = true_dist * np.cos(np.deg2rad(true_angle_deg))
    true_y = true_dist * np.sin(np.deg2rad(true_angle_deg))

    pred_x = pred_dist * np.cos(np.deg2rad(pred_angle_deg))
    pred_y = pred_dist * np.sin(np.deg2rad(pred_angle_deg))

    # 4. Calculate Euclidean Error
    errors = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
    average_error = np.mean(errors)
    increase_rate = average_error / 6.2867

    print(f"📏 Average Displacement Error: {average_error:.4f}")
    print(f"📏 Average Error Increase Rate: {increase_rate:.4f}")

    # if save_csv:
    #     df_results = pd.DataFrame({
    #         "true_angle_deg": true_angle_deg,
    #         "true_dist": true_dist,
    #         "pred_angle_deg": pred_angle_deg,
    #         "pred_dist": pred_dist,
    #         "true_x": true_x,
    #         "true_y": true_y,
    #         "pred_x": pred_x,
    #         "pred_y": pred_y,
    #         "error": errors
    #     })
        
    #     csv_file = os.path.join(SAVE_DIR, "test_predictions.csv")
    #     df_results.to_csv(csv_file, index=False)
    #     print(f"📁 Saved predictions to: {csv_file}")

    return avg_loss

# =========================================================
# Step 4: Main function
# =========================================================
def main():
    print("🚀 Starting model evaluation...")

    checkpoint = find_best_model(SAVE_DIR)
    model = load_model(checkpoint, DEVICE)

    test_set = HDF5Dataset(TEST_DATA_PATH)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    evaluate(model, test_loader, DEVICE, save_csv=True)
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    main()