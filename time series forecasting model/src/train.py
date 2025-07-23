import torch
from torch.utils.data import DataLoader
from dataset_builder import TempSequenceDataset
from model import ImprovedTempModel
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
import json

# Import updated physics loss functions - CORRECTED IMPORTS
from physics_loss import (
    compute_physics_loss_improved,
    compute_enhanced_energy_conservation_loss,
    get_adaptive_physics_weight,
    track_enhanced_physics_metrics,
    calculate_total_energy_stored
)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config - ENHANCED with better physics integration
config = {
    "sequence_length": 20,
    "batch_size": 64,
    "epochs": 200,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "patience": 25,
    "initial_lambda_physics": 2.0,  # Will be adaptive
    "lambda_conservation": 1.0,    # Energy conservation weight
    "conservation_penalty_weight": 5.0,  # Reduced from 10.0 for stability
    "delta_t": 20.0,  # 20-second intervals
    "physics_debug_frequency": 50,  # Debug every N batches
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset
dataset = TempSequenceDataset(
    "data/processed_New_theoretical_data",
    sequence_length=config["sequence_length"]
)

train_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True
)

model = ImprovedTempModel(
    input_size=dataset[0][0].shape[-1],
    output_size=dataset[0][1].shape[-1]
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"]
)
criterion = nn.HuberLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=8,
    factor=0.5
)

# PHYSICS CONSTANTS
rho = 1836.31  # kg/m³
cp = 1512      # J/(kg·K)
diameter = 0.1035  # m
radius = diameter / 2  # m
delta_t = config["delta_t"]  # 20 seconds

print(f"Physics constants: ρ={rho} kg/m³, cp={cp} J/(kg·K), radius={radius:.4f} m")

# Enhanced training metrics tracking
training_metrics = {
    'epochs': [],
    'train_loss': [],
    'data_loss': [],
    'physics_loss': [],
    'conservation_loss': [],
    'adaptive_weight': [],
    'val_mse': [],
    'val_mae': [],
    'val_r2': [],
    'physics_metrics': [],
    'conservation_metrics': [],
    # Enhanced physics components
    'energy_loss': [],
    'gradient_loss': [],
    'temp_bounds_loss': [],
    'temporal_consistency_loss': []
}

best_metrics = {"mse": float('inf'), "mae": float('inf'), "r2": -float('inf')}
best_physics_loss = float('inf')

print("Starting training with IMPROVED physics loss functions...")
print(f"Configuration: {config}")

for epoch in range(config["epochs"]):
    model.train()
    epoch_train_loss = 0
    epoch_data_loss = 0
    epoch_physics_loss = 0
    epoch_conservation_loss = 0
    epoch_adaptive_weights = []
    
    # Enhanced physics metrics tracking
    epoch_physics_components = {
        'energy_loss': [],
        'gradient_loss': [],
        'temp_bounds_loss': [],
        'temporal_consistency_loss': []
    }
    
    # Conservation metrics aggregation
    epoch_conservation_metrics = {
        'violation_ratio': [],
        'max_violation': [],
        'energy_efficiency': [],
        'incoming_energy_avg': [],
        'predicted_energy_avg': [],
        'negative_energy_samples': []
    }
    
    batch_count = 0
    
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        # Get predictions
        outputs = model(X)
        
        # Data loss
        loss_data = criterion(outputs, y)
        
        # Get initial temperatures (t=0, 20 seconds ago)
        initial_temps = X[:, 0, :10]  # T at beginning of sequence
        
        # Extract scaled h_total for physics calculations
        h_scaled = X[:, -1, 10]
        
        # Inverse-transform h_total to get real height in meters
        zeros_params = torch.zeros((h_scaled.shape[0], 4), device=h_scaled.device)
        scaled_params = torch.cat([h_scaled.unsqueeze(1), zeros_params], dim=1)
        
        h_total_raw_np = dataset.param_scaler.inverse_transform(
            scaled_params.detach().cpu().numpy()
        )
        h_total_raw = torch.tensor(
            h_total_raw_np[:, 0], device=h_scaled.device, dtype=torch.float32
        )
        
        # IMPROVED PHYSICS LOSS with multiple constraints
        debug_physics = (epoch == 0 and batch_idx == 0) or (batch_idx % config["physics_debug_frequency"] == 0)
        
        loss_physics, physics_debug_info = compute_physics_loss_improved(
            outputs,
            y,
            initial_temps,
            dataset.thermal_scaler,
            rho,
            cp,
            h_total_raw,
            radius,
            delta_t,
            debug=debug_physics
        )
        
        # Track physics components
        for key in ['energy_loss', 'gradient_loss', 'temp_bounds_loss', 'temporal_consistency_loss']:
            epoch_physics_components[key].append(physics_debug_info[key])
        
        # ENHANCED ENERGY CONSERVATION CONSTRAINT
        loss_conservation, conservation_info = compute_enhanced_energy_conservation_loss(
            outputs,
            X,
            dataset.thermal_scaler,
            dataset.param_scaler,
            rho,
            cp,
            radius,
            delta_t=delta_t,
            penalty_weight=config["conservation_penalty_weight"],
            debug=debug_physics
        )
        
        # Track conservation metrics
        for key in epoch_conservation_metrics.keys():
            if key in conservation_info:
                epoch_conservation_metrics[key].append(conservation_info[key])
        
        # Get adaptive physics weight
        adaptive_weight = get_adaptive_physics_weight(
            loss_physics.item(),
            loss_data.item(),
            epoch
        )
        
        # Combined loss with all constraints
        loss = (loss_data + 
                adaptive_weight * loss_physics + 
                config["lambda_conservation"] * loss_conservation)
        
        # Backpropagation with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        epoch_train_loss += loss.item()
        epoch_data_loss += loss_data.item()
        epoch_physics_loss += loss_physics.item()
        epoch_conservation_loss += loss_conservation.item()
        epoch_adaptive_weights.append(adaptive_weight)
        batch_count += 1
        
        # Enhanced progress reporting
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}: "
                  f"Data: {loss_data.item():.4f}, "
                  f"Physics: {loss_physics.item():.4f} "
                  f"(E:{physics_debug_info['energy_loss']:.3f}, "
                  f"G:{physics_debug_info['gradient_loss']:.3f}, "
                  f"B:{physics_debug_info['temp_bounds_loss']:.3f}, "
                  f"T:{physics_debug_info['temporal_consistency_loss']:.3f}), "
                  f"Conservation: {loss_conservation.item():.4f}, "
                  f"Weight: {adaptive_weight:.2f}, "
                  f"Violations: {conservation_info['violation_ratio']:.1%}")
    
    # Average metrics for the epoch
    avg_train_loss = epoch_train_loss / batch_count
    avg_data_loss = epoch_data_loss / batch_count
    avg_physics_loss = epoch_physics_loss / batch_count
    avg_conservation_loss = epoch_conservation_loss / batch_count
    avg_adaptive_weight = np.mean(epoch_adaptive_weights)
    
    # Average physics components
    avg_physics_components = {
        key: np.mean(values) for key, values in epoch_physics_components.items()
    }
    
    # Average conservation metrics
    avg_conservation_metrics = {
        key: np.mean(values) if values else 0.0 
        for key, values in epoch_conservation_metrics.items()
    }
    
    # VALIDATION with enhanced physics evaluation
    model.eval()
    X_val, y_val = dataset.get_val_data()
    X_val, y_val = X_val.to(device), y_val.to(device)
    
    with torch.no_grad():
        val_preds = model(X_val).cpu().numpy()
        val_targets = y_val.cpu().numpy()
    
    val_preds_raw = dataset.thermal_scaler.inverse_transform(val_preds)
    val_targets_raw = dataset.thermal_scaler.inverse_transform(val_targets)
    
    val_mse = mean_squared_error(val_targets_raw, val_preds_raw)
    val_mae = mean_absolute_error(val_targets_raw, val_preds_raw)
    val_r2 = r2_score(val_targets_raw, val_preds_raw)
    
    # Enhanced validation physics metrics
    val_initial_temps = X_val[:, 0, :10]
    val_h_scaled = X_val[:, -1, 10]
    
    val_zeros_params = torch.zeros((val_h_scaled.shape[0], 4), device=val_h_scaled.device)
    val_scaled_params = torch.cat([val_h_scaled.unsqueeze(1), val_zeros_params], dim=1)
    val_h_total_raw_np = dataset.param_scaler.inverse_transform(
        val_scaled_params.detach().cpu().numpy()
    )
    val_h_total_raw = torch.tensor(
        val_h_total_raw_np[:, 0], device=val_h_scaled.device, dtype=torch.float32
    )
    
    # Enhanced physics metrics tracking
    val_physics_metrics = track_enhanced_physics_metrics(
        torch.tensor(val_preds).to(device),
        y_val,
        val_initial_temps,
        dataset.thermal_scaler,
        rho, cp, val_h_total_raw, radius, delta_t
    )
    
    # Validation conservation metrics
    with torch.no_grad():
        val_conservation_loss, val_conservation_info = compute_enhanced_energy_conservation_loss(
            torch.tensor(val_preds).to(device),
            X_val,
            dataset.thermal_scaler,
            dataset.param_scaler,
            rho,
            cp,
            radius,
            delta_t=delta_t,
            penalty_weight=config["conservation_penalty_weight"],
            debug=False
        )
    
    # Store enhanced metrics
    training_metrics['epochs'].append(epoch + 1)
    training_metrics['train_loss'].append(avg_train_loss)
    training_metrics['data_loss'].append(avg_data_loss)
    training_metrics['physics_loss'].append(avg_physics_loss)
    training_metrics['conservation_loss'].append(avg_conservation_loss)
    training_metrics['adaptive_weight'].append(avg_adaptive_weight)
    training_metrics['val_mse'].append(val_mse)
    training_metrics['val_mae'].append(val_mae)
    training_metrics['val_r2'].append(val_r2)
    training_metrics['physics_metrics'].append(val_physics_metrics)
    training_metrics['conservation_metrics'].append(val_conservation_info)
    
    # Store physics components
    for key in avg_physics_components:
        if key not in training_metrics:
            training_metrics[key] = []
        training_metrics[key].append(avg_physics_components[key])
    
    # Enhanced epoch summary
    print(f"\n=== EPOCH {epoch+1}/{config['epochs']} SUMMARY ===")
    print(f"Total Loss: {avg_train_loss:.4f}")
    print(f"  ├─ Data Loss: {avg_data_loss:.4f}")
    print(f"  ├─ Physics Loss: {avg_physics_loss:.4f}")
    print(f"  │   ├─ Energy: {avg_physics_components['energy_loss']:.4f}")
    print(f"  │   ├─ Gradient: {avg_physics_components['gradient_loss']:.4f}")
    print(f"  │   ├─ Bounds: {avg_physics_components['temp_bounds_loss']:.4f}")
    print(f"  │   └─ Temporal: {avg_physics_components['temporal_consistency_loss']:.4f}")
    print(f"  └─ Conservation: {avg_conservation_loss:.4f}")
    print(f"Adaptive Physics Weight: {avg_adaptive_weight:.2f}")
    
    print(f"Validation Metrics:")
    print(f"  ├─ MSE: {val_mse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")
    print(f"  ├─ Physics Loss: {val_physics_metrics['total_physics_loss']:.4f}")
    print(f"  └─ Conservation: Violations {val_conservation_info['violation_ratio']:.1%}, "
          f"Efficiency {val_conservation_info['energy_efficiency']:.3f}")
    
    # Learning rate scheduling
    scheduler.step(val_mse)
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr != config["learning_rate"]:
        print(f"Learning rate adjusted to: {current_lr:.6f}")
    
    # Save best model with enhanced criteria
    if val_mse < best_metrics["mse"]:
        best_metrics = {"mse": val_mse, "mae": val_mae, "r2": val_r2}
        best_physics_loss = val_physics_metrics['total_physics_loss']
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_metrics': best_metrics,
            'physics_loss': best_physics_loss,
            'config': config
        }, "models_new_theoretical/best_model.pth")
        print("*** SAVED NEW BEST MODEL ***")
    
    # Early stopping check
    if epoch >= config["patience"]:
        recent_mse = training_metrics['val_mse'][-config["patience"]:]
        if all(mse >= best_metrics["mse"] * 1.001 for mse in recent_mse):  # 0.1% tolerance
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

print(f"\n=== TRAINING COMPLETE ===")
print(f"Best validation metrics: MSE={best_metrics['mse']:.2f}, MAE={best_metrics['mae']:.2f}, R²={best_metrics['r2']:.4f}")
print(f"Best physics loss: {best_physics_loss:.4f}")

# Save enhanced training metrics
os.makedirs("results_new_theoretical", exist_ok=True)
with open("results_new_theoretical/training_metrics.json", "w") as f:
    # Convert any numpy types to native Python types for JSON serialization
    serializable_metrics = {}
    for key, value in training_metrics.items():
        if isinstance(value, list):
            serializable_metrics[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in value]
        else:
            serializable_metrics[key] = value
    json.dump(serializable_metrics, f, indent=2)

# FINAL TEST EVALUATION with enhanced metrics
print("\n=== FINAL TEST EVALUATION ===")
checkpoint = torch.load("models_new_theoretical/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

X_test, y_test = dataset.get_test_data()
X_test, y_test = X_test.to(device), y_test.to(device)

with torch.no_grad():
    test_preds = model(X_test).cpu().numpy()
    test_targets = y_test.cpu().numpy()

test_preds_raw = dataset.thermal_scaler.inverse_transform(test_preds)
test_targets_raw = dataset.thermal_scaler.inverse_transform(test_targets)

test_mse = mean_squared_error(test_targets_raw, test_preds_raw)
test_mae = mean_absolute_error(test_targets_raw, test_preds_raw)
test_r2 = r2_score(test_targets_raw, test_preds_raw)

print(f"Final Test Results: MSE={test_mse:.2f}, MAE={test_mae:.2f}, R²={test_r2:.4f}")

# Enhanced final test physics evaluation
test_initial_temps = X_test[:, 0, :10]
test_h_scaled = X_test[:, -1, 10]

test_zeros_params = torch.zeros((test_h_scaled.shape[0], 4), device=test_h_scaled.device)
test_scaled_params = torch.cat([test_h_scaled.unsqueeze(1), test_zeros_params], dim=1)
test_h_total_raw_np = dataset.param_scaler.inverse_transform(
    test_scaled_params.detach().cpu().numpy()
)
test_h_total_raw = torch.tensor(
    test_h_total_raw_np[:, 0], device=test_h_scaled.device, dtype=torch.float32
)

# Final comprehensive physics evaluation
final_physics_metrics = track_enhanced_physics_metrics(
    torch.tensor(test_preds).to(device),
    y_test,
    test_initial_temps,
    dataset.thermal_scaler,
    rho, cp, test_h_total_raw, radius, delta_t
)

# Final conservation evaluation with detailed debug
with torch.no_grad():
    final_conservation_loss, final_conservation_info = compute_enhanced_energy_conservation_loss(
        torch.tensor(test_preds).to(device),
        X_test,
        dataset.thermal_scaler,
        dataset.param_scaler,
        rho,
        cp,
        radius,
        delta_t=delta_t,
        penalty_weight=config["conservation_penalty_weight"],
        debug=True
    )

print(f"\n=== COMPREHENSIVE FINAL PHYSICS EVALUATION ===")
print(f"Enhanced Physics Metrics:")
for key, value in final_physics_metrics.items():
    print(f"  {key}: {value:.4f}")

print(f"\nEnhanced Conservation Metrics:")
for key, value in final_conservation_info.items():
    if isinstance(value, (int, float)):
        if 'ratio' in key or 'efficiency' in key:
            print(f"  {key}: {value:.1%}" if 'ratio' in key else f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")

# ENHANCED PLOTTING with all physics components
plt.figure(figsize=(20, 16))

# Row 1: Loss evolution
plt.subplot(4, 5, 1)
plt.plot(training_metrics['epochs'], training_metrics['train_loss'], 'b-', label='Total Loss')
plt.plot(training_metrics['epochs'], training_metrics['data_loss'], 'g-', label='Data Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 2)
plt.plot(training_metrics['epochs'], training_metrics['physics_loss'], 'r-', label='Physics Loss')
plt.xlabel('Epoch')
plt.ylabel('Physics Loss')
plt.title('Physics Loss Evolution')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 3)
plt.plot(training_metrics['epochs'], training_metrics['conservation_loss'], 'orange', label='Conservation')
plt.xlabel('Epoch')
plt.ylabel('Conservation Loss')
plt.title('Energy Conservation Loss')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 4)
plt.plot(training_metrics['epochs'], training_metrics['adaptive_weight'], 'purple', label='Adaptive Weight')
plt.xlabel('Epoch')
plt.ylabel('Physics Weight')
plt.title('Adaptive Physics Weight')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 5)
plt.plot(training_metrics['epochs'], training_metrics['val_mse'], 'brown', label='Val MSE')
plt.plot(training_metrics['epochs'], training_metrics['val_mae'], 'pink', label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Validation Errors')
plt.legend()
plt.grid(True)

# Row 2: Physics components breakdown
plt.subplot(4, 5, 6)
plt.plot(training_metrics['epochs'], training_metrics['energy_loss'], 'red', label='Energy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Energy Physics Loss')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 7)
plt.plot(training_metrics['epochs'], training_metrics['gradient_loss'], 'blue', label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Gradient Physics Loss')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 8)
plt.plot(training_metrics['epochs'], training_metrics['temp_bounds_loss'], 'green', label='Bounds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Temperature Bounds Loss')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 9)
plt.plot(training_metrics['epochs'], training_metrics['temporal_consistency_loss'], 'purple', label='Temporal')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Temporal Consistency Loss')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 10)
plt.plot(training_metrics['epochs'], training_metrics['val_r2'], 'cyan', label='Validation R²')
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.title('Validation R²')
plt.legend()
plt.grid(True)

# Row 3: Conservation metrics
plt.subplot(4, 5, 11)
violation_ratios = [m['violation_ratio'] for m in training_metrics['conservation_metrics']]
plt.plot(training_metrics['epochs'], violation_ratios, 'red', label='Violation Ratio')
plt.axhline(y=0.0, color='black', linestyle='--', alpha=0.5, label='No Violations')
plt.xlabel('Epoch')
plt.ylabel('Violation Ratio')
plt.title('Energy Conservation Violations')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 12)
energy_efficiencies = [m['energy_efficiency'] for m in training_metrics['conservation_metrics']]
plt.plot(training_metrics['epochs'], energy_efficiencies, 'darkblue', label='Energy Efficiency')
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='100% Efficiency')
plt.xlabel('Epoch')
plt.ylabel('Efficiency')
plt.title('Energy Storage Efficiency')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 13)
max_violations = [m['max_violation'] for m in training_metrics['conservation_metrics']]
plt.plot(training_metrics['epochs'], max_violations, 'darkred', label='Max Violation')
plt.xlabel('Epoch')
plt.ylabel('Max Violation (J/s)')
plt.title('Maximum Energy Violations')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 14)
incoming_energies = [m['incoming_energy_avg'] for m in training_metrics['conservation_metrics']]
predicted_energies = [m['predicted_energy_avg'] for m in training_metrics['conservation_metrics']]
plt.plot(training_metrics['epochs'], incoming_energies, 'green', label='Incoming')
plt.plot(training_metrics['epochs'], predicted_energies, 'blue', label='Predicted')
plt.xlabel('Epoch')
plt.ylabel('Energy (J/s)')
plt.title('Energy Comparison')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 15)
negative_energy_samples = [m['negative_energy_samples'] for m in training_metrics['conservation_metrics']]
plt.plot(training_metrics['epochs'], negative_energy_samples, 'darkred', label='Negative Energy')
plt.xlabel('Epoch')
plt.ylabel('Count')
plt.title('Negative Energy Samples')
plt.legend()
plt.grid(True)

# Row 4: Enhanced physics metrics
plt.subplot(4, 5, 16)
total_physics_losses = [m['total_physics_loss'] for m in training_metrics['physics_metrics']]
plt.plot(training_metrics['epochs'], total_physics_losses, 'red', label='Total Physics')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Physics Loss')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 17)
energy_losses_val = [m['energy_loss'] for m in training_metrics['physics_metrics']]
plt.plot(training_metrics['epochs'], energy_losses_val, 'blue', label='Energy Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Energy Loss')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 18)
predicted_energies_val = [m['predicted_total_energy'] for m in training_metrics['physics_metrics']]
actual_energies_val = [m['actual_total_energy'] for m in training_metrics['physics_metrics']]
plt.plot(training_metrics['epochs'], predicted_energies_val, 'red', label='Predicted')
plt.plot(training_metrics['epochs'], actual_energies_val, 'blue', label='Actual')
plt.xlabel('Epoch')
plt.ylabel('Energy (J/s)')
plt.title('Validation Energy Comparison')
plt.legend()
plt.grid(True)

# Final two plots for overall assessment
plt.subplot(4, 5, 19)
# Combined physics score
combined_physics_score = []
for i in range(len(training_metrics['epochs'])):
    score = (training_metrics['physics_loss'][i] + 
             training_metrics['conservation_loss'][i])
    combined_physics_score.append(score)
plt.plot(training_metrics['epochs'], combined_physics_score, 'darkviolet', label='Combined Physics')
plt.xlabel('Epoch')
plt.ylabel('Combined Loss')
plt.title('Combined Physics Score')
plt.legend()
plt.grid(True)

plt.subplot(4, 5, 20)
# Learning curve comparison
plt.plot(training_metrics['epochs'], training_metrics['train_loss'], 'b-', alpha=0.7, label='Train')
plt.plot(training_metrics['epochs'], training_metrics['val_mse'], 'r-', alpha=0.7, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss/Error')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results_new_theoretical/enhanced_training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate test predictions and plots
print("\nGenerating enhanced test predictions...")

from predicted_graph import process_file, print_numerical_results, plot_depth_profile

all_results = []
output_dir = "results_new_theoretical/predicted_graph_plots"
os.makedirs(output_dir, exist_ok=True)

for filepath in dataset.test_files:
    try:
        result = process_file(
            filepath,
            model,
            dataset.thermal_scaler,
            dataset.param_scaler,
            config["sequence_length"]
        )
        all_results.append(result)
        print(f"Processed: {result['filename']} - Avg Residual: {result['avg_residual']:.3f}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Sort and display results
all_results.sort(key=lambda x: x['avg_residual'])

print(f"\n=== FINAL RESULTS SUMMARY ({len(all_results)} files) ===")
print("Best performing files:")
for r in all_results[:5]:
    print(f"  {r['filename']:<40} Residual: {r['avg_residual']:.3f}")

print("\nWorst performing files:")
for r in all_results[-5:]:
    print(f"  {r['filename']:<40} Residual: {r['avg_residual']:.3f}")

# Generate and save plots
for i, result in enumerate(all_results):
    try:
        print_numerical_results(result)
        save_path = os.path.join(output_dir, f"{result['filename'].replace('.csv', '')}.png")
        plot_depth_profile(result, save_path)
        print(f"Saved plot: {save_path}")
    except Exception as e:
        print(f"Error plotting {result['filename']}: {e}")

