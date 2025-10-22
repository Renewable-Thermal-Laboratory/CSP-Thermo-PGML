# Physics-Guided Neural Network for Thermal Modeling

This directory contains a Physics-Informed Neural Network (PINN) implementation for modeling vertical temperature distribution profiles in Concentrated Solar Power (CSP) receiver tubes.

## Overview

The PhysicsGuidedNeuralNetwork implements a deep learning model that combines data-driven predictions with physical laws governing heat transfer. The model predicts temperature distributions at different depths in a CSP receiver tube while ensuring predictions adhere to energy conservation principles.

## Key Features

### 1. Enhanced Neural Network Architecture
- Multi-layer neural network with residual connections
- Attention mechanisms for feature importance
- Layer normalization and dropout for regularization
- GELU activation functions for improved training

### 2. Physics-Informed Loss Function
- **Energy Conservation Constraint**: Ensures predicted energy storage ≤ incoming energy flux
- **Smoothness Penalties**: Encourages spatially consistent temperature predictions
- **Gradient Regularization**: Maintains physically plausible temperature gradients
- **Conservation Violation Penalty**: Heavily penalizes thermodynamic law violations

### 3. Data Processing Pipeline
- Automated parsing of experimental thermal data files
- Feature engineering with time-based transformations (sin, cos, polynomial features)
- Interactive feature interactions (flux-absorption, heat transfer-flux)
- Proper train/validation/test splitting with scaling

### 4. Model Training with Physics Monitoring
- Real-time conservation violation rate tracking
- Adaptive learning rate scheduling
- Early stopping with physics-aware criteria
- Comprehensive training visualization and logging

## Directory Structure

```
PhysicsGuidedNeuralNetwork/
├── PhysicsInformedNN_Modeling_VTDP.ipynb    # Main implementation notebook
├── data/                                    # Experimental thermal data
│   ├── new_processed_reset/               # Preprocessed dataset files
│   └── new_processed_fix_new/              # Alternative dataset version
├── models/                                 # Saved trained models
├── results/                                # Training results and metrics
└── figures/                               # Generated plots and visualizations
```

## Core Components

### Data Processing
- Parses CSV files with thermal measurements from CSP experiments
- Extracts parameters: heat transfer coefficient (h), heat flux, absorption coefficient
- Handles time-series temperature measurements from multiple thermocouples
- Creates enhanced features for improved model performance

### Model Architecture
```python
EnhancedThermalNet(
    input_size=21,           # Time features + physical parameters + theoretical temps
    output_size=10,          # Predictions for 10 thermocouple positions
    hidden_dims=[512, 256, 256, 128],
    dropout_rate=0.2
)
```

### Physics-Informed Loss
The loss function enforces physical constraints:
- **MSE Loss**: Data fidelity term
- **Smoothness Weight**: Spatial consistency (0.005)
- **Gradient Weight**: Gradient regularization (0.0001)  
- **Physics Weight**: Conservation enforcement (0.5)
- **Conservation Penalty**: Violation penalty (100.0)

### Training Process
1. **Preprocessing**: Load and normalize thermal data
2. **Model Creation**: Initialize enhanced neural network
3. **Physics-Aware Training**: Train with conservation monitoring
4. **Validation**: Monitor energy conservation violations
5. **Evaluation**: Comprehensive performance metrics

## Usage

### Training a New Model
```python
# Run the main execution block in the notebook
model, results = main()

# Or train with custom parameters
train_model_with_conservation_monitoring(
    model, train_loader, val_loader, 
    device, epochs=1000, patience=50
)
```

### Making Predictions
```python
# Load trained model and components
model, scalers, metadata = load_inference_components()

# Predict temperatures for given conditions
result = predict_temperature(
    model, X_scaler, y_scaler,
    time=0, h=0.1575, flux=25900,
    abs_val=20, surf=0.98,
    theoretical_temps=[322.3, 344.7, ...]
)
```

## Results and Evaluation

### Temperature Prediction Metrics
- **Per-Sensor Accuracy**: RMSE, MAE, MAPE for each thermocouple position
- **Overall Performance**: Aggregate metrics across all sensors
- **Spatial Consistency**: Error distribution analysis along receiver tube depth
- **Physics Compliance**: Energy conservation violation monitoring

### Visualization Capabilities
- Training/validation loss curves
- Sensor-specific error distributions (violin plots)
- Vertical temperature profile comparisons
- Predicted vs. actual temperature scatter plots
- R² score distributions by sensor position

## Physics Constraints Implemented

### Energy Conservation Law
Ensures predictions respect fundamental thermodynamic principles:
```
Energy_stored ≤ Energy_incoming
Energy_stored = ρ × h × A_rec × c_p × ΔT / dt
Energy_incoming = flux × A_rec
```

Where:
- ρ: Material density (1836.31 kg/m³)
- h: Receiver height
- A_rec: Receiver cross-sectional area  
- c_p: Specific heat capacity (1512 J/kg·K)
- ΔT: Temperature change
- dt: Time step
- flux: Heat flux

## Best Practices and Features

### Data Quality Assurance
- Automatic filtering of corrupted/missing data points
- Configurable h-parameter filtering for specific receiver geometries
- Robust file parsing with comprehensive error handling
- Duplicate column removal and data validation

### Model Robustness
- Reproducible results with fixed random seeds
- GPU/CPU adaptive computation
- Gradient clipping for training stability
- Comprehensive model checkpointing

### Interpretability
- Attention mechanism for feature importance analysis
- Detailed error distribution reporting
- Per-sensor performance breakdown
- Physics violation rate monitoring

## Requirements

- PyTorch ≥ 1.9
- NumPy ≥ 1.21
- Pandas ≥ 1.3
- Scikit-learn ≥ 1.0
- Matplotlib ≥ 3.5
- Seaborn ≥ 0.11
- Joblib ≥ 1.1

## Contributing

This framework is designed for extensibility:
1. Add new physics constraints to `PhysicsInformedLoss`
2. Extend model architecture with additional layers/features  
3. Incorporate new data sources with custom parsers
4. Enhance visualization capabilities for domain-specific analysis

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. Journal of Computational Physics.
2. CSP Receiver Tube Thermal Modeling Techniques
3. Energy Conservation in Heat Transfer Systems

---
*This implementation bridges machine learning with thermodynamic principles for reliable CSP thermal predictions.*