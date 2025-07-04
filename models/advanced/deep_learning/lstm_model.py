"""
LSTM Model for F1 Lap Time Prediction

This module implements a sophisticated LSTM-based neural network for predicting
F1 lap times using sequential historical data. The model captures temporal
dependencies and racing dynamics that traditional ML models miss.

Key Features:
- Multi-layer LSTM with attention mechanism
- Bidirectional processing for better context understanding
- Dropout and regularization for generalization
- Flexible sequence length handling
- Support for multiple input features
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class F1SequenceDataset(Dataset):
    """
    Dataset class for creating sequences of F1 lap data for LSTM training.
    
    Creates sequences of previous laps to predict the next lap time.
    Handles multiple drivers and sessions properly.
    """
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 10, 
                 target_col: str = 'LapTime', driver_col: str = 'Driver',
                 session_col: str = 'LapNumber', feature_cols: List[str] = None):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with lap data
            sequence_length: Number of previous laps to use as input
            target_col: Column name for target variable (lap time)
            driver_col: Column name for driver identification
            session_col: Column name for lap numbering
            feature_cols: List of feature columns to use as input
        """
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.driver_col = driver_col
        self.session_col = session_col
        
        if feature_cols is None:
            # Use numeric columns as features, excluding target and identifiers
            self.feature_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                               if col not in [target_col, driver_col, session_col]]
        else:
            self.feature_cols = feature_cols
            
        self.sequences, self.targets = self._create_sequences(data)
        
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from the data for each driver session.
        
        Returns:
            sequences: Array of shape (n_sequences, sequence_length, n_features)
            targets: Array of shape (n_sequences,)
        """
        sequences = []
        targets = []
        
        # Group by driver to maintain temporal consistency
        for driver in data[self.driver_col].unique():
            driver_data = data[data[self.driver_col] == driver].sort_values(self.session_col)
            
            # Skip if not enough data for sequences
            if len(driver_data) < self.sequence_length + 1:
                continue
                
            # Create sequences for this driver
            for i in range(self.sequence_length, len(driver_data)):
                # Get sequence of previous laps
                sequence = driver_data.iloc[i-self.sequence_length:i][self.feature_cols].values
                
                # Get target (current lap time)
                target = driver_data.iloc[i][self.target_col]
                
                sequences.append(sequence)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])

class AttentionLSTM(nn.Module):
    """
    LSTM model with attention mechanism for F1 lap time prediction.
    
    Architecture:
    - Bidirectional LSTM layers for temporal processing
    - Attention mechanism to focus on important time steps
    - Fully connected layers for final prediction
    - Dropout for regularization
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2,
                 bidirectional: bool = True, use_attention: bool = True):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
        """
        super(AttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        if use_attention:
            lstm_output_size = hidden_size * (2 if bidirectional else 1)
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False)
            )
        
        # Final prediction layers
        final_input_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(final_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            output: Predicted lap time
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.use_attention:
            # Apply attention mechanism
            attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Weighted sum of LSTM outputs
            context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size)
        else:
            # Use last LSTM output
            context = lstm_out[:, -1, :]
        
        # Final prediction
        output = self.fc(context)
        
        return output

class F1LSTMPredictor:
    """
    Main class for training and using LSTM models for F1 lap time prediction.
    
    Handles data preprocessing, model training, evaluation, and prediction.
    """
    
    def __init__(self, sequence_length: int = 10, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 bidirectional: bool = True, use_attention: bool = True):
        """
        Initialize the predictor.
        
        Args:
            sequence_length: Number of previous laps to use as input
            hidden_size: Hidden state size for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
        """
        self.sequence_length = sequence_length
        self.model_params = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'bidirectional': bidirectional,
            'use_attention': use_attention
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_cols = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Using device: {self.device}")
        
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'LapTime',
                    feature_cols: List[str] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training.
        
        Args:
            data: DataFrame with lap data
            target_col: Target column name
            feature_cols: Feature columns to use
            
        Returns:
            train_loader, val_loader: DataLoaders for training and validation
        """
        print("📊 Preparing sequence data for LSTM...")
        
        # Handle missing values
        data = data.dropna()
        
        # Select features
        if feature_cols is None:
            # Use actual F1 features from the loaded data - prioritize the most important ones
            important_features = [
                'Sector1Time', 'Sector2Time', 'Sector3Time',
                'TyreLife', 'Position', 'LapNumber',
                'rolling_avg_lap_time_3', 'rolling_avg_lap_time_5', 'rolling_avg_lap_time_10',
                'best_lap_so_far', 'team_avg_lap_time', 'avg_lap_so_far',
                'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
                'gap_to_leader', 'gap_to_ahead', 'gap_to_behind',
                'rolling_avg_position_3', 'rolling_avg_position_5',
                'compound_age_ratio', 'tyre_age_normalized',
                'race_progress', 'laps_completed', 'laps_in_stint',
                'speed_consistency', 'speed_range', 'speed_improvement',
                'car_ahead_time', 'car_behind_time', 'lap_time_percentile',
                'rolling_std_lap_time_3', 'rolling_std_lap_time_5',
                'personal_bests_count', 'team_avg_position'
            ]
            self.feature_cols = [col for col in important_features if col in data.columns]
            
            # If still not many features, use all numeric columns except target/identifiers
            if len(self.feature_cols) < 10:
                self.feature_cols = [col for col in data.columns 
                                   if col not in ['LapTime', 'Driver', 'Team', 'DriverNumber'] and 
                                   data[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        else:
            self.feature_cols = feature_cols
            
        print(f"🎯 Using {len(self.feature_cols)} features: {self.feature_cols}")
        
        # Scale features
        data_scaled = data.copy()
        data_scaled[self.feature_cols] = self.scaler.fit_transform(data[self.feature_cols])
        
        # Scale target
        target_reshaped = data[target_col].values.reshape(-1, 1)
        data_scaled[target_col] = self.target_scaler.fit_transform(target_reshaped).flatten()
        
        # Create dataset
        dataset = F1SequenceDataset(
            data_scaled, 
            sequence_length=self.sequence_length,
            target_col=target_col,
            feature_cols=self.feature_cols
        )
        
        print(f"📈 Created {len(dataset)} sequences")
        
        # Split into train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        print(f"✅ Training sequences: {len(train_dataset)}, Validation: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, learning_rate: float = 0.001,
              patience: int = 15, min_delta: float = 0.001) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            
        Returns:
            training_history: Dictionary with training metrics
        """
        print("🚀 Starting LSTM training...")
        
        # Initialize model
        input_size = len(self.feature_cols)
        self.model = AttentionLSTM(input_size=input_size, **self.model_params)
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)
            
            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏰ Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_lstm_model.pth'))
        print(f"✅ Training completed! Best validation loss: {best_val_loss:.6f}")
        
        return history
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: DataFrame with lap data
            
        Returns:
            predictions: Array of predicted lap times
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale features
        data_scaled = data.copy()
        data_scaled[self.feature_cols] = self.scaler.transform(data[self.feature_cols])
        
        # Create dataset
        dataset = F1SequenceDataset(
            data_scaled,
            sequence_length=self.sequence_length,
            target_col='LapTime',  # Won't be used for prediction
            feature_cols=self.feature_cols
        )
        
        # Create data loader
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Make predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.extend(outputs.cpu().numpy().flatten())
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.target_scaler.inverse_transform(predictions).flatten()
        
        return predictions
    
    def evaluate(self, data: pd.DataFrame, target_col: str = 'LapTime') -> Dict:
        """
        Evaluate model performance.
        
        Args:
            data: Test data
            target_col: Target column name
            
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        predictions = self.predict(data)
        
        # Get actual targets (need to account for sequence creation)
        dataset = F1SequenceDataset(
            data,
            sequence_length=self.sequence_length,
            target_col=target_col,
            feature_cols=self.feature_cols
        )
        
        actual = dataset.targets
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        r2 = r2_score(actual, predictions)
        mae = np.mean(np.abs(actual - predictions))
        
        metrics = {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'n_predictions': len(predictions)
        }
        
        print(f"📊 LSTM Model Performance:")
        print(f"   RMSE: {rmse:.4f} seconds")
        print(f"   R²: {r2:.4f}")
        print(f"   MAE: {mae:.4f} seconds")
        print(f"   Predictions: {len(predictions)}")
        
        return metrics
    
    def plot_training_history(self, history: Dict):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Training Loss', alpha=0.7)
        axes[0].plot(history['val_loss'], label='Validation Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1].plot(history['lr'], label='Learning Rate', color='red', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Example usage of the F1 LSTM predictor.
    """
    print("🏎️ F1 LSTM Lap Time Predictor Demo")
    print("=" * 50)
    
    # This would typically load your actual F1 data
    # For demo, we'll create sample data structure
    print("📁 Loading F1 data...")
    
    # You would load your actual data like this:
    # data = pd.read_csv('../../../data/processed/2024/Monaco_Grand_Prix/laps_features.csv')
    
    # For demo purposes, create sample data structure
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'LapTime': np.random.normal(80, 3, n_samples),
        'Sector1Time': np.random.normal(25, 1, n_samples),
        'Sector2Time': np.random.normal(35, 2, n_samples),
        'Sector3Time': np.random.normal(20, 1, n_samples),
        'TyreLife': np.random.randint(1, 40, n_samples),
        'Position': np.random.randint(1, 21, n_samples),
        'LapNumber': np.tile(range(1, 51), 20),
        'Driver': np.repeat(['HAM', 'VER', 'LEC', 'NOR'], 250),
        'rolling_avg_3': np.random.normal(80, 2, n_samples),
        'rolling_avg_5': np.random.normal(80, 2, n_samples),
        'rolling_avg_10': np.random.normal(80, 2, n_samples),
        'best_lap_so_far': np.random.normal(78, 2, n_samples),
        'team_avg_lap_time': np.random.normal(81, 3, n_samples),
        'sector1_delta': np.random.normal(0, 0.5, n_samples),
        'sector2_delta': np.random.normal(0, 0.8, n_samples),
        'sector3_delta': np.random.normal(0, 0.3, n_samples)
    })
    
    print(f"✅ Loaded {len(sample_data)} lap records")
    
    # Initialize predictor
    predictor = F1LSTMPredictor(
        sequence_length=8,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        use_attention=True
    )
    
    # Prepare data
    train_loader, val_loader = predictor.prepare_data(sample_data)
    
    # Train model
    history = predictor.train(
        train_loader, val_loader,
        epochs=50,
        learning_rate=0.001,
        patience=10
    )
    
    # Plot training history
    predictor.plot_training_history(history)
    
    # Evaluate model
    metrics = predictor.evaluate(sample_data)
    
    print("\n🎯 Next Steps:")
    print("1. Replace sample data with your actual F1 data")
    print("2. Experiment with different sequence lengths")
    print("3. Try different LSTM architectures")
    print("4. Add more sophisticated features")
    print("5. Implement ensemble with existing models")

if __name__ == "__main__":
    main()
