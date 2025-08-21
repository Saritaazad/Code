import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import networkx as nx
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


FILENAME = 'data10000.csv'
COLUMN_NAMES = [
    'longitude', 'lattitude', 'Year', 'Month', 'Day', 'T2M', 'T2M_MAX', 'T2M_MIN', 
    'QV2M', 'WD10M', 'WS10M', 'Pres', 'date', 'elevation'
]
#FEATURES_TO_USE = ['T2M', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'WD10M', 'WS10M', 'Pres', 'elevation']
FEATURES_TO_USE = ['longitude', 'lattitude', 'T2M', 'QV2M', 'Pres', 'elevation']
TARGET_VARIABLE = 'Pres'
TARGET_VARIABLE = 'Pres'
DATE_COLUMN = 'Date'


LOOKBACK_WINDOW = 12
N_NEIGHBORS = 8 
N_LOCATIONS_TO_SELECT = 119
EPOCHS = 2  
LEARNING_RATE = 0.001
BATCH_SIZE = 16
DROPOUT_RATE = 0.3

def calculate_multiple_importance_scores(model, train_loader, edge_index, n_locations, location_map):

    attention_importance = calculate_attention_importance(model, train_loader, edge_index, n_locations)
    removal_importance = calculate_removal_impact(model, train_loader, edge_index, n_locations)
    centrality_importance = calculate_centrality_importance(edge_index, n_locations)
    feature_importance = calculate_feature_contribution(model, train_loader, edge_index, n_locations)
    gradient_importance = calculate_gradient_importance(model, train_loader, edge_index, n_locations)
    combined_importance = combine_importance_scores(
        attention_importance, removal_importance, centrality_importance, 
        feature_importance, gradient_importance
    )
    
    return combined_importance

def calculate_attention_importance(model, train_loader, edge_index, n_locations):
    
    model.eval()
    attention_scores = torch.zeros(n_locations)
    attention_variance = torch.zeros(n_locations)
    attention_count = 0
    
    all_attentions = []
    
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(train_loader):
            try:
                output, attention_info = model(features, edge_index)
                
                if attention_info is not None:
                    edge_idx, attention_weights = attention_info
                    
                    # Validate edges
                    valid_mask = (edge_idx[0] < n_locations) & (edge_idx[1] < n_locations)
                    if valid_mask.sum() > 0:
                        valid_edges = edge_idx[:, valid_mask]
                        valid_attention = attention_weights[valid_mask]

                        batch_attention = torch.zeros(n_locations)

                        for i in range(len(valid_attention)):
                            target_node = valid_edges[1, i].item()
                            source_node = valid_edges[0, i].item()
                            
                            if valid_attention[i].numel() > 0:
                                att_val = valid_attention[i].mean().item()
                                attention_scores[target_node] += att_val
                                batch_attention[target_node] += att_val
                        
                        all_attentions.append(batch_attention)
                        attention_count += 1
                        
            except Exception as e:
                continue
    
    # Calculate attention variance (higher variance = more selective attention)
    if all_attentions:
        attention_matrix = torch.stack(all_attentions)
        attention_variance = torch.var(attention_matrix, dim=0)
        
        # Normalize scores
        if attention_count > 0:
            attention_scores /= attention_count
        
        # Combine mean attention with variance (nodes with high and variable attention are important)
        combined_attention = attention_scores + 0.3 * attention_variance
        
        
        return combined_attention
    else:
        return torch.ones(n_locations)


# def save_dtw_results(coordinates, predictions, dtw_results, year, date_range):
    
#     if dtw_results['location_dtw_results']:
#         location_results = []
#         for loc in dtw_results['location_dtw_results']:
#             result = {
#                 'location_idx': loc['location_idx'],
#                 'longitude': coordinates[loc['location_idx'], 0],
#                 'latitude': coordinates[loc['location_idx'], 1],
#                 'valid_days': loc['valid_days'],
#                 'dtw_distance': loc['dtw_distance'],
#                 'dtw_normalized_distance': loc['dtw_normalized_distance'],
#                 'alignment_length': loc['alignment_length'],
#                 'acceptable_matches': loc['acceptable_matches'],
#                 'acceptable_ratio': loc['acceptable_ratio'],
#                 'mean_temporal_shift': loc['mean_temporal_shift'],
#                 'max_temporal_shift': loc['max_temporal_shift'],
#                 'mean_error': loc['error_stats']['mean_error'],
#                 'std_error': loc['error_stats']['std_error'],
#                 'min_error': loc['error_stats']['min_error'],
#                 'max_error': loc['error_stats']['max_error']
#             }
#             location_results.append(result)
        
#         dtw_df = pd.DataFrame(location_results)
#         dtw_df.to_csv(f'dtw_location_results_{year}.csv', index=False)

#     summary = {
#         'year': year,
#         'window_size': dtw_results.get('window_size', 10),
#         'threshold': dtw_results.get('threshold', 0.05),
#         'total_locations': len(coordinates),
#         'valid_locations': dtw_results.get('valid_locations', 0),
#         'total_aligned_points': dtw_results.get('total_aligned_points', 0),
#         'acceptable_matches': dtw_results.get('acceptable_matches', 0),
#         'acceptable_ratio': dtw_results.get('acceptable_ratio', 0),
#         'mean_dtw_distance': dtw_results.get('mean_dtw_distance', 0),
#         'std_dtw_distance': dtw_results.get('std_dtw_distance', 0),
#         'mean_temporal_shift': dtw_results.get('mean_temporal_shift', 0),
#         'max_temporal_shift': dtw_results.get('max_temporal_shift', 0),
#         'median_temporal_shift': dtw_results.get('temporal_shift_stats', {}).get('median', 0),
#         'missing_data_count': dtw_results.get('missing_data_count', 0)
#     }
    
#     summary_df = pd.DataFrame([summary])
#     summary_df.to_csv(f'dtw_summary_{year}.csv', index=False)
    

#     pred_df = pd.DataFrame({
#         'longitude': coordinates[:, 0],
#         'latitude': coordinates[:, 1]
#     })
    

#     for i, date in enumerate(date_range):
#         if i < predictions.shape[1]:
#             pred_df[f'pred_{date.strftime("%Y-%m-%d")}'] = predictions[:, i]
    
#     pred_df.to_csv(f'dtw_daily_predictions_{year}.csv', index=False)
    
#     print(f"DTW results saved for year {year}")



def calculate_removal_impact(model, train_loader, edge_index, n_locations):
    model.eval()
    baseline_loss = evaluate_model_performance(model, train_loader, edge_index)
    
    removal_importance = torch.zeros(n_locations)
    nodes_to_test = torch.randperm(n_locations)[:min(50, n_locations)]  # Sample nodes
    
    for i, node_idx in enumerate(nodes_to_test):
        try:
            node_mask = (edge_index[0] != node_idx) & (edge_index[1] != node_idx)
            modified_edge_index = edge_index[:, node_mask]
            modified_loss = evaluate_model_performance(model, train_loader, modified_edge_index, exclude_node=node_idx.item())
            importance = modified_loss - baseline_loss
            removal_importance[node_idx] = max(0, importance)  # Only positive impact
            
            if (i + 1) % 10 == 0:
                print(f"   Tested {i + 1}/{len(nodes_to_test)} nodes")
                
        except Exception as e:
            print(f"⚠️ Error testing node {node_idx}: {e}")
            continue
    if len(nodes_to_test) < n_locations:
        removal_importance = interpolate_importance_scores(removal_importance, nodes_to_test, n_locations)

    return removal_importance

def calculate_centrality_importance(edge_index, n_locations):
    print("📊 Method 3: Graph Centrality Analysis...")

    G = nx.Graph()
    G.add_nodes_from(range(n_locations))
    edge_list = edge_index.t().numpy().tolist()
    G.add_edges_from(edge_list)
    try:
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        pagerank = nx.pagerank(G)
        
        # Convert to tensors
        betweenness_scores = torch.tensor([betweenness.get(i, 0) for i in range(n_locations)])
        closeness_scores = torch.tensor([closeness.get(i, 0) for i in range(n_locations)])
        eigenvector_scores = torch.tensor([eigenvector.get(i, 0) for i in range(n_locations)])
        pagerank_scores = torch.tensor([pagerank.get(i, 0) for i in range(n_locations)])
        
        # Combine centrality measures
        centrality_importance = (
            0.3 * betweenness_scores +
            0.3 * eigenvector_scores +
            0.2 * pagerank_scores +
            0.2 * closeness_scores
        )
        return centrality_importance
        
    except Exception as e:
        print('e')
        return torch.ones(n_locations)

def calculate_feature_contribution(model, train_loader, edge_index, n_locations):
    
    model.eval()
    feature_importance = torch.zeros(n_locations)
    
    with torch.no_grad():
        for features, targets in train_loader:
            try:
                output_original, _ = model(features, edge_index)

                for loc_idx in range(min(n_locations, 20)):  # Sample locations for efficiency
                    perturbed_features = features.clone()

                    noise_std = 0.1
                    perturbed_features[:, :, loc_idx, :] += torch.randn_like(perturbed_features[:, :, loc_idx, :]) * noise_std
                    output_perturbed, _ = model(perturbed_features, edge_index)
                    sensitivity = torch.mean(torch.abs(output_original - output_perturbed))
                    feature_importance[loc_idx] = sensitivity.item()
                
                break 
                
            except Exception as e:
                print('e')
                continue
    
    # Interpolate for unsampled locations
    if feature_importance.sum() > 0:
        nonzero_indices = torch.nonzero(feature_importance).flatten()
        if len(nonzero_indices) > 0:
            mean_importance = feature_importance[nonzero_indices].mean()
            feature_importance[feature_importance == 0] = mean_importance * 0.5
    return feature_importance

def calculate_gradient_importance(model, train_loader, edge_index, n_locations):
    
    model.train()  # Enable gradients
    gradient_importance = torch.zeros(n_locations)
    
    try:
        for features, targets in train_loader:
            features.requires_grad_(True)
            
            output, _ = model(features, edge_index)
            loss = F.mse_loss(output.squeeze(), targets)
            
            # Calculate gradients with respect to input features
            gradients = torch.autograd.grad(loss, features, retain_graph=True)[0]
            
            # Calculate importance as gradient magnitude for each location
            grad_magnitude = torch.mean(torch.abs(gradients), dim=(0, 1, 3))  # Average over batch, time, features
            gradient_importance = grad_magnitude
            
            break  # Use only first batch
            
    except Exception as e:
        print(f"Error in gradient calculation: {e}")
        gradient_importance = torch.ones(n_locations)
    
    model.eval()  # Back to eval mode
    
    print(f"Gradient importance range: {gradient_importance.min():.6f} - {gradient_importance.max():.6f}")
    return gradient_importance

def combine_importance_scores(attention, removal, centrality, feature, gradient):
    """
    Combine multiple importance scores with weighted average
    """
    print("Combining importance scores...")

    def normalize_scores(scores):
        if scores.max() > scores.min():
            return (scores - scores.min()) / (scores.max() - scores.min())
        else:
            return torch.ones_like(scores) * 0.5
    
    attention_norm = normalize_scores(attention)
    removal_norm = normalize_scores(removal)
    centrality_norm = normalize_scores(centrality)
    feature_norm = normalize_scores(feature)
    gradient_norm = normalize_scores(gradient)

    weights = {
        'attention': 0.25,
        'removal': 0.30,     
        'centrality': 0.20,
        'feature': 0.15,
        'gradient': 0.10
    }
    
    combined = (
        weights['attention'] * attention_norm +
        weights['removal'] * removal_norm +
        weights['centrality'] * centrality_norm +
        weights['feature'] * feature_norm +
        weights['gradient'] * gradient_norm
    )
    
    print(f"Combined importance range: {combined.min():.6f} - {combined.max():.6f}")
    print(f"   Score diversity (std): {combined.std():.6f}")
    
    return combined

def evaluate_model_performance(model, data_loader, edge_index, exclude_node=None):
    model.eval()
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for features, targets in data_loader:
            try:
                # If excluding a node, set its features to zero
                if exclude_node is not None:
                    features[:, :, exclude_node, :] = 0
                
                output, _ = model(features, edge_index)
                loss = F.mse_loss(output.squeeze(), targets)
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                continue
    
    return total_loss / max(batch_count, 1)

def interpolate_importance_scores(scores, tested_indices, n_locations):

    if len(tested_indices) == 0:
        return torch.ones(n_locations)
    tested_mean = scores[tested_indices].mean()
    scores[scores == 0] = tested_mean * 0.7  
    
    return scores
class Enhanced_GAT_LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, n_locations, hidden_dim=64):
        super(Enhanced_GAT_LSTM, self).__init__()
        
        # Multi-head attention with more heads for better differentiation
        self.gat1 = GATConv(in_channels, hidden_dim//2, heads=4, concat=True, dropout=DROPOUT_RATE)
        self.gat2 = GATConv(hidden_dim*2, hidden_dim//2, heads=4, concat=True, dropout=DROPOUT_RATE)
        self.attention_norm = nn.LayerNorm(hidden_dim*2)
        self.lstm = nn.LSTM(hidden_dim*2 * n_locations, 128, num_layers=2, dropout=DROPOUT_RATE, batch_first=False)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, n_locations * out_channels)
        self.dropout = nn.Dropout(DROPOUT_RATE)       
        self.n_locations = n_locations
        self.out_channels = out_channels

    def forward(self, x_sequence, edge_index):
        batch_size, seq_len, n_nodes, n_features = x_sequence.shape
        gat_outputs = []
        attention_weights_list = []

        for t in range(seq_len):
            x_t = x_sequence[:, t, :, :].reshape(-1, n_features)
            gat_out1, attention1 = self.gat1(x_t, edge_index, return_attention_weights=True)
            gat_out1 = F.elu(gat_out1)
            gat_out2, attention2 = self.gat2(gat_out1, edge_index, return_attention_weights=True)
            gat_out2 = F.elu(gat_out2)
            gat_out2 = self.attention_norm(gat_out2)            
            gat_outputs.append(gat_out2.reshape(batch_size, -1))
            attention_weights_list.append(attention2)
        final_attention = attention_weights_list[-1] if attention_weights_list else None
        lstm_input = torch.stack(gat_outputs, dim=0)  # (seq_len, batch_size, features)
        lstm_out, _ = self.lstm(lstm_input)
        linear_out = self.dropout(F.relu(self.linear1(lstm_out[-1])))
        output = self.linear2(linear_out)
        
        return output.reshape(-1, self.n_locations, self.out_channels), final_attention

def main():
    
    def load_data_safely(filename, column_names):
        try:
            sample_df = pd.read_csv(filename, nrows=3)
            first_row = sample_df.iloc[0]
            
            if any(isinstance(val, str) and not str(val).replace('.', '').replace('-', '').isdigit() 
                   for val in first_row):
                df = pd.read_csv(filename)
                if len(df.columns) == len(column_names) and list(df.columns) != column_names:
                    df.columns = column_names
            else:
                df = pd.read_csv(filename, header=None, names=column_names)
                
        except Exception as e:
            try:
                df = pd.read_csv(filename, header=None, names=column_names)
            except Exception as e2:
                print(f"Failed to load file: {e2}")
                exit()
        
        return df
    
    try:
        original_df = load_data_safely(FILENAME, COLUMN_NAMES)
        df = original_df.copy()
        
    except FileNotFoundError:
        print(f"Error: {FILENAME} not found. Please check the file path.")
        exit()
    
    try:
        for data_name, data_df in [("original", original_df), ("working", df)]
            
            data_df['Year'] = pd.to_numeric(data_df['Year'], errors='coerce')
            data_df['Month'] = pd.to_numeric(data_df['Month'], errors='coerce')
            data_df['Day'] = pd.to_numeric(data_df['Day'], errors='coerce')
            
            date_nulls = data_df[['Year', 'Month', 'Day']].isnull().sum()
            if date_nulls.any():
                data_df.dropna(subset=['Year', 'Month', 'Day'], inplace=True)            
            data_df['Year'] = data_df['Year'].astype(int)
            data_df['Month'] = data_df['Month'].astype(int)
            data_df['Day'] = data_df['Day'].astype(int)            
            valid_years = (data_df['Year'] >= 1900) & (data_df['Year'] <= 2030)
            valid_months = (data_df['Month'] >= 1) & (data_df['Month'] <= 12)
            valid_days = (data_df['Day'] >= 1) & (data_df['Day'] <= 31)            
            valid_dates = valid_years & valid_months & valid_days
            invalid_count = (~valid_dates).sum()
            
            if invalid_count > 0:
                data_df = data_df[valid_dates]
            
            data_df[DATE_COLUMN] = pd.to_datetime(data_df[['Year', 'Month', 'Day']], errors='coerce')
            
            date_construction_failed = data_df[DATE_COLUMN].isnull().sum()
            if date_construction_failed > 0:
                data_df.dropna(subset=[DATE_COLUMN], inplace=True)
        
    except Exception as e:
        print(f"Error creating date column: {e}")
        exit()


    missing_features = [feat for feat in FEATURES_TO_USE if feat not in df.columns]
    if missing_features:
        exit()
    
    for feature in FEATURES_TO_USE:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    feature_nulls = df[FEATURES_TO_USE].isnull().sum()
    if feature_nulls.any():
        print(f"Missing values in features: {feature_nulls[feature_nulls > 0]}")
        df = df.dropna(subset=FEATURES_TO_USE)
        print(f"Data shape after dropping missing feature values: {df.shape}")
    
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['lattitude'] = pd.to_numeric(df['lattitude'], errors='coerce')
    df = df.dropna(subset=['longitude', 'lattitude'])
    df = df.sort_values(by=[DATE_COLUMN, 'lattitude', 'longitude']).reset_index(drop=True)
    df['location_id'] = df.groupby(['lattitude', 'longitude']).ngroup()
    location_map = df[['location_id', 'lattitude', 'longitude']].drop_duplicates().set_index('location_id')
    n_locations = len(location_map)
    actual_n_locations_to_select = min(N_LOCATIONS_TO_SELECT, n_locations)
    if n_locations < N_LOCATIONS_TO_SELECT:
        print(f"Warning: Only {n_locations} locations found, but {N_LOCATIONS_TO_SELECT} requested")
        print(f"Adjusting to select {actual_n_locations_to_select} locations")
    scaler = StandardScaler()
    df[FEATURES_TO_USE] = scaler.fit_transform(df[FEATURES_TO_USE])
    features_df = df.pivot(index=DATE_COLUMN, columns='location_id', values=FEATURES_TO_USE)
    target_df = df.pivot(index=DATE_COLUMN, columns='location_id', values=TARGET_VARIABLE)
    if features_df.isnull().any().any():
        features_df = features_df.ffill().bfill()
    
    if target_df.isnull().any().any():
        target_df = target_df.ffill().bfill()
    
    features_np = features_df.values.reshape(len(features_df), n_locations, len(FEATURES_TO_USE))
    target_np = target_df.values
    try:
        nearest_neighbors = NearestNeighbors(n_neighbors=min(N_NEIGHBORS + 1, n_locations), metric='haversine')
        coords = location_map[['lattitude', 'longitude']].values
        
        if np.any(np.isnan(coords)):
            print("Error: NaN values found in coordinates")
            exit()
        
        nearest_neighbors.fit(np.deg2rad(coords))
        distances, indices = nearest_neighbors.kneighbors()
        
        edge_list = []
        for i in range(n_locations):
            for j in indices[i, 1:]:
                if 0 <= i < n_locations and 0 <= j < n_locations and i != j:
                    edge_list.append([i, j])
        
        if not edge_list:
            exit()
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        max_edge_idx = edge_index.max().item()
        if max_edge_idx >= n_locations:
            valid_edge_mask = (edge_index[0] < n_locations) & (edge_index[1] < n_locations)
            edge_index = edge_index[:, valid_edge_mask]
        
    except Exception as e:
        print(f"Error creating graph: {e}")
        exit()

    from torch.utils.data import Dataset, DataLoader
    
    class SpatioTemporalDataset(Dataset):
        def __init__(self, features, targets, lookback):
            self.features = torch.FloatTensor(features)
            self.targets = torch.FloatTensor(targets)
            self.lookback = lookback

        def __len__(self):
            return max(0, len(self.features) - self.lookback)

        def __getitem__(self, idx):
            return (self.features[idx:idx+self.lookback], self.targets[idx+self.lookback])
    
    # Adjust lookback window if necessary
    actual_lookback = min(LOOKBACK_WINDOW, len(features_np) - 1)
    if len(features_np) <= LOOKBACK_WINDOW:
        print(f"Warning: Not enough data for lookback window. Reducing from {LOOKBACK_WINDOW} to {actual_lookback}")
    
    train_dataset = SpatioTemporalDataset(features_np, target_np, actual_lookback)
    print(f"Created dataset with {len(train_dataset)} samples")
    
    if len(train_dataset) == 0:
        print("Error: No training samples available")
        exit()
    
    train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True)
    model = Enhanced_GAT_LSTM(
        in_channels=len(FEATURES_TO_USE), 
        out_channels=1, 
        n_locations=n_locations,
        hidden_dim=64
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        total_loss = 0
        batch_count = 0
        
        for features, targets in train_loader:
            try:
                optimizer.zero_grad()
                output, _ = model(features, edge_index)
                loss = criterion(output.squeeze(), targets)
                l2_lambda = 0.001
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_norm
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch == EPOCHS - 1:
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")
            
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break
    combined_importance = calculate_multiple_importance_scores(
        model, train_loader, edge_index, n_locations, location_map
    )

    importance_df = pd.DataFrame({
        'location_id': range(n_locations),
        'importance_score': combined_importance.numpy()
    })
    
    ranked_df = pd.merge(location_map.reset_index(), importance_df, on='location_id')
    ranked_df = ranked_df.sort_values(by='importance_score', ascending=False)
    
    optimal_network = ranked_df.head(actual_n_locations_to_select)
    print_enhanced_results(optimal_network, ranked_df, combined_importance)

    print(f"\n💾 Saving optimal network data...")
    optimal_location_ids = optimal_network['location_id'].tolist()
    
    # Filter original data for optimal locations
    optimal_data_list = []
    for loc_id in optimal_location_ids:
        if loc_id in location_map.index:
            lat = location_map.loc[loc_id, 'lattitude']
            lon = location_map.loc[loc_id, 'longitude']
            
            # Find data for this location in original dataframe
            tolerance = 1e-6
            lat_match = np.abs(original_df['lattitude'] - lat) < tolerance
            lon_match = np.abs(original_df['longitude'] - lon) < tolerance
            location_data = original_df[lat_match & lon_match].copy()
            
            if len(location_data) > 0:
                optimal_data_list.append(location_data)
    
    if optimal_data_list:
        optimal_data_df = pd.concat(optimal_data_list, ignore_index=True)
        
        # Define columns to save
        columns_to_save = [
            'longitude', 'lattitude', 'Year', 'Month', 'Day', 
            'T2M', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'WD10M', 'WS10M', 'Pres', 
            'date', 'elevation'
        ]
        
        available_columns = [col for col in columns_to_save if col in optimal_data_df.columns]
        final_optimal_data = optimal_data_df[available_columns].copy()
        
        # Clean and sort data
        final_optimal_data = final_optimal_data.dropna(subset=['longitude', 'lattitude'])
        sort_columns = ['Year', 'Month', 'Day', 'lattitude', 'longitude']
        available_sort_columns = [col for col in sort_columns if col in final_optimal_data.columns]
        if available_sort_columns:
            final_optimal_data = final_optimal_data.sort_values(available_sort_columns)
        
        # Save to CSV
        output_filename = 'Enhanced_GNN_optimal_network_data.csv'
        final_optimal_data.to_csv(output_filename, index=False)
        print(f"📍 Unique locations: {final_optimal_data.groupby(['lattitude', 'longitude']).ngroup().nunique()}")
    
    return optimal_network, ranked_df

def print_enhanced_results(optimal_network, ranked_df, importance_scores):
    print(f"\n" + "="*80)
    print("ENHANCED OPTIMAL NETWORK SELECTION RESULTS")
    print("="*80)

    # Score distribution analysis
    high_importance = (importance_scores > importance_scores.quantile(0.8)).sum()
    medium_importance = ((importance_scores > importance_scores.quantile(0.4)) & 
                        (importance_scores <= importance_scores.quantile(0.8))).sum()
    low_importance = (importance_scores <= importance_scores.quantile(0.4)).sum()

    # Top 10 locations - FIX THE FORMAT ERROR HERE
    print(f"\nTOP 10 MOST IMPORTANT LOCATIONS:")
    top_10 = optimal_network.head(10)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        # Convert location_id to int to fix the formatting error
        location_id = int(row['location_id'])
        print(f"   {i:2d}. ID:{location_id:3d} ({row['lattitude']:8.4f}, {row['longitude']:8.4f}) "
              f"Score: {row['importance_score']:.6f}")
    
    print("="*80)

if __name__ == "__main__":
    main()