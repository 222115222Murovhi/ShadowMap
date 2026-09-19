import os
import numpy as np

def clean_and_split_dataset(data_directory):
    print("🚀 Initialising ShadowMap Rigorous Preprocessing Pipeline...")
    
    csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    if not csv_files:
        #Stop the program if no CSV files are found in the specified directory
        raise FileNotFoundError(f" No CSV files found in directory: {data_directory}")
        
    all_features = [] # Will eventually store every feature row from all CSVs
    all_labels = [] # Will store the class labels corresponding to each feature row
    header = None # Flag
    
    for file_name in csv_files:
        file_path = os.path.join(data_directory, file_name) # Build the full path to the CSV file
        print(f" └── Parsing: {file_name}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if header is None:
                header = [col.strip() for col in lines[0].split(',')]
            
            for line in lines[1:]:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) != len(header): # checking if every row has the correct number of cols
                    continue                  # skip if it doesn't
                all_features.append(parts[:-1]) 
                all_labels.append(parts[-1])    

    raw_x = np.array(all_features, dtype=object) #Converting the feature list into a Numpy array of type Object to allow temporary storage of strings eg "Infinity" and "NaN" before we clean them up
    raw_y = np.array(all_labels)
    
    print(f"📊 Raw extraction complete. Found {raw_x.shape[0]} total rows.")

    # Drop Identifiers: Flow ID (0), Source IP (1), Src Port (2), Dst IP (3), Timestamp (6)
    columns_to_drop = [0, 1, 2, 3, 6]
    keep_indices = [i for i in range(raw_x.shape[1]) if i not in columns_to_drop]
    cleaned_header = [header[i] for i in keep_indices]
    raw_x = raw_x[:, keep_indices]

    # Process and drop rows containing mathematical anomalies
    x_floats = np.zeros(raw_x.shape, dtype=np.float32)
    valid_row_mask = np.ones(raw_x.shape[0], dtype=bool)
    
    for row_idx in range(raw_x.shape[0]):
        try:
            row_data = raw_x[row_idx, :]
            # FIX: If 'Infinity', 'inf', 'NaN' or empty strings exist, flag the row for immediate removal
            if np.any((row_data == 'Infinity') | (row_data == 'inf') | (row_data == 'NaN') | (row_data == '')):
                valid_row_mask[row_idx] = False
                continue
                
            x_floats[row_idx, :] = row_data.astype(np.float32) # converts evry value into a float
        except ValueError:
            valid_row_mask[row_idx] = False # if conversion fails, the row is markedinvalid

    # Apply row filter to completely preserve true scaling distributions
    x_floats = x_floats[valid_row_mask] # we keep only the rows marked true
    raw_y = raw_y[valid_row_mask] # same thing with the labels
    print(f"🧹 Safely dropped mathematical outliers. Retained {x_floats.shape[0]} rows.")

    # Drop constant columns (Variance close to 0), how:
    # We Compute the variane for each column(feature)
    # ▪️If High varience -> values change a lot -> keep the cloumn
    # ▪️If Low varience/ Near Zero -> values are almost the same -> drop the column
    variances = np.var(x_floats, axis=0)
    good_Var_indices = np.where(variances > 1e-5)[0]
    x_floats = x_floats[:, good_Var_indices]
    final_features_list = [cleaned_header[i] for i in good_Var_indices]

    # --- THE FIXED 3-WAY SPLIT LOGIC WITH ATTACK SHUFFLING ---
    print("🔀 Constructing Rigorous Train/Validation/Test Splits...")
    
    is_benign = (raw_y == 'BENIGN')
    x_benign_all = x_floats[is_benign] # only benign rows
    
    x_attacks_all = x_floats[~is_benign] # only attack rows(Not Benign rows)
    y_attacks_all = raw_y[~is_benign]

    # Seed for mathematical reproducibility across all runs
    np.random.seed(42) 

    # 1. Shuffle Benign Data
    shuffled_benign_indices = np.random.permutation(len(x_benign_all))
    x_benign_all = x_benign_all[shuffled_benign_indices]

    # Partition Benign: 70% Training, 15% Validation, 15% Testing
    num_benign = len(x_benign_all)
    train_end = int(num_benign * 0.70)
    val_end = int(num_benign * 0.85)

    X_train_raw = x_benign_all[:train_end]
    X_val_benign = x_benign_all[train_end:val_end]
    X_test_benign = x_benign_all[val_end:]

    # 2. FIX: Completely shuffle attack data to prevent temporal bias
    num_attacks = len(x_attacks_all)
    shuffled_attack_indices = np.random.permutation(num_attacks)
    x_attacks_all = x_attacks_all[shuffled_attack_indices]
    y_attacks_all = y_attacks_all[shuffled_attack_indices]

    # Split Attacks 50/50 between Validation and Testing sets
    attack_split = num_attacks // 2
    X_val_attack = x_attacks_all[:attack_split]
    Y_val_attack = y_attacks_all[:attack_split]

    X_test_attack = x_attacks_all[attack_split:]
    Y_test_attack = y_attacks_all[attack_split:]

    # Combine partitions into unified, balanced Val and Test arrays
    X_val_raw = np.vstack([X_val_benign, X_val_attack])
    Y_val_raw = np.concatenate([np.array(['BENIGN'] * len(X_val_benign)), Y_val_attack])

    X_test_raw = np.vstack([X_test_benign, X_test_attack])
    Y_test_raw = np.concatenate([np.array(['BENIGN'] * len(X_test_benign)), Y_test_attack])

    # MinMax Scale everything using the Training boundaries ONLY
    x_min = np.min(X_train_raw, axis=0)
    x_max = np.max(X_train_raw, axis=0)
    range_denominator = np.where((x_max - x_min) == 0, 1.0, x_max - x_min)
    
    X_train = (X_train_raw - x_min) / range_denominator
    X_val = (X_val_raw - x_min) / range_denominator
    X_test = (X_test_raw - x_min) / range_denominator

    # Map text descriptions to numeric vector states (0 = Benign, 1 = Attack)
    Y_val = np.where(Y_val_raw == 'BENIGN', 0, 1)
    Y_test = np.where(Y_test_raw == 'BENIGN', 0, 1)

    print("✅ Preprocessing Pipeline Completed Successfully!")
    print(f" ├── X_train (Pure Normal for Learning): {X_train.shape}")
    print(f" ├── X_val   (Mixed Normal/Attack for Threshold Tuning): {X_val.shape}")
    print(f" └── X_test  (Mixed Normal/Attack for Final Metrics): {X_test.shape}")
    
    return X_train, X_val, Y_val, X_test, Y_test, final_features_list