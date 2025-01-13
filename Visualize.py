import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

def preprocess_data(X):
    for column in X.columns:
        if X[column].dtype == 'object':
            # Apply LabelEncoder 
            X[column] = LabelEncoder().fit_transform(X[column].astype(str))
        elif X[column].dtype == 'category':
            # Tranform to numeric values
            X[column] = X[column].cat.codes
    # Apply KNN imputation for missing values
    imputer = KNNImputer(n_neighbors=5) 
    X = imputer.fit_transform(X)
    return X


data = {
    'Dataset': [],
    'System_No_Accuracy': [],
    'System_Mod_Accuracy': [],
    'System_No_F1_Score': [],
    'System_Mod_F1_Score': [],
    'System_No_Precision': [],
    'System_Mod_Precision': []
}

# Load all non-linearly separable datasets
linear_sep_df = pd.read_csv("LinearDatasets.csv")
lineardatasets = linear_sep_df[linear_sep_df["Linear_sep"] == 0]

save_path = os.path.expanduser('~/openml_benchmark_data_classification9935/')
os.makedirs(save_path, exist_ok=True)

dataset_dirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]


prop = np.loadtxt(os.path.join(save_path, "total_time_prop.txt"))

# Apply log transformation to create symmetry
log_prop = [np.log(x) if x >= 1 else -np.log(1 / x) for x in prop]


bins = np.arange(-2.5, 2.6, 0.05) 
plt.hist(log_prop, bins=bins, edgecolor='black')  
plt.title('Histogram of Standard/Modified Time Ratios (Log-Scaled)')
plt.xlabel('Log(Proportional Time Difference)')
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(save_path,"LogScaledTimePropPlot.pdf"), format="pdf", dpi=300)
plt.show()

for dataset_dir in lineardatasets["Dataset"]:
    print(dataset_dir)

    # Construct file paths
    features_path = os.path.join(save_path, dataset_dir, 'features.csv')
    target_path = os.path.join(save_path, dataset_dir, 'target.csv')
    if dataset_dir == "dataset_40927_CIFAR_10" or dataset_dir == 'dataset_38_sick' or dataset_dir == "dataset_469_analcatdata_dmft":
        continue
    try:
        # Load and preprocess data
        X = preprocess_data(pd.read_csv(features_path))
        y = pd.read_csv(target_path).values.ravel()
    except Exception as e:
        print(f"Error loading data for {dataset_dir}: {e}")
        continue
    try:
        no_mod_perf = pd.read_csv(os.path.join(save_path, dataset_dir, 'no_mod.csv'))
        mod_perf = pd.read_csv(os.path.join(save_path, dataset_dir, 'mod.csv'))
    except Exception as e:
        print(f"Error reading performance metrics for {dataset_dir}: {e}")
        continue

    try:
        # Extract accuracy
        no_mod_acc = no_mod_perf[no_mod_perf.iloc[:, 0] == 'accuracy']['precision'].values[0]
        mod_acc = mod_perf[mod_perf.iloc[:, 0] == 'accuracy']['precision'].values[0]

        # Extract f1-score from the "weighted avg" row
        no_mod_f1 = no_mod_perf[no_mod_perf.iloc[:, 0] == 'weighted avg']['f1-score'].values[0]
        mod_f1 = mod_perf[mod_perf.iloc[:, 0] == 'weighted avg']['f1-score'].values[0]

        # Extract f1-score from the "weighted avg" row
        no_mod_precision = no_mod_perf[no_mod_perf.iloc[:, 0] == 'weighted avg']['precision'].values[0]
        mod_precision = mod_perf[mod_perf.iloc[:, 0] == 'weighted avg']['precision'].values[0]
    except IndexError:
        print(f"Required metrics missing for {dataset_dir}")
        continue

    # Append data to the dictionary
    data['Dataset'].append(dataset_dir)
    data['System_No_Accuracy'].append(no_mod_acc)
    data['System_Mod_Accuracy'].append(mod_acc)
    data['System_No_F1_Score'].append(no_mod_f1)
    data['System_Mod_F1_Score'].append(mod_f1)
    data['System_No_Precision'].append(no_mod_precision)
    data['System_Mod_Precision'].append(mod_precision)

# Create and save dataframe with performance per file
df = pd.DataFrame(data)
df.to_csv(os.path.join(save_path, "PerdatasetPerformance.csv"), index = False)

# Count points above and below the line for Accuracy
above_accuracy = (df['System_Mod_Accuracy'] > df['System_No_Accuracy']).sum()
below_accuracy = (df['System_Mod_Accuracy'] < df['System_No_Accuracy']).sum()

plt.rcParams.update({'font.size': 22})

# Plot accuracy comparison with annotations
plt.figure(figsize=(8, 8))
plt.scatter(df['System_No_Accuracy'], df['System_Mod_Accuracy'], s=10, color='blue', label='Accuracy')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Baseline (Equal Performance)')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Accuracy of System (No Mod)')
plt.ylabel('Accuracy of System (Mod)')
plt.title(f'Accuracy Difference (Mod-Standard) \nAbove Line: {above_accuracy}, Below Line: {below_accuracy}')
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_path,"AccuracyPlot.pdf"), format="pdf", dpi=300)
plt.show()

# Calculate the accuracy difference
df['Accuracy_Difference'] = df['System_Mod_Accuracy'] - df['System_No_Accuracy']
df['F1_Difference'] = df['System_Mod_F1_Score'] - df['System_No_F1_Score']
df['Preci_Difference'] = df['System_Mod_Precision'] - df['System_No_Precision']



# Plot the distribution of the accuracy differences
bin_edges = np.linspace(-0.25, 0.25, 40)
plt.figure(figsize=(10, 6))
plt.hist(df['Accuracy_Difference'], bins=bin_edges, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=1, label="Baseline (No Difference)")
plt.ylim(0, 30)
plt.title("Distribution of Accuracy Difference (Mod - No)", fontsize=26)
plt.xlabel("Accuracy Difference", fontsize=22)
plt.ylabel("Frequency", fontsize=22)
plt.legend(fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "AccuracyDist.pdf"), format="pdf", dpi=300)
# Display the plot
plt.show()

# Count points above and below the line for F1-Score
above_f1 = (df['System_Mod_F1_Score'] > df['System_No_F1_Score']).sum()
below_f1 = (df['System_Mod_F1_Score'] < df['System_No_F1_Score']).sum()



# Plot f1-score comparison with annotations
plt.figure(figsize=(8, 8))
plt.scatter(df['System_No_F1_Score'], df['System_Mod_F1_Score'], s=10, color='green', label='F1-Score')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Baseline (Equal Performance)')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('F1-Score of System (No Mod)')
plt.ylabel('F1-Score of System (Mod)')
plt.title(f'F1-Score Difference (Mod-Standard) \nAbove Line: {above_f1}, Below Line: {below_f1}')
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_path, "F1ScoresPlot.pdf"), format="pdf", dpi=300)
plt.show()


# Count points above and below the line for F1-Score
above_f1 = (df['System_Mod_Precision'] > df['System_No_Precision']).sum()
below_f1 = (df['System_Mod_Precision'] < df['System_No_Precision']).sum()

# Plot f1-score comparison with annotations
plt.figure(figsize=(8, 8))
plt.scatter(df['System_No_Precision'], df['System_Mod_Precision'], s=10, color='green', label='Precision')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Baseline (Equal Performance)')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Precision of System (No Mod)')
plt.ylabel('Precision of System (Mod)')
plt.title(f'Precision Difference (Mod-Standard) \nAbove Line: {above_f1}, Below Line: {below_f1}')
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_path,"PrecisionPlot.pdf"), format="pdf", dpi=300)
plt.show()


# Calculate the accuracy difference
df['Precision_Difference'] = df['System_Mod_Precision'] - df['System_No_Precision']

plt.figure(figsize=(10, 6))
plt.hist(df['Precision_Difference'], bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=1, label="Baseline (No Difference)")
plt.title("Distribution of Precision Difference (Mod - No)", fontsize=14)
plt.xlabel("Precision Difference", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(save_path,"PrecisionDist.pdf"), format="pdf", dpi=300)
plt.show()
