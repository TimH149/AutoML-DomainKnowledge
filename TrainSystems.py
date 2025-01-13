# Prevents recursive imports and errors by Autosklearn
# Ensures proper execution of multiprocessing code
if __name__ == "__main__": 
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import KNNImputer
    import autosklearn.classification
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import time 


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

    def evaluate_model(model, x_test, y_test):  
        y_pred= model.predict(x_test)

        # Generate the classification report as a dictionary
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Convert the dictionary to a DataFrame
        report_df = pd.DataFrame(report_dict).transpose()

        alg_weight = model.leaderboard()[["type","ensemble_weight"]]
        
        result = pd.concat([report_df, alg_weight])

        return result

    def plot_convergence(automl, save_path, modified):
        a = automl.leaderboard(ensemble_only= False, detailed = True, sort_by = "start_time")

        # Cost is loss on validation dataset
        a['accuracy'] = 1 - a['cost']
        a['accuracy_train'] = 1 - a['train_loss']

        # Get cumulative max accuracy scores
        a['cumulative_max_accuracy'] = a['accuracy'].cummax()
        a['cumulative_max_accuracy_train'] = a['accuracy_train'].cummax()

        # Plot the cumulative maximum accuracy (validation and training) over time
        plt.figure(figsize=(10, 6))

        # Cumulative maximum accuracy on validation data
        plt.plot(a['start_time'], a['cumulative_max_accuracy'], marker='o'
        ,label='Cumulative Max Accuracy (Validation)')

        # Cumulative maximum accuracy on training data
        plt.plot(a['start_time'], a['cumulative_max_accuracy_train'], marker='x'
        ,linestyle='--', label='Cumulative Max Accuracy (Training)')

        # Annotate each point with its corresponding type
        for i, row in a.iterrows():
            plt.text(
                rotation=-45,
                x=row['start_time'],
                y=row['cumulative_max_accuracy'],
                s=row['type'],
                fontsize=9,
                ha='right'  # Horizontal alignment
            )

        # Formatting the plot
        if modified:
            plt.title('Cumulative Maximum Accuracy (Validation and Training) Over Time MODIFIED VERSION', fontsize=14)
        else:
            plt.title('Cumulative Maximum Accuracy (Validation and Training) Over Time STANDARD VERSION', fontsize=14)

        plt.xlabel('Start Time', fontsize=12)
        plt.ylabel('Cumulative Max Accuracy', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)


    # Load all classifiers and their information
    classification_models = pd.read_csv("ThesisAlgoInfoClassifiers.csv", delimiter = ";")

    # Select which models are linear and non-linear
    df_linear_models = classification_models[classification_models["linear_model"] == 1]
    df_non_linear_models = classification_models[classification_models["linear_model"] == 0]

    #Convert dataframes to usable list
    linear_models = list(df_linear_models["algorithm"])
    non_linear_models = list(df_non_linear_models["algorithm"])

    # Load all non-linearly separable datasets
    linear_sep_df = pd.read_csv("LinearDatasets.csv")
    lineardatasets = linear_sep_df[linear_sep_df["Linear_sep"] == 0]


    # Define a path to save the datasets
    save_path = os.path.expanduser('~/openml_benchmark_data_classification99/')
    os.makedirs(save_path, exist_ok=True)
    dataset_dirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]



    # Set smac arguments for Auto-sklearn
    smac_scenario_args = {
        'runcount-limit': 200,             # Limit to 200 configurations
        "initial_incumbent": "RANDOM", 
    }
    total_time_prop = []
    for dataset_dir in lineardatasets["Dataset"]:
        print(dataset_dir)

        # Load features and targets
        features_path = os.path.join(save_path, dataset_dir, 'features.csv')
        target_path = os.path.join(save_path, dataset_dir, 'target.csv')

        X = preprocess_data(pd.read_csv(features_path))
        y = pd.read_csv(target_path).values.ravel()

        # Modified Auto-sklearn system
        automl_mod = autosklearn.classification.AutoSklearnClassifier(
            include={"classifier": non_linear_models, "feature_preprocessor": ['no_preprocessing']},
            smac_scenario_args=smac_scenario_args,
            per_run_time_limit=40,  # Time limit for each model
            initial_configurations_via_metalearning=0,
            metadata_directory=None,
            dataset_compression=False,  # Skip dataset compression
            memory_limit=50000,
            ensemble_kwargs = {"ensemble_size" : 1},    
            delete_tmp_folder_after_terminate=False,
            n_jobs=14,  # Use all threads
            seed=42, # For reproducibility
        )

        # Standard Auto-sklearn system
        automl_no_mod = autosklearn.classification.AutoSklearnClassifier(
            smac_scenario_args=smac_scenario_args,
            include={"feature_preprocessor": ['no_preprocessing']},
            per_run_time_limit=40,  # Time limit for each model
            initial_configurations_via_metalearning=0,
            metadata_directory=None,
            dataset_compression=False,  # Skip dataset compression
            memory_limit=50000,
            ensemble_kwargs = {"ensemble_size" : 1},
            delete_tmp_folder_after_terminate=False,
            n_jobs=14,  # Use all threads
            seed=42,  # For reproducibility
        )

        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # train = 0.8 & test = 0.2
    
        # Train and get runtime of Standard system
        start_time = time.time()    
        automl_no_mod.fit(X_train, y_train, X_test, y_test)
        end_time = time.time()

        # Total time
        elapsed_time_no_mod = end_time - start_time
        print(f"Training time cost of Standard system: {elapsed_time_no_mod:.2f} seconds.")

        # Train and get runtime of Modifed system
        start_time = time.time()
        automl_mod.fit(X_train, y_train, X_test, y_test)
        end_time = time.time()

        # Total time
        elapsed_time_mod = end_time - start_time
        print(f"Training time cost of Modified System: {elapsed_time_mod:.2f} seconds.")

        proportion = elapsed_time_no_mod/elapsed_time_mod
        total_time_prop.append(proportion)


        # Evaluate models
        y_pred_no_mod = automl_no_mod.predict(X_test)
        y_pred_mod = automl_mod.predict(X_test)

        accuracy_no_mod = accuracy_score(y_test, y_pred_no_mod)
        accuracy_mod = accuracy_score(y_test, y_pred_mod)

        print(f"Dataset: {dataset_dir}")
        print(f"  Proportional time is:  {proportion}")
        print(f"  No modification model accuracy: {accuracy_no_mod:.4f}")
        print(f"  Modified model accuracy: {accuracy_mod:.4f}")



        no_mod = evaluate_model(automl_no_mod, X_test, y_test)
        mod = evaluate_model(automl_mod, X_test, y_test)


        save_no_mod = os.path.join(save_path, dataset_dir, 'no_mod.csv')
        save_mod = os.path.join(save_path, dataset_dir, 'mod.csv')

        no_mod_image_path = os.path.join(save_path, dataset_dir, 'no_mod.png')
        mod_image_path = os.path.join(save_path, dataset_dir, 'no_mod.png')
        
        plot_convergence(automl_no_mod, no_mod_image_path, modified=0)
        plot_convergence(automl_mod, mod_image_path, modified=1)
            
        no_mod.to_csv(save_no_mod, index = True)
        mod.to_csv(save_mod, index = True)

        # Retrieve cv_results_
        cv_results = automl_no_mod.cv_results_
        cv_results_mod = automl_mod.cv_results_

        # Convert cv_results_ to a pandas DataFrame for easier analysis
        df = pd.DataFrame(cv_results)
        df_mod = pd.DataFrame(cv_results_mod)

        df.to_csv(os.path.join(save_path, dataset_dir, 'cv_standard.csv'))
        df_mod.to_csv(os.path.join(save_path, dataset_dir, 'cv_mod.csv'))
        
        # Retrieve leaderboard
        mod = automl_mod.leaderboard(ensemble_only= False
        , include=  {"train_loss","rank","model_id","type","cost","duration","start_time"}
        , sort_by = "start_time"
        )

        no_mod = automl_no_mod.leaderboard(ensemble_only= False
        , include=  {"train_loss","rank","model_id","type","cost","duration","start_time"}
        , sort_by = "start_time"
        )

        mod.to_csv(os.path.join(save_path, dataset_dir, 'Leader_mod.csv'))
        no_mod.to_csv(os.path.join(save_path, dataset_dir, 'Leader_no_mod.csv'))

        print("Standard: " + str(len(df['param_classifier:__choice__'])))
        print("Modified: " + str(len(df_mod['param_classifier:__choice__'])))

    np.array(total_time_prop)
    # Save to a text file
    with open(os.path.join(save_path, "total_time_prop.txt"), 'w') as file:
        for item in total_time_prop:
            file.write(f"{item}\n")  # Write each item on a new line
