from MLNPCapstone import MLNPCapstone

if __name__ == "__main__":
   
    # Load the data from the file, and do any necessary preprocessing on it 
    mlnp_capstone = MLNPCapstone(
                            target_name = 'class', 
                            positive_label = 'g', 
                            negative_label = 'h',
                            pca_max_n = 8, 
    )
    mlnp_capstone.load_and_prepare_data('data/magic_gamma_telescope.csv')

    # Analyze the data using PCA
    mlnp_capstone.analyze_data()
 
    mlnp_capstone.set_use_pca(False)
    
    # Try a Random Forest Classifier
    mlnp_capstone.set_learner('rfc')
    mlnp_capstone.run(verbose=True)

    # Try a Logistic Regression Classifier
    mlnp_capstone.set_learner('logistic')
    mlnp_capstone.run(verbose=True)
    
    # Try a Support Vector Machine
    mlnp_capstone.set_learner('svc')
    mlnp_capstone.run(verbose=True)

    # Try a K-Nearest Neighbors Classifier
    mlnp_capstone.set_learner('knc')
    mlnp_capstone.run(verbose=True)

