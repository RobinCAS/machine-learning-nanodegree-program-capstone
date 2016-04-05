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

    # Perform PCA analysis on the data so we can get an idea on the number
    # of principal components, which might help us make a decision on a
    # classifier
    mlnp_capstone.analyze_data()

    # Use of PCA appears to be unnecessary for this problem, as the use of
    # the full feature set is fast enough, and the resulting precision is always
    # higher when using the full feature set  
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

    # Look at the ROC curve for visual comparison of classifiers
    mlnp_capstone.plot_roc()

