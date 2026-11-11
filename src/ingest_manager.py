"""Multi-Lecture Course Ingestion and Knowledge Graph Construction Manager.

Orchestrates the end-to-end ingestion pipeline across multiple lecture PDFs:
1. Extraction & Cleaning (PyMuPDF + cleaner)
2. Natural boundary Chunking & Embedding (all-MiniLM-L6-v2)
3. ChromaDB Vector Storage
4. Knowledge Graph Construction (Course -> Lecture -> Topic -> Chunk <-> Equation)
5. JSON persistence to storage/graph/
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from .extractor import extract_pdf
    from .cleaner import clean_pages
    from .chunker import chunk_pages
    from .embedder import Embedder, get_default_embedder
    from .vector_store import VectorStore
    from .graph_store import CourseGraph
    from .equation_parser import Equation, EquationStore
except ImportError:
    from extractor import extract_pdf
    from cleaner import clean_pages
    from chunker import chunk_pages
    from embedder import Embedder, get_default_embedder
    from vector_store import VectorStore
    from graph_store import CourseGraph
    from equation_parser import Equation, EquationStore


# Predefined syllabus hierarchy and topic mapping for MTH441
MTH441_LECTURE_METADATA: Dict[str, Dict[str, Any]] = {
    "lecture01.pdf": {
        "lecture_id": "MTH441_L01",
        "lecture_num": 1,
        "chapter": 1,
        "title": "Introduction to Linear Models and Distribution Theory",
        "description": "Probability distributions, multivariate normal, chi-square, t, F distributions, quadratic forms, and model linearity criteria.",
        "topics": [
            ("MTH441_L01_TOPIC_intro", "Introduction to Regression Analysis", (1, 8)),
            ("MTH441_L01_TOPIC_multivariate_normal", "Multivariate Normal Distribution", (9, 9)),
            ("MTH441_L01_TOPIC_chi_square", "Chi-Square and Noncentral Chi-Square Distributions", (9, 10)),
            ("MTH441_L01_TOPIC_t_and_f", "Student's t and Snedecor's F Distributions", (11, 12)),
            ("MTH441_L01_TOPIC_quad_forms", "Independence of Linear and Quadratic Forms", (13, 13)),
            ("MTH441_L01_TOPIC_linear_criteria", "Linear Models and Parameter Linearity Criteria", (14, 20)),
        ],
    },
    "lecture02.pdf": {
        "lecture_id": "MTH441_L02",
        "lecture_num": 2,
        "chapter": 2,
        "title": "Simple Linear Regression Analysis",
        "description": "Classical simple linear regression, OLS parameter estimation, variance of estimators, Gauss-Markov theorem, hypothesis testing, and goodness of fit.",
        "topics": [
            ("MTH441_L02_TOPIC_slr_model", "Simple Linear Regression Model Formulation and Assumptions", (1, 4)),
            ("MTH441_L02_TOPIC_ols_estimation", "Ordinary Least Squares Estimation in SLR", (5, 7)),
            ("MTH441_L02_TOPIC_estimator_properties", "Sampling Properties and Variances of OLS Estimators", (8, 11)),
            ("MTH441_L02_TOPIC_gauss_markov", "Gauss-Markov Theorem and BLUE Properties in SLR", (12, 14)),
            ("MTH441_L02_TOPIC_mle", "Maximum Likelihood Estimation in SLR", (15, 15)),
            ("MTH441_L02_TOPIC_slr_inference", "Hypothesis Testing and Confidence Intervals in SLR", (16, 24)),
            ("MTH441_L02_TOPIC_r_squared", "Goodness of Fit and Coefficient of Determination (R^2)", (25, 28)),
            ("MTH441_L02_TOPIC_alt_regression", "Direct, Reverse, and Orthogonal Regression Approaches", (29, 41)),
        ],
    },
    "lecture03.pdf": {
        "lecture_id": "MTH441_L03",
        "lecture_num": 3,
        "chapter": 3,
        "title": "Multiple Linear Regression Model",
        "description": "Matrix formulation of MLR, normal equations, hat matrix projection, Gauss-Markov theorem in matrix form, ANOVA, general linear hypothesis, and GLS/WLS.",
        "topics": [
            ("MTH441_L03_TOPIC_mlr_model", "Multiple Linear Regression Model in Matrix Notation", (1, 4)),
            ("MTH441_L03_TOPIC_normal_equations", "Matrix Normal Equations and Least Squares Estimator", (5, 7)),
            ("MTH441_L03_TOPIC_hat_matrix", "Hat Matrix, Orthogonal Projections, and Residuals", (8, 10)),
            ("MTH441_L03_TOPIC_mlr_properties", "Properties of MLR Estimators and Error Variance Estimation", (11, 15)),
            ("MTH441_L03_TOPIC_gauss_markov_mlr", "Gauss-Markov Theorem in Matrix Form", (16, 18)),
            ("MTH441_L03_TOPIC_general_linear_hypothesis", "Testing General Linear Hypotheses and ANOVA in MLR", (19, 33)),
            ("MTH441_L03_TOPIC_r_squared_mlr", "Coefficient of Determination (R^2) and Adjusted R^2 in MLR", (34, 40)),
            ("MTH441_L03_TOPIC_gls", "Generalized and Weighted Least Squares (GLS/WLS)", (41, 44)),
        ],
    },
    "lecture04.pdf": {
        "lecture_id": "MTH441_L04",
        "lecture_num": 4,
        "chapter": 4,
        "title": "Model Adequacy Checking & Residual Analysis",
        "description": "Methods for checking regression assumptions, residual analysis, standardized and studentized residuals, R-student, PRESS residuals, residual plots, and Box-Cox transformations.",
        "topics": [
            ("MTH441_L04_TOPIC_assumptions", "Regression Assumptions and Residual Definition", (1, 5)),
            ("MTH441_L04_TOPIC_residuals_scaling", "Standardized and Studentized Residuals", (6, 8)),
            ("MTH441_L04_TOPIC_press_rstudent", "PRESS Residuals and R-Student Outlier Detection", (9, 13)),
            ("MTH441_L04_TOPIC_residual_plots", "Residual Plots and Graphical Model Checking", (14, 23)),
            ("MTH441_L04_TOPIC_partial_plots", "Partial Regression and Partial Residual Plots", (24, 28)),
            ("MTH441_L04_TOPIC_lack_of_fit", "Near Neighbors and Testing Lack of Fit", (29, 37)),
            ("MTH441_L04_TOPIC_transformations", "Box-Cox Power Transformations and Box-Tidwell Procedure", (38, 43)),
        ],
    },
    "lecture05.pdf": {
        "lecture_id": "MTH441_L05",
        "lecture_num": 5,
        "chapter": 5,
        "title": "Diagnostics for Leverage and Influence",
        "description": "Identification of influential observations and leverage points, Cook's distance, DFFITS, DFBETAS, and COVRATIO diagnostics.",
        "topics": [
            ("MTH441_L05_TOPIC_leverage", "Leverage Points and Hat Matrix Diagnostics", (1, 3)),
            ("MTH441_L05_TOPIC_cooks_distance", "Cook's Distance and Overall Influence", (4, 5)),
            ("MTH441_L05_TOPIC_dfbetas_dffits", "DFBETAS and DFFITS Deletion Measures", (6, 9)),
            ("MTH441_L05_TOPIC_covratio", "COVRATIO and Precision of Estimates", (10, 10)),
        ],
    },
    "lecture06.pdf": {
        "lecture_id": "MTH441_L06",
        "lecture_num": 6,
        "chapter": 6,
        "title": "Multicollinearity, Ridge Regression & Principal Components",
        "description": "Sources and consequences of multicollinearity, diagnostics via VIF and condition numbers, Ridge regression estimator, and Principal Component Regression.",
        "topics": [
            ("MTH441_L06_TOPIC_multicollinearity_intro", "Nature, Sources, and Consequences of Multicollinearity", (1, 8)),
            ("MTH441_L06_TOPIC_vif_and_condition", "Variance Inflation Factors (VIF) and Condition Numbers", (9, 16)),
            ("MTH441_L06_TOPIC_pcr", "Principal Component Regression (PCR)", (17, 19)),
            ("MTH441_L06_TOPIC_ridge_regression", "Ridge Regression Estimator, Properties, and Ridge Trace", (20, 24)),
        ],
    },
}

# Predefined syllabus hierarchy and topic mapping for CS771 (Introduction to Machine Learning)
CS771_LECTURE_METADATA: Dict[str, Dict[str, Any]] = {
    "lecture00.pdf": {
        "lecture_id": "CS771_L00",
        "lecture_num": 0,
        "chapter": 0,
        "title": "Maths Refresher: Linear Algebra and Probability",
        "description": "Foundational mathematics for machine learning: vectors, matrices, transposes, inner products, random variables, independence, marginals, and multivariate Gaussian distributions.",
        "topics": [
            ("CS771_L00_TOPIC_linear_algebra", "Basics of Linear Algebra and Vector Spaces", (1, 15)),
            ("CS771_L00_TOPIC_probability_basics", "Basics of Probability and Random Variables", (16, 35)),
            ("CS771_L00_TOPIC_gaussian_distributions", "Univariate and Multivariate Gaussian Distributions", (36, 55)),
        ],
    },
    "lecture01.pdf": {
        "lecture_id": "CS771_L01",
        "lecture_num": 1,
        "chapter": 1,
        "title": "Course Logistics and Introduction to Machine Learning",
        "description": "Overview of machine learning, AI revolution, machine learning workflow, supervised vs unsupervised learning, and course logistics.",
        "topics": [
            ("CS771_L01_TOPIC_intro_ai", "The AI Revolution and ML Landscape", (1, 15)),
            ("CS771_L01_TOPIC_ml_workflow", "The Machine Learning Workflow and Paradigms", (16, 29)),
        ],
    },
    "lecture02.pdf": {
        "lecture_id": "CS771_L02",
        "lecture_num": 2,
        "chapter": 2,
        "title": "Data, Features, and Learning with Distances and Similarities",
        "description": "Feature representations, feature spaces, labels, distance and similarity metrics, and the Learning with Prototypes (LwP) classifier.",
        "topics": [
            ("CS771_L02_TOPIC_features_labels", "Feature Representations and Label Types", (1, 10)),
            ("CS771_L02_TOPIC_distances_lwp", "Distances, Similarities, and Learning with Prototypes (LwP)", (11, 22)),
        ],
    },
    "lecture03.pdf": {
        "lecture_id": "CS771_L03",
        "lecture_num": 3,
        "chapter": 3,
        "title": "Evaluating ML Models (Cross-Validation) and Nearest Neighbors",
        "description": "Model validation techniques, train/val/test splits, K-fold cross-validation, hyperparameter tuning, and K-Nearest Neighbors (KNN) classification/regression.",
        "topics": [
            ("CS771_L03_TOPIC_cross_validation", "Evaluation and K-Fold Cross-Validation", (1, 10)),
            ("CS771_L03_TOPIC_knn", "K-Nearest Neighbors (KNN) Classifier and Regression", (11, 24)),
        ],
    },
    "lecture04.pdf": {
        "lecture_id": "CS771_L04",
        "lecture_num": 4,
        "chapter": 4,
        "title": "Decision Trees: Divide, Conquer, and Ensemble",
        "description": "Decision tree induction, impurity and homogeneity metrics, Information Gain, entropy, continuous vs categorical splits, and tree ensembles.",
        "topics": [
            ("CS771_L04_TOPIC_tree_induction", "Decision Tree Construction and Purity Criteria", (1, 10)),
            ("CS771_L04_TOPIC_entropy_info_gain", "Entropy and Information Gain Splitting", (11, 18)),
            ("CS771_L04_TOPIC_ensembles", "Decision Tree Ensembles and Forest Overview", (19, 22)),
        ],
    },
    "lecture05.pdf": {
        "lecture_id": "CS771_L05",
        "lecture_num": 5,
        "chapter": 5,
        "title": "Linear Models: Simple Predictors, Universal Building Blocks",
        "description": "Formulation of linear models for regression, linear feature transformations, squared error loss, residuals, and parameter estimation via optimization.",
        "topics": [
            ("CS771_L05_TOPIC_linear_formulation", "Linear Models and Feature Transformations", (1, 8)),
            ("CS771_L05_TOPIC_linear_estimation", "Loss Formulation and Parameter Optimization in Linear Regression", (9, 16)),
        ],
    },
    "lecture06.pdf": {
        "lecture_id": "CS771_L06",
        "lecture_num": 6,
        "chapter": 6,
        "title": "Solving Optimization Problems in Machine Learning",
        "description": "Calculus in higher dimensions, gradients, Hessians, unconstrained optimization, Gradient Descent update rules, step sizes, and learning rates.",
        "topics": [
            ("CS771_L06_TOPIC_gradients_calculus", "Multivariate Derivatives, Gradients, and Hessians", (1, 8)),
            ("CS771_L06_TOPIC_gradient_descent", "Gradient Descent Updates and Learning Rate Dynamics", (9, 18)),
            ("CS771_L06_TOPIC_constrained_opt", "Constrained Optimization and Mini-Batch Principles", (19, 24)),
        ],
    },
    "lecture07.pdf": {
        "lecture_id": "CS771_L07",
        "lecture_num": 7,
        "chapter": 7,
        "title": "The Linear Models Zoo: Losses and Regularization",
        "description": "Regularized empirical risk minimization, norm-based regularization, Ridge Regression (L2), Lasso Regression (L1), sparsity, and primal-dual perspectives.",
        "topics": [
            ("CS771_L07_TOPIC_losses_regularization", "Loss Functions and Norm-Based Regularization", (1, 5)),
            ("CS771_L07_TOPIC_ridge_lasso", "Ridge (L2) and Lasso (L1) Regularization", (6, 10)),
            ("CS771_L07_TOPIC_primal_dual", "Optimal Weight Vector and Representer Perspectives", (11, 14)),
        ],
    },
    "lecture08.pdf": {
        "lecture_id": "CS771_L08",
        "lecture_num": 8,
        "chapter": 8,
        "title": "Solving Optimization Problems in ML (contd): SGD and Duality",
        "description": "Stochastic Gradient Descent (SGD), mini-batch updates, convexity in ML, Alternating Optimization (ALT-OPT), and Lagrangian duality.",
        "topics": [
            ("CS771_L08_TOPIC_sgd", "Stochastic Gradient Descent and Mini-Batch Updates", (1, 10)),
            ("CS771_L08_TOPIC_alt_opt", "Alternating Optimization (ALT-OPT) Algorithms", (11, 18)),
            ("CS771_L08_TOPIC_duality", "Lagrangian, Primal and Dual Formulations", (19, 26)),
        ],
    },
    "lecture09.pdf": {
        "lecture_id": "CS771_L09",
        "lecture_num": 9,
        "chapter": 9,
        "title": "Linear Models for Classification",
        "description": "Linear classifiers, decision hyperplanes, functional and geometric margins, classification loss functions (0-1, Perceptron, Hinge, Logistic), and Logistic Regression.",
        "topics": [
            ("CS771_L09_TOPIC_hyperplanes_margins", "Decision Hyperplanes, Scores, and Margins", (1, 6)),
            ("CS771_L09_TOPIC_classification_losses", "Loss Functions for Binary Classification", (7, 8)),
            ("CS771_L09_TOPIC_logistic_regression", "Logistic Regression and Class Probability Estimation", (9, 10)),
        ],
    },
    "lecture10.pdf": {
        "lecture_id": "CS771_L10",
        "lecture_num": 10,
        "chapter": 10,
        "title": "Linear Classification, Support Vector Machines, and Kernels",
        "description": "Hard-margin and soft-margin SVM, slack variables, hinge loss + L2 regularization, the kernel trick, Mercer's condition, RBF kernel, and nonlinear decision boundaries.",
        "topics": [
            ("CS771_L10_TOPIC_svm_max_margin", "Hard-Margin Support Vector Machines", (1, 8)),
            ("CS771_L10_TOPIC_soft_margin_svm", "Soft-Margin SVM and Hinge Loss Formulation", (9, 12)),
            ("CS771_L10_TOPIC_kernel_trick", "The Kernel Trick, RBF and Polynomial Kernels", (13, 21)),
        ],
    },
    "lecture11.pdf": {
        "lecture_id": "CS771_L11",
        "lecture_num": 11,
        "chapter": 11,
        "title": "Basics of Probabilistic Machine Learning",
        "description": "Likelihood vs probability, Maximum Likelihood Estimation (MLE), prior beliefs, Bayes' rule, Maximum A Posteriori (MAP) estimation, and MAP as regularized MLE.",
        "topics": [
            ("CS771_L11_TOPIC_likelihood_mle", "Likelihood, Negative Log-Likelihood, and MLE", (1, 10)),
            ("CS771_L11_TOPIC_bayes_priors", "Prior Distributions and Bayes' Rule", (11, 14)),
            ("CS771_L11_TOPIC_map_estimation", "Maximum A Posteriori (MAP) Estimation", (15, 17)),
        ],
    },
    "lecture12.pdf": {
        "lecture_id": "CS771_L12",
        "lecture_num": 12,
        "chapter": 12,
        "title": "Basics of Probabilistic Machine Learning (contd)",
        "description": "Multivariate Gaussian distributions, marginal and conditional distributions, prediction in probabilistic models, and fully Bayesian predictive distributions.",
        "topics": [
            ("CS771_L12_TOPIC_multivariate_gaussian", "Multivariate Gaussian Distribution Modeling", (1, 6)),
            ("CS771_L12_TOPIC_bayesian_prediction", "Prediction and Bayesian Predictive Distributions", (7, 12)),
        ],
    },
    "lecture13.pdf": {
        "lecture_id": "CS771_L13",
        "lecture_num": 13,
        "chapter": 13,
        "title": "Probabilistic Models for Supervised Learning",
        "description": "Probabilistic linear regression, posterior distributions over weights, generative vs discriminative classification, and Gaussian Naive Bayes classifiers.",
        "topics": [
            ("CS771_L13_TOPIC_prob_regression", "Probabilistic Linear Regression and Weight Posterior", (1, 10)),
            ("CS771_L13_TOPIC_generative_vs_discriminative", "Generative vs Discriminative Classification Approaches", (11, 15)),
            ("CS771_L13_TOPIC_gaussian_naive_bayes", "Gaussian Class Conditionals and Naive Bayes", (16, 20)),
        ],
    },
    "lecture14.pdf": {
        "lecture_id": "CS771_L14",
        "lecture_num": 14,
        "chapter": 14,
        "title": "Unsupervised Learning: Data Clustering",
        "description": "Clustering as unsupervised learning, K-Means clustering algorithm, quantization loss, alternating optimization for cluster centers, and clustering geometry.",
        "topics": [
            ("CS771_L14_TOPIC_clustering_intro", "Introduction to Clustering and Quantization Loss", (1, 5)),
            ("CS771_L14_TOPIC_kmeans_algo", "K-Means Algorithm, Convergence, and Cluster Shapes", (6, 10)),
            ("CS771_L14_TOPIC_beyond_kmeans", "Clustering Extensions and Supervised Applications", (11, 17)),
        ],
    },
    "lecture15.pdf": {
        "lecture_id": "CS771_L15",
        "lecture_num": 15,
        "chapter": 15,
        "title": "Unsupervised Learning: Dimensionality Reduction",
        "description": "Dimensionality reduction motivation, matrix factorization perspective, linear encoder-decoder, Principal Component Analysis (PCA) variance maximization derivation.",
        "topics": [
            ("CS771_L15_TOPIC_dim_red_intro", "Motivation for Dimensionality Reduction and Matrix Factorization", (1, 9)),
            ("CS771_L15_TOPIC_pca_derivation", "Principal Component Analysis (PCA) Variance Maximization", (10, 16)),
        ],
    },
    "lecture16.pdf": {
        "lecture_id": "CS771_L16",
        "lecture_num": 16,
        "chapter": 16,
        "title": "Unsupervised Learning: Dimensionality Reduction (contd)",
        "description": "Nonlinear dimensionality reduction, manifold learning, SNE and t-SNE, task-oriented dimensionality reduction, and Low-Rank Adaptation (LoRA).",
        "topics": [
            ("CS771_L16_TOPIC_nonlinear_dim_red", "Nonlinear Dimensionality Reduction and Manifold Learning", (1, 9)),
            ("CS771_L16_TOPIC_sne_tsne", "Stochastic Neighbor Embedding (SNE, t-SNE)", (10, 14)),
            ("CS771_L16_TOPIC_lora", "Task-Oriented Reduction and Low-Rank Adaptation (LoRA)", (15, 17)),
        ],
    },
}

EE602_CHAPTER_METADATA: Dict[str, Dict[str, Any]] = {
    "Frontmatter": {
        "lecture_id": "EE602_FRONTMATTER",
        "lecture_num": 0,
        "chapter": 0,
        "title": "Title Page, Table of Contents, and Foreword",
        "description": "Front matter, table of contents, and notices for Fundamentals of Statistical Signal Processing.",
        "page_range": (1, 7),
        "topics": [],
    },
    "Preface": {
        "lecture_id": "EE602_PREFACE",
        "lecture_num": 0,
        "chapter": 0,
        "title": "Preface",
        "description": "Preface.",
        "page_range": (8, 9),
        "topics": [
        ],
    },
    "Chapter 01": {
        "lecture_id": "EE602_CH01",
        "lecture_num": 1,
        "chapter": 1,
        "title": "Introduction",
        "description": "Estimation Theory Chapter 1: Introduction.",
        "page_range": (10, 23),
        "topics": [
            ("EE602_CH01_TOPIC_1_1_estimation_in_signal_", "1.1 Estimation in Signal Processing", (10, 15)),
            ("EE602_CH01_TOPIC_1_2_the_mathematical_esti", "1.2 The Mathematical Estimation Problem", (16, 17)),
            ("EE602_CH01_TOPIC_1_3_assessing_estimator_p", "1.3 Assessing Estimator Performance", (18, 20)),
            ("EE602_CH01_TOPIC_1_4_some_notes_to_the_rea", "1.4 Some Notes to the Reader", (21, 21)),
            ("EE602_CH01_TOPIC_references", "References", (22, 22)),
            ("EE602_CH01_TOPIC_problems", "Problems", (23, 23)),
        ],
    },
    "Chapter 02": {
        "lecture_id": "EE602_CH02",
        "lecture_num": 2,
        "chapter": 2,
        "title": "Minimum Variance Unbiased Estimation",
        "description": "Estimation Theory Chapter 2: Minimum Variance Unbiased Estimation.",
        "page_range": (24, 35),
        "topics": [
            ("EE602_CH02_TOPIC_2_1_introduction", "2.1 Introduction", (24, 24)),
            ("EE602_CH02_TOPIC_2_2_summary", "2.2 Summary", (24, 24)),
            ("EE602_CH02_TOPIC_2_3_unbiased_estimators", "2.3 Unbiased Estimators", (25, 27)),
            ("EE602_CH02_TOPIC_2_4_minimum_variance_crit", "2.4 Minimum Variance Criterion", (28, 28)),
            ("EE602_CH02_TOPIC_2_5_existence_of_the_mini", "2.5 Existence of the Minimum Variance Unbiased Estimator", (29, 29)),
            ("EE602_CH02_TOPIC_2_6_finding_the_minimum_v", "2.6 Finding the Minimum Variance Unbiased Estimator", (30, 30)),
            ("EE602_CH02_TOPIC_2_7_extension_to_a_vector", "2.7 Extension to a Vector Parameter", (31, 31)),
            ("EE602_CH02_TOPIC_references", "References", (32, 32)),
            ("EE602_CH02_TOPIC_problems", "Problems", (32, 35)),
        ],
    },
    "Chapter 03": {
        "lecture_id": "EE602_CH03",
        "lecture_num": 3,
        "chapter": 3,
        "title": "Cramer-Rao Lower Bound",
        "description": "Estimation Theory Chapter 3: Cramer-Rao Lower Bound.",
        "page_range": (36, 91),
        "topics": [
            ("EE602_CH03_TOPIC_3_1_introduction", "3.1 Introduction", (36, 36)),
            ("EE602_CH03_TOPIC_3_2_summary", "3.2 Summary", (36, 36)),
            ("EE602_CH03_TOPIC_3_3_estimator_accuracy_co", "3.3 Estimator Accuracy Considerations", (37, 38)),
            ("EE602_CH03_TOPIC_3_4_cramer_rao_lower_boun", "3.4 Cramer-Rao Lower Bound", (39, 43)),
            ("EE602_CH03_TOPIC_3_5_general_crlb_for_sign", "3.5 General CRLB for Signals in White Gaussian Noise", (44, 45)),
            ("EE602_CH03_TOPIC_3_6_transformation_of_par", "3.6 Transformation of Parameters", (46, 47)),
            ("EE602_CH03_TOPIC_3_7_extension_to_a_vector", "3.7 Extension to a Vector Parameter", (48, 53)),
            ("EE602_CH03_TOPIC_3_8_vector_parameter_crlb", "3.8 Vector Parameter CRLB for Transformations", (54, 55)),
            ("EE602_CH03_TOPIC_3_9_crlb_for_the_general_", "3.9 CRLB for the General Gaussian Case", (56, 58)),
            ("EE602_CH03_TOPIC_3_10_asymptotic_crlb_for_", "3.10 Asymptotic CRLB for WSS Gaussian Random Processes", (59, 61)),
            ("EE602_CH03_TOPIC_3_11_signal_processing_ex", "3.11 Signal Processing Examples", (62, 71)),
            ("EE602_CH03_TOPIC_references", "References", (72, 72)),
            ("EE602_CH03_TOPIC_problems", "Problems", (72, 75)),
            ("EE602_CH03_TOPIC_3a_derivation_of_scalar_p", "3A Derivation of Scalar Parameter CRLB", (76, 78)),
            ("EE602_CH03_TOPIC_3b_derivation_of_vector_p", "3B Derivation of Vector Parameter CRLB", (79, 81)),
            ("EE602_CH03_TOPIC_3c_derivation_of_general_", "3C Derivation of General Gaussian CRLB", (82, 85)),
            ("EE602_CH03_TOPIC_3d_derivation_of_asymptot", "3D Derivation of Asymptotic CRLB", (86, 91)),
        ],
    },
    "Chapter 04": {
        "lecture_id": "EE602_CH04",
        "lecture_num": 4,
        "chapter": 4,
        "title": "Linear Models",
        "description": "Estimation Theory Chapter 4: Linear Models.",
        "page_range": (92, 109),
        "topics": [
            ("EE602_CH04_TOPIC_4_1_introduction", "4.1 Introduction", (92, 92)),
            ("EE602_CH04_TOPIC_4_2_summary", "4.2 Summary", (92, 92)),
            ("EE602_CH04_TOPIC_4_3_definition_and_proper", "4.3 Definition and Properties", (92, 94)),
            ("EE602_CH04_TOPIC_4_4_linear_model_examples", "4.4 Linear Model Examples", (95, 102)),
            ("EE602_CH04_TOPIC_4_5_extension_to_the_line", "4.5 Extension to the Linear Model", (103, 105)),
            ("EE602_CH04_TOPIC_references", "References", (106, 106)),
            ("EE602_CH04_TOPIC_problems", "Problems", (106, 109)),
        ],
    },
    "Chapter 05": {
        "lecture_id": "EE602_CH05",
        "lecture_num": 5,
        "chapter": 5,
        "title": "General Minimum Variance Unbiased Estimation",
        "description": "Estimation Theory Chapter 5: General Minimum Variance Unbiased Estimation.",
        "page_range": (110, 141),
        "topics": [
            ("EE602_CH05_TOPIC_5_1_introduction", "5.1 Introduction", (110, 110)),
            ("EE602_CH05_TOPIC_5_2_summary", "5.2 Summary", (110, 110)),
            ("EE602_CH05_TOPIC_5_3_sufficient_statistics", "5.3 Sufficient Statistics", (111, 112)),
            ("EE602_CH05_TOPIC_5_4_finding_sufficient_st", "5.4 Finding Sufficient Statistics", (113, 115)),
            ("EE602_CH05_TOPIC_5_5_using_sufficiency_to_", "5.5 Using Sufficiency to Find the MVU Estimator", (116, 124)),
            ("EE602_CH05_TOPIC_5_6_extension_to_a_vector", "5.6 Extension to a Vector Parameter", (125, 130)),
            ("EE602_CH05_TOPIC_references", "References", (131, 131)),
            ("EE602_CH05_TOPIC_problems", "Problems", (131, 135)),
            ("EE602_CH05_TOPIC_5a_proof_of_neyman_fisher", "5A Proof of Neyman-Fisher Factorization Theorem (Scalar Parameter)", (136, 138)),
            ("EE602_CH05_TOPIC_5b_proof_of_rao_blackwell", "5B Proof of Rao-Blackwell-Lehmann-Scheffe Theorem (Scalar Parameter)", (139, 141)),
        ],
    },
    "Chapter 06": {
        "lecture_id": "EE602_CH06",
        "lecture_num": 6,
        "chapter": 6,
        "title": "Best Linear Unbiased Estimators",
        "description": "Estimation Theory Chapter 6: Best Linear Unbiased Estimators.",
        "page_range": (142, 165),
        "topics": [
            ("EE602_CH06_TOPIC_6_1_introduction", "6.1 Introduction", (142, 142)),
            ("EE602_CH06_TOPIC_6_2_summary", "6.2 Summary", (142, 142)),
            ("EE602_CH06_TOPIC_6_3_definition_of_the_blu", "6.3 Definition of the BLUE", (143, 144)),
            ("EE602_CH06_TOPIC_6_4_finding_the_blue", "6.4 Finding the BLUE", (145, 147)),
            ("EE602_CH06_TOPIC_6_5_extension_to_a_vector", "6.5 Extension to a Vector Parameter", (148, 149)),
            ("EE602_CH06_TOPIC_6_6_signal_processing_exa", "6.6 Signal Processing Example", (150, 154)),
            ("EE602_CH06_TOPIC_references", "References", (155, 155)),
            ("EE602_CH06_TOPIC_problems", "Problems", (155, 159)),
            ("EE602_CH06_TOPIC_6a_derivation_of_scalar_b", "6A Derivation of Scalar BLUE", (160, 161)),
            ("EE602_CH06_TOPIC_6b_derivation_of_vector_b", "6B Derivation of Vector BLUE", (162, 165)),
        ],
    },
    "Chapter 07": {
        "lecture_id": "EE602_CH07",
        "lecture_num": 7,
        "chapter": 7,
        "title": "Maximum Likelihood Estimation",
        "description": "Estimation Theory Chapter 7: Maximum Likelihood Estimation.",
        "page_range": (166, 227),
        "topics": [
            ("EE602_CH07_TOPIC_7_1_introduction", "7.1 Introduction", (166, 166)),
            ("EE602_CH07_TOPIC_7_2_summary", "7.2 Summary", (166, 166)),
            ("EE602_CH07_TOPIC_7_3_an_example", "7.3 An Example", (167, 170)),
            ("EE602_CH07_TOPIC_7_4_finding_the_mle", "7.4 Finding the MLE", (171, 172)),
            ("EE602_CH07_TOPIC_7_5_properties_of_the_mle", "7.5 Properties of the MLE", (173, 181)),
            ("EE602_CH07_TOPIC_7_6_mle_for_transformed_p", "7.6 MLE for Transformed Parameters", (182, 185)),
            ("EE602_CH07_TOPIC_7_7_numerical_determinati", "7.7 Numerical Determination of the MLE", (186, 190)),
            ("EE602_CH07_TOPIC_7_8_extension_to_a_vector", "7.8 Extension to a Vector Parameter", (191, 198)),
            ("EE602_CH07_TOPIC_7_9_asymptotic_mle", "7.9 Asymptotic MLE", (199, 199)),
            ("EE602_CH07_TOPIC_7_10_signal_processing_ex", "7.10 Signal Processing Examples", (200, 207)),
            ("EE602_CH07_TOPIC_references", "References", (208, 208)),
            ("EE602_CH07_TOPIC_problems", "Problems", (208, 213)),
            ("EE602_CH07_TOPIC_7a_monte_carlo_methods", "7A Monte Carlo Methods", (214, 219)),
            ("EE602_CH07_TOPIC_7b_asymptotic_pdf_of_mle_", "7B Asymptotic PDF of MLE for a Scalar Parameter", (220, 222)),
            ("EE602_CH07_TOPIC_7c_derivation_of_conditio", "7C Derivation of Conditional Log-Likelihood for EM Algorithm Example", (223, 227)),
        ],
    },
    "Chapter 08": {
        "lecture_id": "EE602_CH08",
        "lecture_num": 8,
        "chapter": 8,
        "title": "Least Squares",
        "description": "Estimation Theory Chapter 8: Least Squares.",
        "page_range": (228, 297),
        "topics": [
            ("EE602_CH08_TOPIC_8_1_introduction", "8.1 Introduction", (228, 228)),
            ("EE602_CH08_TOPIC_8_2_summary", "8.2 Summary", (228, 228)),
            ("EE602_CH08_TOPIC_8_3_the_least_squares_app", "8.3 The Least Squares Approach", (229, 231)),
            ("EE602_CH08_TOPIC_8_4_linear_least_squares", "8.4 Linear Least Squares", (232, 234)),
            ("EE602_CH08_TOPIC_8_5_geometrical_interpret", "8.5 Geometrical Interpretations", (235, 240)),
            ("EE602_CH08_TOPIC_8_6_order_recursive_least", "8.6 Order-Recursive Least Squares", (241, 250)),
            ("EE602_CH08_TOPIC_8_7_sequential_least_squa", "8.7 Sequential Least Squares", (251, 259)),
            ("EE602_CH08_TOPIC_8_8_constrained_least_squ", "8.8 Constrained Least Squares", (260, 262)),
            ("EE602_CH08_TOPIC_8_9_nonlinear_least_squar", "8.9 Nonlinear Least Squares", (263, 268)),
            ("EE602_CH08_TOPIC_8_10_signal_processing_ex", "8.10 Signal Processing Examples", (269, 283)),
            ("EE602_CH08_TOPIC_references", "References", (284, 284)),
            ("EE602_CH08_TOPIC_problems", "Problems", (284, 290)),
            ("EE602_CH08_TOPIC_8a_derivation_of_order_re", "8A Derivation of Order-Recursive Least Squares", (291, 293)),
            ("EE602_CH08_TOPIC_8b_derivation_of_recursiv", "8B Derivation of Recursive Projection Matrix", (294, 294)),
            ("EE602_CH08_TOPIC_8c_derivation_of_sequenti", "8C Derivation of Sequential Least Squares", (295, 297)),
        ],
    },
    "Chapter 09": {
        "lecture_id": "EE602_CH09",
        "lecture_num": 9,
        "chapter": 9,
        "title": "Method of Moments",
        "description": "Estimation Theory Chapter 9: Method of Moments.",
        "page_range": (298, 317),
        "topics": [
            ("EE602_CH09_TOPIC_9_1_introduction", "9.1 Introduction", (298, 298)),
            ("EE602_CH09_TOPIC_9_2_summary", "9.2 Summary", (298, 298)),
            ("EE602_CH09_TOPIC_9_3_method_of_moments", "9.3 Method of Moments", (298, 300)),
            ("EE602_CH09_TOPIC_9_4_extension_to_a_vector", "9.4 Extension to a Vector Parameter", (301, 302)),
            ("EE602_CH09_TOPIC_9_5_statistical_evaluatio", "9.5 Statistical Evaluation of Estimators", (303, 307)),
            ("EE602_CH09_TOPIC_9_6_signal_processing_exa", "9.6 Signal Processing Example", (308, 312)),
            ("EE602_CH09_TOPIC_references", "References", (313, 313)),
            ("EE602_CH09_TOPIC_problems", "Problems", (313, 317)),
        ],
    },
    "Chapter 10": {
        "lecture_id": "EE602_CH10",
        "lecture_num": 10,
        "chapter": 10,
        "title": "The Bayesian Philosophy",
        "description": "Estimation Theory Chapter 10: The Bayesian Philosophy.",
        "page_range": (318, 349),
        "topics": [
            ("EE602_CH10_TOPIC_10_1_introduction", "10.1 Introduction", (318, 318)),
            ("EE602_CH10_TOPIC_10_2_summary", "10.2 Summary", (318, 318)),
            ("EE602_CH10_TOPIC_10_3_prior_knowledge_and_", "10.3 Prior Knowledge and Estimation", (319, 324)),
            ("EE602_CH10_TOPIC_10_4_choosing_a_prior_pdf", "10.4 Choosing a Prior PDF", (325, 329)),
            ("EE602_CH10_TOPIC_10_5_properties_of_the_ga", "10.5 Properties of the Gaussian PDF", (330, 333)),
            ("EE602_CH10_TOPIC_10_6_bayesian_linear_mode", "10.6 Bayesian Linear Model", (334, 336)),
            ("EE602_CH10_TOPIC_10_7_nuisance_parameters", "10.7 Nuisance Parameters", (337, 338)),
            ("EE602_CH10_TOPIC_10_8_bayesian_estimation_", "10.8 Bayesian Estimation for Deterministic Parameters", (339, 340)),
            ("EE602_CH10_TOPIC_references", "References", (341, 341)),
            ("EE602_CH10_TOPIC_problems", "Problems", (341, 345)),
            ("EE602_CH10_TOPIC_10a_derivation_of_conditi", "10A Derivation of Conditional Gaussian PDF", (346, 349)),
        ],
    },
    "Chapter 11": {
        "lecture_id": "EE602_CH11",
        "lecture_num": 11,
        "chapter": 11,
        "title": "General Bayesian Estimators",
        "description": "Estimation Theory Chapter 11: General Bayesian Estimators.",
        "page_range": (350, 387),
        "topics": [
            ("EE602_CH11_TOPIC_11_1_introduction", "11.1 Introduction", (350, 350)),
            ("EE602_CH11_TOPIC_11_2_summary", "11.2 Summary", (350, 350)),
            ("EE602_CH11_TOPIC_11_3_risk_functions", "11.3 Risk Functions", (351, 352)),
            ("EE602_CH11_TOPIC_11_4_minimum_mean_square_", "11.4 Minimum Mean Square Error Estimators", (353, 358)),
            ("EE602_CH11_TOPIC_11_5_maximum_a_posteriori", "11.5 Maximum A Posteriori Estimators", (359, 367)),
            ("EE602_CH11_TOPIC_11_6_performance_descript", "11.6 Performance Description", (368, 373)),
            ("EE602_CH11_TOPIC_11_7_signal_processing_ex", "11.7 Signal Processing Example", (374, 378)),
            ("EE602_CH11_TOPIC_references", "References", (379, 379)),
            ("EE602_CH11_TOPIC_problems", "Problems", (379, 383)),
            ("EE602_CH11_TOPIC_11a_conversion_of_continu", "11A Conversion of Continuous-Time System to DIscrete-TIme System", (384, 387)),
        ],
    },
    "Chapter 12": {
        "lecture_id": "EE602_CH12",
        "lecture_num": 12,
        "chapter": 12,
        "title": "Linear Bayesian Estimators",
        "description": "Estimation Theory Chapter 12: Linear Bayesian Estimators.",
        "page_range": (388, 427),
        "topics": [
            ("EE602_CH12_TOPIC_12_1_introduction", "12.1 Introduction", (388, 388)),
            ("EE602_CH12_TOPIC_12_2_summary", "12.2 Summary", (388, 388)),
            ("EE602_CH12_TOPIC_12_3_linear_mmse_estimati", "12.3 Linear MMSE Estimation", (389, 392)),
            ("EE602_CH12_TOPIC_12_4_geometrical_interpre", "12.4 Geometrical Interpretations", (393, 397)),
            ("EE602_CH12_TOPIC_12_5_the_vector_lmmse_est", "12.5 The Vector LMMSE Estimator", (398, 400)),
            ("EE602_CH12_TOPIC_12_6_sequential_lmmse_est", "12.6 Sequential LMMSE Estimation", (401, 408)),
            ("EE602_CH12_TOPIC_12_7_signal_processing_ex", "12.7 Signal Processing Examples - Wiener Filtering", (409, 418)),
            ("EE602_CH12_TOPIC_references", "References", (419, 419)),
            ("EE602_CH12_TOPIC_problems", "Problems", (419, 423)),
            ("EE602_CH12_TOPIC_12a_derivation_of_sequent", "12A Derivation of Sequential LMMSE Estimator", (424, 427)),
        ],
    },
    "Chapter 13": {
        "lecture_id": "EE602_CH13",
        "lecture_num": 13,
        "chapter": 13,
        "title": "Kalman Filters",
        "description": "Estimation Theory Chapter 13: Kalman Filters.",
        "page_range": (428, 487),
        "topics": [
            ("EE602_CH13_TOPIC_13_1_introduction", "13.1 Introduction", (428, 428)),
            ("EE602_CH13_TOPIC_13_2_summary", "13.2 Summary", (428, 428)),
            ("EE602_CH13_TOPIC_13_3_dynamical_signal_mod", "13.3 Dynamical Signal Models", (429, 439)),
            ("EE602_CH13_TOPIC_13_4_scalar_kalman_filter", "13.4 Scalar Kalman Filter", (440, 450)),
            ("EE602_CH13_TOPIC_13_5_kalman_versus_wiener", "13.5 Kalman Versus Wiener Filters", (451, 454)),
            ("EE602_CH13_TOPIC_13_6_vector_kalman_filter", "13.6 Vector Kalman Filter", (455, 457)),
            ("EE602_CH13_TOPIC_13_7_extended_kalman_filt", "13.7 Extended Kalman Filter", (458, 460)),
            ("EE602_CH13_TOPIC_13_8_signal_processing_ex", "13.8 Signal Processing Examples", (461, 475)),
            ("EE602_CH13_TOPIC_references", "References", (476, 476)),
            ("EE602_CH13_TOPIC_problems", "Problems", (476, 479)),
            ("EE602_CH13_TOPIC_13a_vector_kalman_filter_", "13A Vector Kalman Filter Derivation", (480, 484)),
            ("EE602_CH13_TOPIC_13b_extended_kalman_filte", "13B Extended Kalman Filter Derivation", (485, 487)),
        ],
    },
    "Chapter 14": {
        "lecture_id": "EE602_CH14",
        "lecture_num": 14,
        "chapter": 14,
        "title": "Summary of Estimators",
        "description": "Estimation Theory Chapter 14: Summary of Estimators.",
        "page_range": (488, 501),
        "topics": [
            ("EE602_CH14_TOPIC_14_1_introduction", "14.1 Introduction", (488, 488)),
            ("EE602_CH14_TOPIC_14_2_estimation_approache", "14.2 Estimation Approaches", (488, 494)),
            ("EE602_CH14_TOPIC_14_3_linear_model", "14.3 Linear Model", (495, 497)),
            ("EE602_CH14_TOPIC_14_4_choosing_an_estimato", "14.4 Choosing an Estimator", (498, 501)),
        ],
    },
    "Chapter 15": {
        "lecture_id": "EE602_CH15",
        "lecture_num": 15,
        "chapter": 15,
        "title": "Extensions for Complex Data and Parameters",
        "description": "Estimation Theory Chapter 15: Extensions for Complex Data and Parameters.",
        "page_range": (502, 575),
        "topics": [
            ("EE602_CH15_TOPIC_15_1_introduction", "15.1 Introduction", (502, 502)),
            ("EE602_CH15_TOPIC_15_2_summary", "15.2 Summary", (502, 502)),
            ("EE602_CH15_TOPIC_15_3_complex_data_and_par", "15.3 Complex Data and Parameters", (503, 508)),
            ("EE602_CH15_TOPIC_15_4_complex_random_varia", "15.4 Complex Random Variables and PDFs", (509, 521)),
            ("EE602_CH15_TOPIC_15_5_complex_wss_random_p", "15.5 Complex WSS Random Processes", (522, 525)),
            ("EE602_CH15_TOPIC_15_6_derivatives_gradient", "15.6 Derivatives, Gradients, and Optimization", (526, 532)),
            ("EE602_CH15_TOPIC_15_7_classical_estimation", "15.7 Classical Estimation with Complex Data", (533, 540)),
            ("EE602_CH15_TOPIC_15_8_bayesian_estimation", "15.8 Bayesian Estimation", (541, 543)),
            ("EE602_CH15_TOPIC_15_9_asymptotic_complex_g", "15.9 Asymptotic Complex Gaussian PDF", (544, 547)),
            ("EE602_CH15_TOPIC_15_10_signal_processing_e", "15.10 Signal Processing Examples", (548, 557)),
            ("EE602_CH15_TOPIC_references", "References", (558, 558)),
            ("EE602_CH15_TOPIC_problems", "Problems", (558, 563)),
            ("EE602_CH15_TOPIC_15a_derivation_of_propert", "15A Derivation of Properties of Complex Covariance Matrices", (564, 566)),
            ("EE602_CH15_TOPIC_15b_derivation_of_propert", "15B Derivation of Properties of Complex Gaussian PDF", (567, 571)),
            ("EE602_CH15_TOPIC_15c_derivation_of_crlb_an", "15C Derivation of CRLB and MLE Formulas", (572, 575)),
        ],
    },
    "Appendix A1": {
        "lecture_id": "EE602_APP_A1",
        "lecture_num": 16,
        "chapter": 16,
        "title": "Review of Important Concepts",
        "description": "Estimation Theory Chapter 16: Review of Important Concepts.",
        "page_range": (576, 591),
        "topics": [
            ("EE602_APP_A1_TOPIC_a1_1_linear_and_matrix_al", "A1.1 Linear and Matrix Algebra", (576, 582)),
            ("EE602_APP_A1_TOPIC_a1_2_probability_random_p", "A1.2 Probability, Random Processes. and Time Series Models", (583, 591)),
        ],
    },
    "Appendix A2": {
        "lecture_id": "EE602_APP_A2",
        "lecture_num": 17,
        "chapter": 17,
        "title": "Glossary of Symbols and Abbreviations",
        "description": "Estimation Theory Chapter 17: Glossary of Symbols and Abbreviations.",
        "page_range": (592, 597),
        "topics": [
        ],
    },
    "Index": {
        "lecture_id": "EE602_INDEX",
        "lecture_num": 18,
        "chapter": 18,
        "title": "Index",
        "description": "Estimation Theory Chapter 18: Index.",
        "page_range": (598, 604),
        "topics": [
        ],
    },
}

COURSE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "MTH441": {
        "course_name": "Regression Analysis and ANOVA",
        "instructor": "Prof. Shalabh",
        "institution": "IIT Kanpur",
        "graph_path": "storage/graph/MTH441_graph.json",
        "metadata_catalog": MTH441_LECTURE_METADATA,
    },
    "CS771": {
        "course_name": "Introduction to Machine Learning",
        "instructor": "Prof. Piyush Rai",
        "institution": "IIT Kanpur",
        "graph_path": "storage/graph/CS771_graph.json",
        "metadata_catalog": CS771_LECTURE_METADATA,
    },
    "EE602": {
        "course_name": "Statistical Signal Processing / Estimation Theory (Steven M. Kay)",
        "instructor": "Prof. Ketan Rajawat / Steven M. Kay",
        "institution": "IIT Kanpur",
        "graph_path": "storage/graph/EE602_graph.json",
        "metadata_catalog": EE602_CHAPTER_METADATA,
    },
}


class IngestManager:
    """Manages multi-lecture ingestion, embedding indexing, and knowledge graph construction."""

    def __init__(
        self,
        course_id: str = "MTH441",
        course_name: Optional[str] = None,
        instructor: Optional[str] = None,
        institution: Optional[str] = None,
        persist_dir: str = "storage/chroma",
        collection_name: str = "hello_lets_talk_iitk_chunks_v1",
        graph_path: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        equation_store: Optional[EquationStore] = None,
    ):
        """Initialize IngestManager.

        Args:
            course_id: Course code (e.g. 'MTH441' or 'CS771').
            course_name: Course title (defaults to course default if None).
            instructor: Professor name (defaults to course default if None).
            institution: Institution name (defaults to 'IIT Kanpur').
            persist_dir: Path to ChromaDB storage directory.
            collection_name: ChromaDB collection identifier.
            graph_path: Path for saving serialized knowledge graph JSON.
            embedder: Embedder instance (created if None).
            equation_store: EquationStore instance (created if None).
        """
        defaults = COURSE_DEFAULTS.get(course_id.upper(), {})
        self.course_id = course_id.upper()
        self.course_name = course_name or defaults.get("course_name", self.course_id)
        self.instructor = instructor or defaults.get("instructor", "Faculty")
        self.institution = institution or defaults.get("institution", "IIT Kanpur")
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.graph_path = graph_path or defaults.get("graph_path", f"storage/graph/{self.course_id}_graph.json")

        self.embedder = embedder or get_default_embedder()
        self.vector_store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
        self.equation_store = equation_store or EquationStore()
        self.graph = CourseGraph(course_id=self.course_id)

    def _determine_topic_for_chunk(
        self,
        lecture_info: Dict[str, Any],
        page: int,
        chunk_text: str,
    ) -> Optional[str]:
        """Find the best matching topic ID for a chunk using page ranges and keywords."""
        topics = lecture_info.get("topics", [])
        matched_topic = None

        # Check page range match
        for top_id, top_title, (p_start, p_end) in topics:
            if p_start <= page <= p_end:
                matched_topic = top_id
                break

        return matched_topic

    def ingest_lecture_file(
        self,
        pdf_path: str,
        lecture_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest a single lecture PDF into both VectorStore and CourseGraph.

        Args:
            pdf_path: Path to lecture PDF.
            lecture_label: Optional lecture label override (e.g. 'Lecture 01').

        Returns:
            Dict containing ingestion metrics for this lecture.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Lecture PDF not found: {pdf_path}")

        filename = path.name
        catalog = COURSE_DEFAULTS.get(self.course_id, {}).get("metadata_catalog", MTH441_LECTURE_METADATA)
        meta_info = catalog.get(
            filename,
            {
                "lecture_id": f"{self.course_id}_{path.stem.upper()}",
                "lecture_num": 1,
                "chapter": 1,
                "title": path.stem.replace("_", " ").title(),
                "description": "",
                "topics": [],
            },
        )

        lecture_id = meta_info["lecture_id"]
        lecture_num = meta_info["lecture_num"]
        lec_display_name = lecture_label or f"Lecture {lecture_num:02d}"

        print(f"\n[IngestManager] Processing {filename} ({self.course_id} - {lec_display_name})")

        # 1. Extraction
        raw_pages = extract_pdf(
            pdf_path=str(path),
            course=self.course_id,
            lecture=lec_display_name,
        )
        print(f" -> Extracted {len(raw_pages)} pages.")

        # 2. Cleaning
        cleaned_pages = clean_pages(raw_pages)

        # 3. Chunking
        chunks = chunk_pages(cleaned_pages, chunk_size=800, chunk_overlap=120)
        print(f" -> Created {len(chunks)} chunks.")

        # 4. Dense Embeddings
        embeddings = self.embedder.embed_chunks(chunks)

        # 5. ChromaDB Upsert
        upserted_count = self.vector_store.add_chunks(chunks, embeddings)
        print(f" -> Upserted {upserted_count} chunks to ChromaDB collection '{self.collection_name}'.")

        # 6. Populate CourseGraph
        self.graph.add_course(
            course_id=self.course_id,
            name=self.course_name,
            instructor=self.instructor,
            institution=self.institution,
        )

        self.graph.add_lecture(
            lecture_id=lecture_id,
            course_id=self.course_id,
            lecture_num=lecture_num,
            title=meta_info["title"],
            source_file=filename,
            chapter=meta_info.get("chapter", lecture_num),
            description=meta_info.get("description", ""),
        )

        # Register Topic nodes
        for top_id, top_title, _ in meta_info.get("topics", []):
            self.graph.add_topic(
                topic_id=top_id,
                lecture_id=lecture_id,
                title=top_title,
            )

        # Add Chunk nodes and connect to topics
        chunk_ids_in_order = []
        for c in chunks:
            c_meta = c["metadata"]
            c_id = self.vector_store.generate_chunk_id(c)
            page = int(c_meta.get("page", 0))
            c_num = int(c_meta.get("chunk", 0))
            text = c["text"]

            matched_topic = self._determine_topic_for_chunk(meta_info, page, text)

            self.graph.add_chunk(
                chunk_id=c_id,
                lecture_id=lecture_id,
                page=page,
                chunk_num=c_num,
                text=text,
                source=filename,
                topic_id=matched_topic,
            )
            chunk_ids_in_order.append(c_id)

        # Link sequential chunks via NEXT_CHUNK
        seq_links = self.graph.link_sequential_chunks(chunk_ids_in_order)

        # 7. Add Equations from Catalog and link to Topics and Chunks
        lecture_equations = self.equation_store.get_by_lecture(lec_display_name, course=self.course_id)
        eq_links_count = 0

        for eq in lecture_equations:
            matched_top_id = None
            for top_id, top_title, _ in meta_info.get("topics", []):
                if top_title.lower() in eq.topic.lower() or eq.topic.lower() in top_title.lower():
                    matched_top_id = top_id
                    break

            self.graph.add_equation(
                equation_id=eq.id,
                name=eq.name,
                latex=eq.latex,
                lecture_id=lecture_id,
                page=eq.page,
                variables=eq.variables,
                conditions=eq.conditions,
                topic=eq.topic,
                description=eq.description,
                topic_id=matched_top_id,
            )

            # Link chunks on the same page (or mentioning key terms) to this equation
            for c_id in chunk_ids_in_order:
                node = self.graph.get_node(c_id)
                if node and node.get("page") == eq.page:
                    self.graph.link_equation_to_chunk(c_id, eq.id)
                    eq_links_count += 1

        print(f" -> Linked {len(lecture_equations)} discrete equations with {eq_links_count} chunk connections.")

        return {
            "lecture_id": lecture_id,
            "filename": filename,
            "pages": len(raw_pages),
            "chunks": len(chunks),
            "equations": len(lecture_equations),
            "equation_links": eq_links_count,
            "sequential_links": seq_links,
        }

    def ingest_book_file(
        self,
        pdf_path: str,
    ) -> Dict[str, Any]:
        """Ingest a multi-chapter textbook PDF (such as Steven M. Kay's Estimation Theory for EE602).

        Splits pages dynamically by chapter boundaries, creates enriched embeddings,
        and constructs a comprehensive multi-chapter Knowledge Graph with discrete equations.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Book PDF not found: {pdf_path}")

        filename = path.name
        print(f"\n[IngestManager] Processing Textbook: {filename} ({self.course_id})")

        # Dynamic resolvers for chapter labels and numbers
        def resolve_chapter_label(p: int) -> str:
            for ch_lbl, ch_info in EE602_CHAPTER_METADATA.items():
                p_s, p_e = ch_info["page_range"]
                if p_s <= p <= p_e:
                    return ch_lbl
            return "Book"

        def resolve_chapter_num(p: int) -> Optional[int]:
            for ch_lbl, ch_info in EE602_CHAPTER_METADATA.items():
                p_s, p_e = ch_info["page_range"]
                if p_s <= p <= p_e:
                    return ch_info["chapter"]
            return None

        # 1. Extraction with dynamic chapter tagging per page
        raw_pages = extract_pdf(
            pdf_path=str(path),
            course=self.course_id,
            lecture="Book",
            lecture_resolver=resolve_chapter_label,
            chapter_resolver=resolve_chapter_num,
        )
        print(f" -> Extracted {len(raw_pages)} pages from textbook.")

        # 2. Cleaning
        cleaned_pages = clean_pages(raw_pages)

        # 3. Natural boundary Chunking
        chunks = chunk_pages(cleaned_pages, chunk_size=800, chunk_overlap=120)
        print(f" -> Created {len(chunks)} chunks across all chapters.")

        # 4. Dense Embeddings
        embeddings = self.embedder.embed_chunks(chunks)

        # 5. ChromaDB Upsert
        upserted_count = self.vector_store.add_chunks(chunks, embeddings)
        print(f" -> Upserted {upserted_count} chunks to ChromaDB collection '{self.collection_name}'.")

        # 6. Populate CourseGraph
        self.graph.add_course(
            course_id=self.course_id,
            name=self.course_name,
            instructor=self.instructor,
            institution=self.institution,
        )

        # Register all Chapter (Lecture) and Topic nodes
        for ch_lbl, ch_info in EE602_CHAPTER_METADATA.items():
            self.graph.add_lecture(
                lecture_id=ch_info["lecture_id"],
                course_id=self.course_id,
                lecture_num=ch_info["lecture_num"],
                title=ch_info["title"],
                source_file=filename,
                chapter=ch_info.get("chapter", ch_info["lecture_num"]),
                description=ch_info.get("description", ""),
            )
            for top_id, top_title, _ in ch_info.get("topics", []):
                self.graph.add_topic(
                    topic_id=top_id,
                    lecture_id=ch_info["lecture_id"],
                    title=top_title,
                )

        # Register Chunks and connect to Topics & Chapters
        chunk_ids_by_chapter: Dict[str, List[str]] = {}
        for c in chunks:
            c_meta = c["metadata"]
            c_id = self.vector_store.generate_chunk_id(c)
            page = int(c_meta.get("page", 0))
            c_num = int(c_meta.get("chunk", 0))
            ch_lbl = c_meta.get("lecture", "Book")
            text = c["text"]

            ch_info = EE602_CHAPTER_METADATA.get(ch_lbl, {})
            ch_id = ch_info.get("lecture_id", f"{self.course_id}_{ch_lbl}")

            matched_topic = None
            for top_id, top_title, (p_s, p_e) in ch_info.get("topics", []):
                if p_s <= page <= p_e:
                    matched_topic = top_id
                    break

            self.graph.add_chunk(
                chunk_id=c_id,
                lecture_id=ch_id,
                page=page,
                chunk_num=c_num,
                text=text,
                source=filename,
                topic_id=matched_topic,
            )
            chunk_ids_by_chapter.setdefault(ch_id, []).append(c_id)

        # Link sequential chunks per chapter
        total_seq_links = 0
        for ch_id, ch_chunk_ids in chunk_ids_by_chapter.items():
            total_seq_links += self.graph.link_sequential_chunks(ch_chunk_ids)

        # 7. Add Equations from Catalog and link to Chunks
        ee602_equations = self.equation_store.get_by_course(self.course_id)
        eq_links_count = 0

        for eq in ee602_equations:
            ch_info = EE602_CHAPTER_METADATA.get(eq.lecture, {})
            ch_id = ch_info.get("lecture_id", f"{self.course_id}_{eq.lecture}")
            matched_top_id = None
            for top_id, top_title, _ in ch_info.get("topics", []):
                if top_title.lower() in eq.topic.lower() or eq.topic.lower() in top_title.lower():
                    matched_top_id = top_id
                    break

            self.graph.add_equation(
                equation_id=eq.id,
                name=eq.name,
                latex=eq.latex,
                lecture_id=ch_id,
                page=eq.page,
                variables=eq.variables,
                conditions=eq.conditions,
                topic=eq.topic,
                description=eq.description,
                topic_id=matched_top_id,
            )

            # Link chunks on the same page
            for ch_chunk_ids in chunk_ids_by_chapter.values():
                for c_id in ch_chunk_ids:
                    node = self.graph.get_node(c_id)
                    if node and node.get("page") == eq.page:
                        self.graph.link_equation_to_chunk(c_id, eq.id)
                        eq_links_count += 1

        print(f" -> Linked {len(ee602_equations)} discrete equations with {eq_links_count} chunk connections.")

        return {
            "lecture_id": f"{self.course_id}_BOOK",
            "filename": filename,
            "pages": len(raw_pages),
            "chunks": len(chunks),
            "equations": len(ee602_equations),
            "equation_links": eq_links_count,
            "sequential_links": total_seq_links,
        }

    def ingest_all(
        self,
        lectures_dir: str = "data/lectures/MTH441",
        reset_chroma: bool = False,
    ) -> Dict[str, Any]:
        """Batch ingest all lecture PDFs in the course directory.

        Args:
            lectures_dir: Path to directory containing lecture PDF files.
            reset_chroma: If True, clears existing vector collection first.

        Returns:
            Comprehensive summary dict of multi-lecture ingestion.
        """
        dir_path = Path(lectures_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Lectures directory not found: {lectures_dir}")

        pdf_files = sorted(list(dir_path.glob("*.pdf")))
        if not pdf_files:
            raise ValueError(f"No PDF lecture files found in {lectures_dir}")

        if reset_chroma:
            print(f"[IngestManager] Resetting ChromaDB collection: {self.collection_name}")
            self.vector_store.reset_collection()

        lecture_reports = []
        total_pages = 0
        total_chunks = 0
        total_equations = 0

        is_book_course = (self.course_id == "EE602") or any("estimation_theory" in p.name.lower() or "book" in p.name.lower() for p in pdf_files)
        if is_book_course:
            for pdf_path in pdf_files:
                report = self.ingest_book_file(str(pdf_path))
                lecture_reports.append(report)
                total_pages += report["pages"]
                total_chunks += report["chunks"]
                total_equations += report["equations"]
        else:
            for i, pdf_path in enumerate(pdf_files, 1):
                m = re.search(r"lecture(\d+)", pdf_path.stem, re.IGNORECASE)
                if m:
                    lec_num = int(m.group(1))
                    lec_label = f"Lecture {lec_num:02d}"
                else:
                    lec_label = f"Lecture {i:02d}"
                report = self.ingest_lecture_file(str(pdf_path), lecture_label=lec_label)
                lecture_reports.append(report)
                total_pages += report["pages"]
                total_chunks += report["chunks"]
                total_equations += report["equations"]

        # Persist full course graph
        saved_graph_path = self.graph.save_json(self.graph_path)
        graph_stats = self.graph.summary()

        print("\n" + "=" * 65)
        print(f" Multi-Lecture Course Ingestion Complete for {self.course_id}!")
        print(f" - Lectures Ingested: {len(pdf_files)}")
        print(f" - Total Pages: {total_pages}")
        print(f" - Total Chunks in ChromaDB: {self.vector_store.count()}")
        print(f" - Discrete Equations Linked: {total_equations}")
        print(f" - Graph Summary: {graph_stats['total_nodes']} nodes, {graph_stats['total_edges']} edges")
        print(f" - Graph Saved To: {saved_graph_path}")
        print("=" * 65 + "\n")

        return {
            "course_id": self.course_id,
            "lectures_ingested": len(pdf_files),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_equations": total_equations,
            "vector_store_count": self.vector_store.count(),
            "graph_summary": graph_stats,
            "graph_path": saved_graph_path,
            "lecture_reports": lecture_reports,
        }

