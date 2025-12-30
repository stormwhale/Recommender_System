**Hybrid-LLM Recommender Ecosystem**

A comprehensive suite of recommendation engines, ranging from baseline statistical estimators to scalable, cloud-deployed hybrid systems integrating Large Language Models (LLMs).

📂 Repository Structure

- 🚀 [A] "LLM_hybrid_rec_system" (Core Project)
    - This is the hybrid system that addresses the "cold-start" problem by combining Collaborative Filtering with Semantic Content Analysis.

    - Methodology: Integrates Alternating Least Squares (ALS) with Content-Based Filtering.
        
    - LLM Integration: Leverages Ollama Gemma 3 to extract deep semantic features from unstructured item metadata.
    
    - Impact: Enhances recommendation accuracy and user interpretation of the recommeded results.

- ☁️ [B] "LLM_Azure_cloud_deployment_files" for Azure Cloud Deployment & Containerization
    - Contains the infrastructure-as-code and configuration files required for productionizing the models.

    - Containerization: Full Docker configuration for environment consistency.
    
    - Orchestration: Deployment pipelines for Microsoft Azure Container Instances (ACI).
        
    - Environment: Optimized for Databricks and Apache Spark to handle large-scale inference requests.

- 🧪 [C] "test_projects" - Experimental Testbed
    - A collection of benchmarking projects used to evaluate and tune various recommendation architectures:

        1) PySpark ALS: Testing scalable collaborative filtering on distributed datasets.
        
        
        2) SVD (Singular Value Decomposition): Matrix factorization experimentation for latent feature extraction.
        
        3) Content-Based Filtering: Logic-driven recommendations based on item attribute similarity.
        
        4) Global Average Estimator: A statistical baseline used for model performance benchmarking.


- 🛠 Technical Stack

    - Languages: Python, PySpark, SQL.

    - LLM API: Ollama (Gemma 3).

    - Cloud & DevOps: Azure (ACI), Docker, Databricks.

- 📈 Engineering Highlights

    - Hybrid Logic: Successfully merged a high-dimensionality hybrid recommender system with user-friendly interaction through the integration of LLMs.

    - Scalability: Implementation of the *Apache Spark* framework allows for distributed model training and evaluation.

    - Latency Optimization: Containerized model serving reduces inference overhead for real-time recommendation delivery.

- 👤 Author
    - Licensed Clinical Laboratory Scientist & M.S. Data Science Candidate

     - Current GPA: 3.95/4.0 

     - Experience: 9+ years in clinical data integrity and system implementation
