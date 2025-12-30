**Hybrid-LLM Recommender Ecosystem**

A comprehensive suite of recommendation engines, ranging from baseline statistical estimators to scalable, cloud-deployed hybrid systems integrating Large Language Models (LLMs).

📂 Repository Structure

- 🚀 [A] LLM Hybrid Recommender (Core Project)
    - This is the flagship system that addresses the "cold-start" problem by combining Collaborative Filtering with Semantic Content Analysis.

    - Methodology: Integrates Alternating Least Squares (ALS) with Content-Based Filtering.
        
    - LLM Integration: Leverages Ollama Gemma 3 to extract deep semantic features from unstructured item metadata.
    
    - Impact: Enhances recommendation accuracy in sparse data environments by bridging collaborative user patterns with LLM-driven content embeddings.

- ☁️ [B] Azure Cloud Deployment & Containerization
    - Contains the infrastructure-as-code and configuration files required for productionizing the models.

    - Containerization: Full Docker configuration for environment consistency.
    
    - Orchestration: Deployment pipelines for Microsoft Azure Container Instances (ACI).
        
    - Environment: Optimized for Databricks and Apache Spark to handle large-scale inference requests.

- 🧪 [C] Experimental Testbed
    - A collection of benchmarking projects used to evaluate and tune various recommendation architectures:

PySpark ALS: Scalable collaborative filtering for massive datasets.

Content-Based Filtering: Similarity-based logic using item attributes.


- 🛠 Technical Stack

    - Languages: Python, PySpark.


ML Frameworks: Scikit-learn, Ollama (Gemma 3).

Cloud & DevOps: Azure (ACI), Docker, Databricks.


- 📈 Engineering Highlights

- Hybrid Logic: Successfully merged user-interaction matrices with high-dimensional LLM embeddings.


- Scalability: Implementation of the Apache Spark framework allows for distributed model training and evaluation.


- Latency Optimization: Containerized model serving reduces inference overhead for real-time recommendation delivery.

- 👤 Author
 - Licensed Clinical Laboratory Scientist & M.S. Data Science Candidate


 - Current GPA: 3.95/4.0 


Experience: 9+ years in clinical data integrity and system implementation
