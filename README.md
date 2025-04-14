
# 🧠 Clustering Algorithms
## **Mall Customer Segmentation Model**

This project demonstrates the application of unsupervised learning using **KMeans clustering** to segment mall customers into meaningful groups based on their attributes. It's designed to help mall businesses target the right customers for marketing by understanding customer behavior through data.

---

## 📌 Project Scope

Malls often aim to boost their customer base and revenue through intelligent marketing. Machine learning models like clustering help analyze customer attributes and group them based on similar purchasing behavior.


### **Goal**
Develop an **unsupervised clustering model** that accurately segments customers into distinct groups using KMeans.



---

## 📁 Folder Structure

```
├── data/
│   └── mall_customers.csv            
├── models/
│   └── kmeans_model.pkl              
├── notebooks/
│   └── Unsupervised_Clustering_solution.ipynb 
├── src/
│   ├── __init__.py
│   ├── data_loader.py                
│   ├── model.py                      
│   └── predictor.py                  
├── app.py                        
├── main.py                       
├── README.md
└── requirements.txt
```

---

## 🔍 Key Techniques Used

- **KMeans Clustering**
- **Elbow Method** to determine optimal number of clusters
- **Silhouette Score** for validating cluster quality
- **Data visualization** (scatter plots, elbow plot, silhouette plot)

---

---

## 🧪 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/allenecstadam/mall_customers_k-means_clustering.git
cd mall_customers_k-means_clustering
```

### 2. Install the requirements

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python src/main.py
```

### 4. Launch the Streamlit app


[Streamlit App](https://mallcustomersk-meansclustering-bsfamzxjmxqqfvthpfcqcv.streamlit.app/)


---

## 📦 Requirements

```
streamlit
pandas
scikit-learn
matplotlib
seaborn
joblib
os
```

---
## 👤 Author

**Allen Adam**  
🔗 [GitHub Profile](https://github.com/allenecstadam)

---

## 📃 License

MIT License - Free to use and modify.
