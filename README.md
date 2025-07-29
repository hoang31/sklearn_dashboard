
-----

# Interactive Scikit-learn Dashboard 📊

This project is an interactive web dashboard built with Python's Dash framework. It provides a user-friendly interface to perform exploratory data analysis (EDA) and visualize the results of common unsupervised machine learning models from `scikit-learn`.

You can load your own dataset and interactively explore it through various tabs for data description, analysis, and modeling.

-----

## ✨ Features

  * **Data Description Tab**:

      * Automatically generates a summary statistics table for all **numerical columns**.
      * Provides a summary table for all **categorical columns**, showing the distinct categories and their counts.

  * **Data Analysis Tab**:

      * Create interactive **scatter plots** to visualize the correlation between any two numerical variables in your dataset.

  * **Data Modelling Tab**:

      * A dynamic interface that allows you to select a model and see only the relevant options and visualizations.
      * **PCA (Principal Component Analysis)**:
          * Specify the number of principal components ($n$) to compute.
          * Visualize the results in an interactive scatter plot matrix.
          * Color the data points by any categorical variable.
          * View a heatmap of **feature contributions** to each principal component.
      * **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
          * Reduce high-dimensional data to 2 components for visualization.
          * Color the resulting clusters by any categorical variable.
      * **k-Means**:
          * Specify the number of clusters ($k$) to create.
          * Visualize the clustered data.
          * Color the plot by the generated k-Means clusters or any other categorical variable.

-----

## 🛠️ Technologies Used

  * **Backend & Frontend**: [Dash](https://dash.plotly.com/)
  * **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
  * **Machine Learning**: [Scikit-learn](https://scikit-learn.org/stable/)
  * **Interactive Visualizations**: [Plotly](https://plotly.com/python/)
  * **Styling**: [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)

-----

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need to have Python 3.x installed on your system.

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**

      * On macOS/Linux:
        ```sh
        conda create --name sklearn-dashboard python=3.10 -y
        conda activate sklearn-dashboard
        ```

3.  **Install the required dependencies:**

    ```sh
    conda install -c conda-forge pandas scikit-learn dash dash-bootstrap-components plotly

    ```

4.  **Prepare your data:**

      * Place your CSV data file in the project directory.
      * Update the following line in the script to point to your file:
        ```python
        # Incorporate data
        df = pd.read_csv('your_data_file.csv')
        ```

### Running the Application

1.  **Execute the main script:**
    ```sh
    python app.py
    ```
2.  Open your web browser and navigate to the address shown in the terminal (usually `http://127.0.0.1:8050`).

-----

## ToDo List

Potential tasks and idea for improvements:

  * Add more models for clustering and classification (e.g., **Linear Regression**, **Logistic Regression**, **Random Forest**, etc).
  * Allow users to upload their own datasets directly from the interface.
  * Allow users to split the data into training and testing sets.
  * Add more visualizations (e.g., **heatmaps**, **dendrograms**, **clustering maps**).
  * Improve the styling and layout of the dashboard.
  * Implement the code by imported modules.
