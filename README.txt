# Airbus Ship Detection

## How to Run the Project

1. Install `uv` by following the instructions on the official website: [https://astral.sh/blog/uv/](https://astral.sh/blog/uv/)

2. Create and activate a virtual environment using `uv`.

3. Install the required dependencies using the `pyproject.toml` file located in the project folder. If a `requirements.txt` file is available, it can be used as an alternative.

4. Download the dataset from the following link:
   [https://www.kaggle.com/competitions/airbus-ship-detection/data](https://www.kaggle.com/competitions/airbus-ship-detection/data)

   Unzip all files and place them directly into the root folder of the project. After extraction, the folder should contain:

   * a `train_v2` folder
   * a `test_v2` folder
   * the file `train_ship_segmentations_v2.csv`

5. Open the file `final_main.ipynb` using Jupyter Notebook.

6. Run all the notebook cells in order to execute data preprocessing, model training, and evaluation.

7. To run the dashboard, digit in the terminal "streamlit run dashboard.py"
