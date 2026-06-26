# CO2 Emission Prediction 🚗💨

Predict vehicle CO2 emissions using **Linear Regression** based on engine specifications.

## 📊 Dataset
- **Samples:** 500 vehicles
- **Features:** Engine size, Cylinders, Fuel type
- **Target:** CO2 emission level

## 🎯 Results
| Metric | Value |
|:---|---:|
| **R² Score** | 0.803 |
| **MAE** | 23.33 |
| **RMSE** | ~30 |

## 📁 Files
- `co2_prediction.py` — Main code
- `co2.csv` — Dataset
- `*.png` — Generated plots

## 🚀 How to Run
```bash
pip install numpy pandas scikit-learn seaborn matplotlib
python co2_prediction.py
