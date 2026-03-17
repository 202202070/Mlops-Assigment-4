# Assignment 3 - MLflow Experiment Tracking

## Step-by-step Commands

### Step 1: Create Conda Environment
```
conda create -n mlops_ass3 python=3.10 -y
```

### Step 2: Install Dependencies
```
conda run -n mlops_ass3 pip install mlflow torch torchvision pandas pyyaml
```

### Step 3: Run the 5 Experiments
```
conda run -n mlops_ass3 python run_experiments.py
```

### Step 4: Launch MLflow UI
```
conda run -n mlops_ass3 mlflow ui --port 5000 --backend-store-uri "file:///d:/MLops ASSIGMENT 3/mlruns"
```

### Step 5: Open MLflow Dashboard
Open browser and go to:
```
http://localhost:5000
```

### Docker (Optional)
```
docker build -t mlops_ass3 .
docker run mlops_ass3
```
