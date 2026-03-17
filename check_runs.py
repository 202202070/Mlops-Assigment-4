import mlflow

mlflow.set_experiment("Assignment3_Ahmed")
runs = mlflow.search_runs()

print(f"Total runs: {len(runs)}")
print()

for i, (_, row) in enumerate(runs.iterrows()):
    lr = row["params.learning_rate"]
    bs = row["params.batch_size"]
    ep = row["params.epochs"]
    acc = row["metrics.test_accuracy"]
    loss = row["metrics.test_loss"]
    print(f"Run {i+1}: lr={lr}, bs={bs}, epochs={ep}, test_acc={acc:.4f}, test_loss={loss:.4f}")
