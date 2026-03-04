## 什么是计算图

计算图可以粗略的等价于一个函数 `f(input_features)=outputs` ，严格说input/output不是图的一部分

线性回归的静态图 $$y = xA^T + b$$

```python
import torch
import torch.nn as nn

# Data: house size (sq ft) → price ($1000s)
X = torch.tensor([[1000.], [1500.], [2000.], [2500.], [3000.]])
y = torch.tensor([[200.], [300.], [400.], [500.], [600.]])

# Model, loss, optimizer
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
criterion = nn.MSELoss()

# Train
for epoch in range(1000):
    loss = criterion(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Predicted price for 1800 sq ft: ${model(torch.tensor([[1800.]])).item():.1f}k")
```

线性回归的动态图 (with mini-batches 和 early stopping ) $$y = xA^T + b$$

```python
import torch
import torch.nn as nn

# Data: [size, bedrooms, age] → price ($1000s)
X = torch.tensor([
    [1000., 2., 10.],
    [1500., 3.,  5.],
    [2000., 3., 20.],
    [2500., 4.,  2.],
    [3000., 5.,  8.],
    [3500., 4., 15.],
])
y = torch.tensor([[200.], [300.], [400.], [500.], [600.], [650.]])

model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
criterion = nn.MSELoss()

BATCH_SIZE = 2
PATIENCE = 100        # stop if no improvement for 100 epochs
best_loss = float("inf")
no_improve_count = 0

for epoch in range(5000):
    # --- for: iterate over mini-batches ---
    for i in range(0, len(X), BATCH_SIZE):
        X_batch = X[i : i + BATCH_SIZE]
        y_batch = y[i : i + BATCH_SIZE]

        loss = criterion(model(X_batch), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- if: early stopping check ---
    epoch_loss = criterion(model(X), y).item()
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        no_improve_count = 0
    else:
        no_improve_count += 1

    if no_improve_count >= PATIENCE:
        print(f"Early stopping at epoch {epoch}, loss={best_loss:.4f}")
        break

# Predict: 1800 sq ft, 3 bedrooms, 7 years old
sample = torch.tensor([[1800., 3., 7.]])
print(f"Predicted price: ${model(sample).item():.1f}k")
```

**PyTorch的作用就是定义，执行和优化计算图** , 但是复杂的模型会依赖input做逻辑判断（）