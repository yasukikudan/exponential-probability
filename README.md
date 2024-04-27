# ExP2 指数確率関数

### 概要
`ExP2` 指数確率関数（ExPonential Probability Function）は、指数関数をベースにした活性化関数で、ニューラルネットワークのトレーニングに適した非線形性を提供します。この関数は、正の入力に対しては確率的な特性を反映し、負の入力に対しては損失関数として機能します。このような特性は、特に不均衡なデータや特異なデータポイントが存在する場合に有効です。

### 関数の定義

ExP2関数は以下のように定義されます：

- **\( x >=0 \)の場合：**
```math
  f(x) = 1 - e^{-x}
```

- **\( x <  0 \)の場合：**
```math

  f(x) = -(1 - e^x) \cdot (1 - x)
```
この定義により、関数は正の入力で確率的な特性を示し、負の入力でペナルティ機能として振る舞います。

### 導関数

ExP2関数の導関数は以下の通りです：

- **\( x >= 0 \)の場合：**
```math
  f'(x) = e^{-x}
```

- **\( x <  0 \)の場合：**
```math
  f'(x) = -x \cdot e^x + 1
```

これらの導関数は、勾配降下法などの最適化アルゴリズムで使用する際に効率的な学習を促進します。

### PyTorchコード
```python
import torch
import torch.autograd as autograd

class ExP2(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        s = torch.sign(x)
        exp_component = torch.exp(-torch.abs(x))
        y = (1 - exp_component) * s * (torch.relu(-x) + 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        s = torch.sign(x)
        dx = torch.exp(-(x * s))
        itr = -(s - (s * s)) *0.5
        sc = ((1 - itr) + (itr * x) * -1)
        dx = (dx * sc + itr)
        grad_input = dx * grad_output
        return grad_input
```

### 出力と微分値のプロット
以下に、`ExP2`関数の出力と微分値のプロットを示します。これにより、関数の挙動を直感的に理解することができます。

```python
import matplotlib.pyplot as plt
import torch

x = torch.linspace(-4, 4, 100, requires_grad=True)
y = ExP2.apply(x)
y.backward(torch.ones_like(x))
grad = x.grad

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x.detach().numpy(), y.detach().numpy(), label='ExP2 Output')
plt.title('Function Output')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x.detach().numpy(), grad.detach().numpy(), label='ExP2 Derivative')
plt.title('Function Derivative')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.grid(True)
plt.legend()

plt.show()
```
###実際の出力と微分値
