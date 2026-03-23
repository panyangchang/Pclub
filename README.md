# Pclub

## 双层转角石墨烯紧束缚模型 / Twisted Bilayer Graphene Tight-Binding

本仓库实现了任意转角（包括非公度角）双层石墨烯的实空间紧束缚哈密顿量构建方案。

This repository implements a complete real-space tight-binding Hamiltonian for twisted bilayer graphene (TBG) at **arbitrary twist angles**, including incommensurate (non-magic) angles.

---

### 功能特性 / Features

| 功能 | 描述 |
|------|------|
| 任意转角支持 | 适用 0–30°，含非公度角（1.08°, 1.47°, 2.0°等） |
| 可扩展原子编号 | 唯一整数 ID + 反向查找表，便于缺陷/应变/边界调制 |
| Moon–Koshino 模型 | 层内 + 层间 Slater–Koster hopping 函数 |
| 稀疏矩阵 | scipy.sparse CSR 格式，支持 ≥10⁴ 原子 |
| DOS / LDOS 计算 | 高斯展宽，全局+局域态密度 |
| 扰动支持 | 门电压、子晶格错位、任意 on-site 能量 |
| 完整测试套件 | 91 个 pytest 测试，覆盖物理正确性验证 |

### 快速开始 / Quick Start

```bash
pip install numpy scipy matplotlib pytest
```

```python
from tbg import generate_tbg_geometry, build_hamiltonian, compute_dos
import numpy as np

# 生成 1.08° 魔角石墨烯（约 1250 个原子）
bot, top = generate_tbg_geometry(twist_angle_deg=1.08, n_cells=8)
H = build_hamiltonian([bot, top])
print(f"N = {H.shape[0]}, NNZ = {H.nnz}")

# 计算态密度
energies, dos = compute_dos(H, eta=0.02)
print(f"DOS at E=0: {np.interp(0.0, energies, dos):.4f} states/eV/atom")
```

### 运行示例 / Run Example

```bash
PYTHONPATH=. python examples/compute_tbg.py
```

### 运行测试 / Run Tests

```bash
pytest tests/ -v
```

### 目录结构 / Repository Structure

```
tbg/
├── __init__.py         # 包入口
├── lattice.py          # 原子生成与编号策略
├── hopping.py          # Slater–Koster hopping 参数
├── hamiltonian.py      # 稀疏哈密顿量组装
├── dos.py              # DOS / LDOS 计算
└── utils.py            # 摩尔几何、验证、扰动工具
tests/
├── test_lattice.py     # 晶格与编号测试
├── test_hamiltonian.py # 哈密顿量物理验证
├── test_dos.py         # DOS 数值测试
└── test_utils.py       # 工具函数测试
examples/
└── compute_tbg.py      # 端到端演示脚本
docs/
└── TBG_tight_binding.md  # 完整数学文档
```

### 数学文档 / Mathematical Documentation

完整的数学推导、参数表和验证方案见：[docs/TBG_tight_binding.md](docs/TBG_tight_binding.md)

Full mathematical derivation, parameter table, and validation methodology: [docs/TBG_tight_binding.md](docs/TBG_tight_binding.md)

### 参考文献 / Key References

- Moon & Koshino, *Phys. Rev. B* **87**, 205404 (2013) — Slater–Koster model
- Bistritzer & MacDonald, *PNAS* **108**, 12233 (2011) — Magic angle & flat bands
- Lopes dos Santos et al., *PRL* **99**, 256802 (2007) — Commensurate angles
