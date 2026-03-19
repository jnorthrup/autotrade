# Autotrade

**NO HEURISTICS. This system operates on pure model-driven logic. No hardcoded thresholds, no arbitrary constants, no rule-based 'patience', and no human-defined decision dampers are permitted. Every action must be an emergent property of the model's learning and the mathematical first principles of the environment.**

GAT-based force-directed coin graph trading with HRM (Hierarchical Reasoning Model).

## Symmetric World Model Architecture

The system treats past and future as a single, coherent physical entity. The resolution of the model's memory (Fisheye Lens) is mathematically mirrored into the resolution of its foresight (Staged Horizons).

```mermaid
graph TD
    subgraph Input_Physics ["PAST: The Fisheye Lens (Memory)"]
        A[Macro History: Y Depth] --> B{Fisheye Buckets: W}
        C[Indicator Constants: A, B, C Curves] --> B
        D[Raw ClosePrice] --> B
        E[Volume/Candle Metadata] --> B
        B --> F[Fisheye Sparkline: X Pixels]
    end

    subgraph HRM_Core ["THE BRIDGE: Recurrent Reasoning"]
        F --> G[z_H: High-Level Macro State]
        G <--> H[z_L: Low-Level Detail State]
        H -- "H/L Cycles" --> G
    end

    subgraph Output_Physics ["FUTURE: The Symmetric Projection (Foresight)"]
        G --> I[Reflected Horizons: W2]
        H --> I
        I --> J[Staged Predictions: X2 Pixels]
        K["{A, B, C}2 Curve Hyperparams"] --> J
        J --> L[Predicted log-returns]
    end

    style Input_Physics fill:#f9f,stroke:#333,stroke-width:2px
    style HRM_Core fill:#bbf,stroke:#333,stroke-width:2px
    style Output_Physics fill:#dfd,stroke:#333,stroke-width:2px
```

### Key Technical Concepts:
- **Non-Frame Inputs**: The edge predictor integrates `VOLUME` and `CANDLE` junctures as exogenous variables, while `VOLUME` influence is implicitly baked into the historical `CLOSEPRICE` trajectory.
- **Hyperbolic X/Y**: The fisheye resolution (`X`) holds the historical depth (`Y`) using a power-law curvature (`A, B, C`) to emphasize recent entropy while maintaining macro-scale context.
- **Symmetric Mirroring**: The output horizons (`X2, Y2`) are the exact reflection of the input buckets. The model must bridge the gap between the macro "vanishing point" and the micro "present tense."
- **Entropy Exploitation**: Using a Hyperbolic Loss (Power 2.0+), the model ignores linear "sinewave" wiggles and rewards the capture of high-intensity outliers (spikes).

## Installation

```bash
uv sync
```

## Usage

Run the main simulation:
```bash
uv run autotrade
```

Run autonomous research (ANE-powered):
```bash
cd src/autotrade/research && uv run train.py
```
