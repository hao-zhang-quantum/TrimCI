# ⟨S²⟩ Diagnostic

Reports `s_squared = ⟨Ψ|S²|Ψ⟩` and `spin = S` (from S(S+1) = ⟨S²⟩).

```python
result = run_expansion(..., measure_s2=True)        # → result['s_squared'], result['spin']
result = run_full(..., config_dict={'measure_s2': True})  # → result[3]['s_squared']

# Standalone:
from trimci.TrimCI_runner.s2_probe import measure_s2
s2, s = measure_s2(alphas=a, betas=b, coeffs=c)
```

Pure diagnostic. Auto-saved to results JSON. ~0 cost vs Davidson.
Singlet ⟨S²⟩=0, triplet=2, quintet=6, … anything off-grid = spin contamination.
