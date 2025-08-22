![CI](https://github.com/samuel-pedrielli/ego-centric-agi/actions/workflows/ci.yml/badge.svg)
## Quickstart (toy)

```bash
pip install -r requirements.txt
python -m EAFCode.toy.run_toy --mode baseline --steps 300 --T 64
python -m EAFCode.toy.run_toy --mode sweep --steps 300 --T 64
python -m EAFCode.toy.plot
```
> See also the [Reproducibility Guide](./REPRODUCIBILITY.md) and the CI-generated artifacts in the **Actions** tab ([CI workflow](../../actions/workflows/ci.yml)).

# ego-centric-agi
We present an ego-centric architecture for AGI safety that achieves alignment through self-aware identity preservation. By structuring the AI's ego with benevolence at its core and implementing self-preservation mechanisms, we create a system that maintains beneficial behavior as an intrinsic property of its identity.
