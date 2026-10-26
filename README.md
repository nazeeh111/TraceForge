# TraceForge

A Python workspace for electromagnetic trace processing, correlation metrics and model-based analysis.

[Try standalone metrics](#start-with-the-standalone-metrics) · [Workspace map](#workspace-map) · [Full environment](#environment-and-workflow) · [Source terms](#source-terms)

## Start with the standalone metrics

The two metric accumulators can run with **NumPy alone**, without TensorFlow, Redis, capture hardware or experimental data. From the repository root:

```sh
python -m pip install numpy
python - <<'PY'
import numpy as np
from emma.metrics.correlationlist import CorrelationList
from emma.metrics.distancelist import DistanceList

x = np.array([1., 2., 3., 4.])
y = 2 * x + 3
reverse = x[::-1]
c = CorrelationList((2,))
c.update(0, x[:2], y[:2])
c.update(0, x[2:], y[2:])
c.update(1, x, reverse)
d = DistanceList((1,))
d.update(0, x, reverse)
print(f'positive correlation: {c[0]:.1f}')
print(f'negative correlation: {c[1]:.1f}')
print(f'absolute-distance sum: {d[0]:.1f}')
PY
```

```text
positive correlation: 1.0
negative correlation: -1.0
absolute-distance sum: 8.0
```

Run the standalone accumulator tests with `python -m unittest discover -s tests -v`; they also need only NumPy.

This checked synthetic example shows incremental correlation accumulation and the **sum** of absolute differences. It is not an experimental trace, model result or key-recovery claim. The [correlation](emma/metrics/correlationlist.py) and [distance](emma/metrics/distancelist.py) implementations are directly inspectable. Correlation needs observations with nonzero variation to be meaningful; an untouched accumulator is not a valid result.

## Workspace map

| Component | Role |
| --- | --- |
| `emma/processing/` | Trace transformations and processing operations |
| `emma/metrics/` | Correlation and distance accumulators |
| `emma/ai/` | Model definitions, losses and training helpers |
| `emma/io/` | Dataset and trace representations |
| `emma.py` | Existing analysis command-line entry point |
| `emcap.py` | Existing capture entry point |

## Environment and workflow

The pinned requirements target an older Python ecosystem, including TensorFlow 1.14 and Keras 2.2.5. Use a matching isolated environment when reproducing existing experiments. Package names, command names and configuration sections are preserved. Existing saved correlation objects remain readable; new saves use centered floating-point statistics to avoid cancellation when observations have large offsets. Precision already lost in old accumulated sums cannot be recovered.

The full `emma.py` workflow invokes `redis-cli flushall` before analysis and purges queued tasks afterward, including when `--local` is selected. Run it only with a dedicated, disposable research Redis instance; `--local` describes task execution, not isolation from Redis side effects. The standalone metrics example above does not import or run that entry point.

[Configuration and workflow](docs/WORKFLOW.md) covers the existing local and worker-based commands. [Preservation checks](COMPATIBILITY.md) records the narrower offline validation performed for this publication.

The bundled ASCAD companion files retain their separate [notice](ascad/LICENSE_ASCAD).

## Source terms

First-party source is available under the [MIT license](LICENSE). Separate bundled components and external modules retain their own terms.
