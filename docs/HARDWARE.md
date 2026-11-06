# Demo hardware (inference only)

## Measured

`small` (350M): **56 ms per 80 ms step on a 4-core cloud CPU, fp32, 1 call**.
The codec (Mimi) adds its own encode/decode time per frame. Measure it on
the target box.

## Recommended box for 10–20 concurrent calls (~₦1.5M–2.5M)

| Part | Pick |
|---|---|
| Mini PC | Ryzen 7 / Core i7, 8+ cores |
| RAM | 32 GB |
| GPU | RTX 4060 / 4060 Ti (8–16 GB) via OCuLink/eGPU, or a small-form-factor desktop |
| Storage | 1 TB NVMe (call recordings for the flywheel) |
| Power | inverter + solar or a UPS (an outage kills the line) |
| Network | fibre or a 4G router; or a GSM gateway for the offline setup |

## Capacity, honestly

* **CPU only**: ~1 call in real time with the `small` model.
* **RTX 4060 class, fp16**: an estimated 10–20 calls. Each call is one batch-1
  step every 80 ms, behind a lock in this first version.
* **100 calls** needs batched stepping across calls (one forward for all
  active calls). That's the next engineering task, with a bigger GPU, or two boxes.
