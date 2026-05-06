# handeye_sim_project

A small Python simulation project for Eye-in-Hand hand-eye calibration.

## Run

```bash
pip install -r requirements.txt
python main.py
```

In VSCode, open the project folder and press F5. The default debug config runs `main.py`.

## Structure

```text
main.py
handeye_sim/
  core/              # SE(3), SO(3), pose utilities
  simulation/        # ground truth scene generation and noisy observations
  calibration/       # OpenCV hand-eye wrapper
  evaluation/        # error metrics and reporting
  test_blocks/       # test1(), test2(), ... non-pytest entry blocks



```

