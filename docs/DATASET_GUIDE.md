# Dataset Guide: Kaggle Robot Predictive Maintenance

## Source & Background
- **Source**: Kaggle Robot Predictive Maintenance Competition
- **Physical System**: 6-axis articulated robot arm executing industrial manipulation sequences:
  - "Pick up and place"
  - "Not moving"
  - "Move motor 6 -> motor 1 sequentially"
  - "Moving one motor"
  - "Turning motor 6"
- **Sampling Rate**: ~10 Hz (approx. 0.1 second per observation)

## Folder & File Structure
```
robot-predictive-maintenance/
├── training_data/training_data/
│   ├── Test conditions.xlsx   # Run metadata, operating conditions, motor failure flags
│   ├── 20240105_164214/       # Test run directory containing 6 synchronized CSV files
│   │   ├── data_motor_1.csv
│   │   ├── ...
│   │   └── data_motor_6.csv
│   └── ... (23 total train run folders)
└── testing_data/testing_data/
    ├── Test conditions.xlsx
    └── ... (8 unlabeled test run folders for competition submission)
```

## Schema & Column Definitions

| Column | Type | Physical Range | Meaning & Diagnostic Role |
| :--- | :--- | :--- | :--- |
| `time` | float64 | Continuous (s) | Elapsed operating time within the test run sequence. |
| `position` | float64 | 0 to 999 steps | Rotary encoder angle of the motor shaft. Negative values (-32k) indicate transient serial bus glitches. |
| `temperature` | float64 | 21 to 258 °C | Motor internal coil temperature. Values above 50°C suggest thermal stress; over 100°C indicates severe friction or overcurrent. |
| `voltage` | float64 | 6000 to 7600 mV | Operational voltage across motor driver circuit (nominal ~7.2 V). Voltage drops below 6900 mV under excessive load. |
| `label` | int64 | {0, 1} | Binary health ground truth (0 = Normal, 1 = Fault). In test directories, `label` is empty/withheld. |

## Motor-by-Motor Fault Breakdown (Training Data)
- **Motor 1**: 39,309 samples | 1,349 faults (3.43%)
- **Motor 2**: 39,309 samples | 6,732 faults (17.13%)
- **Motor 3**: 39,309 samples | 127 faults (0.32%)
- **Motor 4**: 39,309 samples | 6,739 faults (17.14%)
- **Motor 5**: 39,309 samples | 184 faults (0.47%)
- **Motor 6**: 39,309 samples | 1,932 faults (4.91%)
- **Overall Total**: 235,854 motor-level samples | 17,063 fault labels (7.24% positive class rate).
