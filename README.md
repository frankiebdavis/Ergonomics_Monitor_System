# ​ Desk Posture & Ergonomics Monitor

A webcam-based ergonomics coach. Detects slouching, shoulder drop, arm flare, elbow height, forearm tilt, wrist extension/deviation, and keyboard contact, offering live feedback to keep you typing posture-friendly. RSI's (repetitive strain injuries) due to poor typing ergonomics cost employers dearly, from lost workdays to treatment expenses. Employees who sit comfortably and safely stay engaged longer, reduce fatigue, and produce better work. This system helps intercept poor posture *before* it becomes an injury.

---

## ​ Problem

###  Why RSI's Need real Attention

- **What is an RSI?** A Repetitive Strain Injury (RSI) is damage to muscles, tendons, or nerves caused by repetitive motion or sustained awkward posture — most often in the hands, wrists, neck, and shoulders of desk workers. Common examples include carpal tunnel syndrome, tendonitis, and chronic neck/shoulder pain.
-  **Why ergonomics matter:** Good ergonomics (neutral wrist angles, upright posture, proper elbow height, and balanced trunk alignment) reduce the strain that accumulates with long hours of typing and static sitting. Without these habits, micro-injuries build up until they become painful, costly medical conditions.
- **RSIs are widespread and impactful.** In 2021, **9% of U.S. adults** reported having a repetitive strain injury in the past 3 months, and about 44% of those cases led to at least **24 hours of limited activity** (CDC). **Office workers are especially vulnerable** due to long hours of repetitive typing and static posture (CDC).  
- **RSIs seriously disrupt productivity and comfort.** They’re the workplace injury that causes the **longest absences**, often averaging **17 days away from work**, with **billions of dollars** lost annually in productivity and healthcare (BackSafe).
 
Yet despite this, many companies treat ergonomics as a **one-time training task**, with little ongoing tracking. This leads to a **reactive approach** — action is only taken once damage is already done. Through tracking sessions, commpanies could monitor the success of their training programs.

Some companies engage in **participatory ergonomics** programs, where employees actively engage in shaping ergonomic practices. By giving workers real-time feedback and measurable posture data, companies can move from **reactive to preventive** ergonomics — reducing RSIs, improving efficiency, and even lowering insurance costs.

---

## ​ Method

This system turns ergonomics into real-time, contextual coaching using only a **single webcam**.

- **6-second setup:** You sit upright and type normally while the system captures calibration baselines (trunk angle, head tilt, wrist plane).  
- **Real-time tracking:** Monitor head/neck, trunk lean, shoulders, elbows, wrists, and hands using MediaPipe.  
- **Smart posture analysis:** We calculate angles (TFA, RFH, flares, wrist deviations, elbow height) and apply thresholds with **hysteresis** and **adaptive smoothing (EMA/medians)**.  
- **Easy coaching UI:** A banner reads **GOOD** or **ADJUST**, gives a simple tip (e.g. “Sit tall”), shows **session time & % good**, and overlays useful visual cues (near/far arms, trunk/keyboard plane).  
- **Recalibration (`toggle r`):** At any time, you can reset posture and wrist baselines if your seating position changes, ensuring the system adapts to real-world desk adjustments.  
- **Advanced statistics view (toggle 'a'):** Displays detailed numeric metrics (e.g., trunk flexion angle, neck slope, wrist extension/deviation, elbow height relative to keyboard plane, shoulder drop %) in a side panel. Useful for ergonomics researchers, data collection, or validating posture training programs. 

---

## Results

- Achieved stable, real-time tracking of trunk, head, and arm posture at ~30 FPS using a standard webcam.  
- Reduced false positives through adaptive smoothing, hysteresis, and per-user calibration/recalibration.  
- Provided actionable coaching feedback (“Sit tall”, “Raise elbows”, “Straighten wrist”) that aligns with ergonomic best practices for preventing RSIs.  
- Created a low-cost system that could be deployed in offices as part of participatory ergonomics programs to monitor posture, encourage healthy habits, and reduce the long-term risk of RSI's.  

---

## ​ Visual and Image Demos

Video example of launching and using the system:

[![Watch the tutorial](https://img.youtube.com/vi/2mi29EBLYoI/hqdefault.jpg)](https://youtube.com/shorts/2mi29EBLYoI?feature=share)

Example of bad posture detection:


Example of good posture detection:


---

## ​ Skills Demonstrated

- Python  
- MediaPipe  
- OpenCV  
- Signal processing (EMA, median filters, hysteresis)  
- Human factors engineering
- System integration and debugging  
- Applied ergonomics and workplace safety  

---

##  Citations

- CDC: https://www.cdc.gov/nchs/data/nhsr/nhsr189.pdf
- BackSafe: https://www.backsafe.com/repetitive-strain-injuries/

