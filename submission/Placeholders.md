# MTAH 2027 — Multimodal Football Possession Engine
## Technical Methodology & System Architecture Document

> **Note:** This is an earlier draft. The final technical document is `Placeholders.pdf`.
> The final submission adds four rules that this draft does not describe:
> `--flight-barrier` (the ball cannot change team in fast, undeflected flight),
> `--confirm-turnover-frames 50` (an opponent spell under 2 s between two spells of the same
> team goes back to that team), `--turnover-at-receiver` (a change of team starts when the new
> team controls the ball) and `--broadcast-fill` (missing tactical ball frames are filled from the
> broadcast tracking). `bash run.sh` runs all four.

* **Challenge:** Metrica Sports × FC Barcelona × Mobile World Capital Barcelona (*More Than A Hack 2027*, Talent Arena Barcelona)  
* **Target Match:** FC Barcelona vs. Athletic Club (`BAR-ATH`, 199,319 frames @ 25.0 fps, ~106 minutes across two halves)  
* **Target Alphabet:** `{'H', 'A', 'D', 'X'}` (Home, Away, Dead-ball, Abstain)  
* **Team Name:** `Placeholders`  
* **Team Members:**
  * Rushabh Shah (India)
  * Karan Kewalramani (India)
  * Samarth Gupta (India)
  * Madhur Kalia (India)

---

## 1. The Approach: What We Built & Why

### 1.1 The Core Football Axiom
> **A touch is not possession. Proximity is not possession. A deflection is not possession.**  
> Under professional match-tagging conventions (StatsBomb, Opta, Metrica) and tournament guidelines, possession represents *controlled team agency*. A team retains possession while the ball is travelling, loose, or contested, and loses possession *only* when the opposing team establishes settled motor control, or when play becomes dead under IFAB Laws.

### 1.2 System Architecture
To enforce this principle without overfitting to individual match incidents, we formulated possession tracking as a **physically-grounded, multimodal sequence model** operating across six specialized stages:

```
[EPTS Optical Tracking & Referee XML]
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: Data Ingestion & Kinematic Conditioning        │
│ • Survey coordinate normalization to metric pitch bounds│
│ • Biomechanical sprint ceiling filter (v < 14.0 m/s)    │
│ • Ballistic constant-acceleration micro-gap spline      │
└─────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: Game Phase & Episode Evidence Engine           │
│ • Kinematic Stoppage Hysteresis (P75 speed < 1.4 m/s)   │
│ • Fast Ball Transit Aerial Detection (v > 6.5 m/s)      │
│ • Set-Piece Restart Kick Backtracking & Impulse Gating  │
│ • Sweeper-Keeper Jurisdiction & 18-Yard Penalty Box     │
└─────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 3: Player Interaction & Control Engine            │
│ • Foot-Ball Acceleration Strike Detection (v > 2.5 m/s) │
│ • Radial Closing & Continuous Carry Tracking            │
│ • Accidental Deflection & Header De-amplification       │
│ • Contested 50-50 Duel Modulation (|dH - dA| < 0.8m)    │
└─────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 4: Spatial Likelihoods & Stoppage Priors          │
│ • Softmin Multi-Player Spatial Aggregation (tau = 1.5m) │
│ • Open-Space Loose Ball Attenuation (d > 2.8m ramp)     │
│ • Decoupled Undeflected Flight Gate (90% suppression)   │
│ • In-Pitch -40.0 nat Dead-Ball Barrier in Open Play     │
│ • IFAB Law 9 Out-of-Bounds Geometric Hard Gate          │
└─────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 5: Context-Aware Log-Viterbi Sequence Decoder     │
│ • Global MAP Sequence Recovery via Bellman Recursion    │
│ • Premature Stoppage Exit Penalty (-8.0 nats)           │
│ • Confirmed Restart Kick Transition Snap (-0.1 nats)    │
│ • Unconfirmed Turnover Resistance Penalty (-6.0 nats)   │
└─────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 6: Minimum-Dwell Pruning & Output Formatting      │
│ • Controlled Possession Dwell Floor (min_poss = 30f)    │
│ • Stoppage Dead Dwell Floor (min_dead = 35f)            │
│ • "La Pausa" Same-Team Tempo Protection (min_same = 45f)│
│ • Intentional Action Fast-Pass Shield (min_action = 10f)│
│ • Dead-Ball Setup Touch Merger (D -> H/A -> D)          │
│ • Metadata Window Clamping & Full In-Play Coverage      │
└─────────────────────────────────────────────────────────┘
                  │
                  ▼
[Certified Output: Placeholders.csv (199,319 rows)]
```

### 1.3 Why This Rather Than Something Else?
1. **Why Not Naive Spatial Proximity?**  
   Assigning possession to whichever player is closest to the ball causes catastrophic failure on aerial crosses, long clearances, deflections, and stoppage maintenance, producing fragmented micro-possessions of under 2 seconds duration rather than coherent team phases.
2. **Why Not Supervised Deep Learning (LSTM / Transformers)?**  
   With unlabelled tracking telemetry and event metadata for an unseen match, training complex neural architectures risks severe data memorization and unphysical state predictions. Furthermore, deep sequence models lack physical grounding: they cannot guarantee that out-of-bounds balls are marked Dead or that goalkeepers possess special handling privileges inside the 18-yard box.
3. **Why A Generative Factorized HMM + Contextual Trellis?**  
   The forward-backward Log-Viterbi recursion finds the exact Maximum A Posteriori (MAP) sequence over the entire 106-minute timeline in under 1 second. It combines spatial likelihoods with physical inertia, guaranteeing unbroken sequence continuity without frame-by-frame flickering.

---

## 2. Match Assets: Used vs. Dropped

### 2.1 Assets Used in the Pipeline
1. `BAR-ATH_tactical_atd.txt` (EPTS Optical Tracking Stream):
   * 199,319 frames sampled at 25.0 Hz (~106 minutes).
   * Provided 2D pitch coordinates for all 22 outfield players, goalkeepers, and ball.
2. `BAR-ATH_tactical_atd.xml` (Tactical Camera Calibration Survey):
   * Survey pitch dimensions: $105.3\text{m} \times 68.0\text{m}$.
   * Period boundaries: Half 1 $[11185 \dots 90651]$, Half 2 $[110808 \dots 190197]$.
3. `BAR-ATH_tactical_smart_tagging.xml` (Tactical Event Metadata):
   * Official XML stream containing tagged set-piece event intervals (kickoffs, corners, fouls/free kicks, penalties, throw-ins, and goals) with start and end timestamps.

### 2.2 Assets Dropped (and Why)
1. `BAR-ATH_broadcast_atd.txt` (Broadcast Camera Tracking):  
   * **Dropped:** Broadcast tracking suffers from frequent camera cuts, replays, zoom-ins, and player coordinate dropouts. The tactical optical camera provides a fixed, full-pitch perspective with continuous tracking.
2. `BAR-ATH_broadcast_atd.xml`:  
   * **Dropped:** Unneeded once the broadcast tracking feed was discarded.
3. `BAR-ATH_broadcast_homography.csv` & `BAR-ATH_tactical_homography.csv`:  
   * **Dropped:** The ATD text stream already provided pre-projected metric coordinates calibrated to the pitch survey, making raw camera homography matrices redundant.
4. **External Match Videos & Third-Party Statistics:**  
   * **Dropped:** No external video footage, Opta/StatsBomb event dumps, or third-party web scrapers were used. The system is 100% self-contained on the official challenge assets.

---

## 3. Training Data & Label Provenance

In strict compliance with competition rules:
* **Zero Label Tuning / Zero Supervised Fitting:** No thresholds, decay factors, transition penalties, or dwell limits were fitted or tuned against match labels or external statistics.
* **Zero Human Annotations:** No human watching video footage manually created label lookup tables or frame-specific rules.
* **Physics & Literature Grounding:** All parameters stand strictly on IFAB Laws of the Game, athletic movement ceilings (Weyand et al., 2000), and peer-reviewed sports science literature (Spearman et al., 2017; Link et al., 2016).

---

## 4. Possession Rules Implemented

### 4.1 Loose Balls in Open Grass
When the ball is travelling in open grass ($d_{\min} > 2.80\text{m}$), spatial proximity likelihoods decay exponentially toward zero. The HMM sequence persistence automatically retains possession with the passing team until an opponent establishes physical contact.

### 4.2 High-Speed Aerial Flight Passes
When ball velocity exceeds $6.50\text{ m/s}$ ($23.40\text{ km/h} = \mathbf{2:34\text{ min/km}}$) and directional deflection is minimal ($\theta < 25^\circ$), 2D ground-plane proximity is suppressed by 90% (retention factor = 0.10). This prevents opposing players standing beneath an aerial lob from falsely triggering a turnover while the ball is in mid-air.

### 4.3 Contested 50-50 Duels
When opposing players contest a ball within physical tackle reach ($1.80\text{m}$) and proximity difference is negligible ($|d_H - d_A| < 0.80\text{m}$), spatial likelihoods are dampened by 40%. The prior team retains possession through sequence inertia, preventing rapid alternation during physical scrambles.

### 4.4 Set-Piece Restarts & Setup Touches
* **Setup Touches:** Casual touches before a restart (e.g. a defender rolling the ball back to the goalkeeper) are merged into Dead Ball if players are walking ($v < 1.95\text{ m/s} = 7.02\text{ km/h} = \mathbf{8:33\text{ min/km}}$) and no legal kick occurred.
* **Restart Impulse:** Once a legal restart kick is detected ($v > 2.50\text{ m/s} = 9.00\text{ km/h} = \mathbf{6:40\text{ min/km}}$), an immediate bonus snaps the match into active live play with zero lag.

### 4.5 Empirical Minimum Dwell & Intentional Action Shield
* **Possession Dwell Floor:** In our post-processing stage, sustained team possession uses a 30-frame ($1.20\text{s}$) dwell heuristic to prune transient tracking noise and momentary contested touches that do not constitute controlled possession.
* **Stoppage Dwell Floor:** Dead-ball intervals use a 35-frame ($1.40\text{s}$) dwell heuristic reflecting the physical latency required for players to retrieve the ball and initiate a restart following a whistle or stoppage.
* **The Intentional Action Shield:** When Stage 3 detects an intentional strike, first-time pass, or shot, the dwell floor drops to 10 frames ($0.40\text{s}$), protecting deliberate quick volleys and clearances from being filtered.
* **In-Play Coverage Policy:** On active gameplay intervals (Half 1 and Half 2), our model treats unclassified frames as Dead Ball rather than abstaining with 'X', ensuring unbroken match coverage across all live minutes.

---

## 5. Master Running Pace Reference Table

| Metric / Parameter | Velocity (m/s) | Speed (km/h) | Running Pace (min/km) | Physical / Tactical Function |
| :--- | :---: | :---: | :---: | :--- |
| `BALL_STATIONARY_FLOOR` | $1.00\text{ m/s}$ | $3.60\text{ km/h}$ | **16:40 min/km** | Stationary ball threshold; radial closing gate floor |
| `PLAYER_WALKING_SPEED` | $1.40\text{ m/s}$ | $5.04\text{ km/h}$ | **11:54 min/km** | Stoppage entry 75th percentile team walking speed |
| `SETUP_TOUCH_WALKING_CEILING` | $1.95\text{ m/s}$ | $7.02\text{ km/h}$ | **8:33 min/km** | Player walking ceiling during dead-ball setup maintenance |
| `FIFA_ZONE_1_2_BOUNDARY` | $2.00\text{ m/s}$ | $7.20\text{ km/h}$ | **8:20 min/km** | FIFA Zone 1 (walking) to Zone 2 (jogging) transition |
| `PLAYER_JOGGING_SPEED` | $2.20\text{ m/s}$ | $7.92\text{ km/h}$ | **7:35 min/km** | Active jogging surge; stoppage exit 75th percentile threshold |
| `INTENTIONAL_KICK_IMPULSE_FLOOR`| $2.50\text{ m/s}$ | $9.00\text{ km/h}$ | **6:40 min/km** | Strike acceleration triggering Intentional Action Shield |
| `SHORT_PASS_ROLLING_SPEED` | $2.80\text{ m/s}$ | $10.08\text{ km/h}$ | **5:57 min/km** | Open-space loose ball attenuation distance and rolling speed floor |
| `DEFLECTION_REBOUND_FLOOR` | $4.00\text{ m/s}$ | $14.40\text{ km/h}$ | **4:10 min/km** | Spatial decay scale ($\sigma = 4.0\text{m}$) and deflection rebound floor |
| `DRIVEN_SET_PIECE_LAUNCH` | $4.50\text{ m/s}$ | $16.20\text{ km/h}$ | **3:42 min/km** | Minimum launch speed for driven set pieces and crosses |
| `INTENTIONAL_REDIRECT_SPEED` | $5.00\text{ m/s}$ | $18.00\text{ km/h}$ | **3:20 min/km** | Closing sprint speed and volley redirect exit speed |
| `SWEEPER_KEEPER_SPRINT_CLEAR` | $6.00\text{ m/s}$ | $21.60\text{ km/h}$ | **2:47 min/km** | Sprint clearance velocity by sweeper-keeper outside 18-yard box |
| `DRIBBLE_VELOCITY_CEILING` | $6.50\text{ m/s}$ | $23.40\text{ km/h}$ | **2:34 min/km** | Continuous foot dribble ceiling before entering ballistic flight |
| `AERIAL_FLIGHT_TRANSIT_FULL` | $10.00\text{ m/s}$| $36.00\text{ km/h}$ | **1:40 min/km** | Certain aerial pass transit ($p_{\text{transit}} = 1.0$) |
| `PLAYER_MAX_SPRINT_CEILING` | $14.00\text{ m/s}$| $50.40\text{ km/h}$ | **1:11 min/km** | Physical human speed barrier; filters tracking ID swaps |

---

## 6. Limitations & Future Work

1. **Unobserved Z-Dimension in 2D EPTS:**  
   The optical tracking feed provides only $(x, y)$ coordinates. While ballistic 2D velocity drop models flight accurately in most situations, true 3D triangulation would improve aerial header attribution in crowded goalmouth scuffles.
2. **Crowded Goalmouth Occlusions:**  
   During corner kicks with 16+ players clustered inside the 6-yard box, proximity alone is ambiguous. Incorporating player jersey color or body orientation would sharpen contested corner attribution.
3. **Automated Referee Whistle Audio Integration:**  
   Currently, stoppage evidence relies on collective player deceleration and XML timestamps. Processing raw ambient pitch audio for referee whistle frequencies would pinpoint the exact millisecond of stoppage onset.

---

## 7. Reproduction Instructions

### 7.1 Setup
```bash
pip install -r requirements.txt
```

### 7.2 Run Test Suite
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
*(All 204 unit tests pass in 0.17 seconds).*

### 7.3 Generate Submission CSV
```bash
bash run.sh
```
*(Generates `submission/Placeholders.csv` across all 199,319 frames in ~13 seconds).*

### 7.4 Validate Format
```bash
python3 -m src.validate_submission submission/Placeholders.csv
```
*(Confirms exactly 199,319 rows, alphabet in `{'H', 'A', 'D', 'X'}`).*
