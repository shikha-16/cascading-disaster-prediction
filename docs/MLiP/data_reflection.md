# Data Reflection: Cascading Disaster Prediction

**10-718 Machine Learning in Practice — Spring 2026**

---

## 1. Project Status

Our project addresses the question: *Given a primary weather event, can we predict which types of secondary events it will trigger?* We frame this as a multilabel classification problem — for each storm event, the model predicts zero or more secondary event types (e.g., a hurricane may trigger Flash Flooding, High Wind, and Storm Surge simultaneously).

We have completed a fully automated data pipeline (`prepare_data.py`), exploratory analysis (`01_cascade_eda.ipynb`), per-label XGBoost models with negative subsampling (`02_xgboost_multilabel_training.ipynb`), and sklearn multilabel baselines including OneVsRest Logistic Regression, Classifier Chains, and MLPClassifier (`03_sklearn_multilabel_baselines.ipynb`). Ongoing work includes fairness analysis using CDC Social Vulnerability Index data.

---

## 2. Provenance and Preparation

### Raw Data Source

Our data comes from the **NOAA Storm Events Database** ([ncei.noaa.gov/stormevents](https://www.ncei.noaa.gov/stormevents/ftp.jsp)), which records all significant U.S. weather events as documented by the National Weather Service (NWS). It contains three tables per year — *details* (event attributes), *fatalities* (death records), and *locations* (geographic coordinates) — covering 48 standardized event types. We use **2010–2025**, totaling ~967,000 storm events. NOAA collects this data for climatological reference, insurance, and public safety, compiled from NWS forecasters, trained spotters, law enforcement, and emergency managers.

### Data Preparation Pipeline

**Stage 1 — Loading and Joining.** We load annual CSVs for details, fatalities, and locations, joining on `EVENT_ID`. Fatality counts are aggregated per event; damage strings (e.g., `"25K"`, `"1.5M"`) are parsed into numeric USD values.

**Stage 2 — Cascade Identification.** We define a *cascade* as a primary event causally triggering secondary events, identified using five simultaneous constraints: (1) the secondary event starts within **7 days** of the primary ending; (2) both events occur in the **same county** (FIPS match); (3) both share the same NOAA **`EPISODE_ID`**; (4) primary and secondary are **different event types**; and (5) only **domain-validated causal pairs** from peer-reviewed literature are accepted (100+ pairs across 35+ event types, sourced from NOAA NWS, USGS, FEMA, IPCC AR6, Cannon et al. 2008, and Gill & Malamud 2014). Requiring shared episode IDs is our strongest filter against spurious co-occurrences, though it depends on NWS forecasters' judgment in grouping events.

**Stage 3 — Feature Engineering.** We engineer ~165 features: temporal features (month, hour, cyclical encodings, duration), impact/severity features (parsed USD damage, log-damage, injuries, deaths, composite severity score), event details (magnitude, severe wind/hail flags, flood cause), spatial features (coordinates, Haversine path length, tornado footprint), historical windowed features (7/30-day rolling event counts, cumulative damage), and ~120 aggregate statistics learned from training data only — including location/state cascade rates and conditional probabilities P(Secondary | Primary).

**Stage 4 — Splitting.** We use a **chronological 80/20 split**, sorting by `BEGIN_DATETIME` so the test set is strictly more recent than training data. This avoids temporal leakage and mirrors real-world deployment. Labels with <10 positive training samples and zero-variance features are removed.

**Stage 5 — Validation Split and Subsampling.** At training time, both notebooks carve out a validation set from the **final 10%** of the chronologically-sorted training data, again preserving temporal ordering. Both apply a minimum prevalence filter (`MIN_POS_FRAC = 0.0005`) to drop labels with too few positives. To address extreme class imbalance (~95% of events trigger no cascade), the XGBoost notebook applies **per-label negative subsampling** — for each label, it retains all positives and samples at most 5× negatives. The sklearn notebook applies an analogous **global negative subsampling** — retaining all rows with any positive label and sampling negative-only rows to ~50K total.

---

## 3. Potential Biases

**Geographic and reporting bias.** NOAA data reflects *reported* events, which depends on population density and spotter network coverage. Rural and underserved areas likely have under-reported events, making some states appear disproportionately cascade-prone simply due to better detection.

**Temporal non-stationarity.** Climate change is altering event frequencies and cascade dynamics. Our 2010–2025 window captures recent trends but may not generalize to future climate regimes (e.g., increasing wildfire-to-debris-flow cascades).

**Label bias from domain constraints.** Restricting to literature-backed causal pairs biases the dataset toward well-studied cascades (e.g., Hurricane→Flood) and against emerging or under-studied ones. The `require_same_episode` filter additionally depends on NWS forecasters' consistency in grouping events into episodes.

**Class imbalance.** ~95% of events trigger no cascade. Among those that do, common secondary types (Thunderstorm Wind, Flash Flood) vastly outnumber rare ones (Debris Flow, Wildfire). We mitigate this through negative subsampling and class-weighted loss functions, but rare-label performance remains challenging.

---

## 4. Alternative Use Cases

This dataset could support several other tasks: **binary cascade detection** (predict whether *any* cascade occurs), **cascade severity regression** (predict total damage from cascade sequences), **temporal point process modeling** (estimate the timing and rate of secondary events), and **spatial propagation modeling** (model cross-county cascade spread as graph diffusion). We selected multilabel cascade-type prediction because it provides the most actionable output — knowing *which* secondary events to expect enables targeted emergency preparation.

---

## 5. Related Work and Alternative Data Sources

**FEMA Disaster Declarations Database:** Contains federally declared disasters with damage assessments. We considered it for validation but not as a primary source — declarations are administrative events influenced by political processes, lacking the granular event-level timing needed for cascade identification.

**ERA5 Reanalysis Data (ECMWF):** Gridded atmospheric variables at hourly resolution. Could provide rich environmental context, but its 0.25° resolution does not align well with county-level records, and our engineered features (conditional cascade probabilities, historical statistics) already serve as data-driven proxies for atmospheric drivers.

**EM-DAT International Disasters Database:** Covers global events but only records large-scale disasters (≥10 deaths or ≥100 affected), missing the moderate events that form most cascade chains. Its coarser temporal resolution also makes cascade identification infeasible.

**ShakeAlert / USGS Earthquake Data:** Could enable earthquake-triggered cascades (e.g., Earthquake→Landslide), but earthquakes fall outside the NOAA taxonomy, and cross-agency data integration would add complexity without proportional gain.

We selected NOAA Storm Events because it offers standardized event types, precise timestamps, the unique `EPISODE_ID` for causal grouping, sufficient volume (~967K events) for ML, and county-level FIPS codes for integration with census/vulnerability data.

### Modeling Approaches in Context

**XGBoost for hazard prediction.** Gradient-boosted trees, particularly XGBoost, have become the dominant approach for individual hazard prediction — landslide susceptibility, flood forecasting, and wildfire risk — consistently outperforming Random Forest, SVM, and logistic regression on tabular environmental data. However, most prior work treats each hazard as an independent binary classification task, ignoring inter-hazard dependencies that are central to cascading disasters.

**Classifier chains for label dependencies.** Read et al. (2011) introduced classifier chains as a multilabel method that captures label correlations by chaining binary classifiers, where each classifier receives previous labels' predictions as additional features. This is particularly relevant for cascade prediction, where label dependencies are *causal* (e.g., predicting Flash Flood increases the probability of Debris Flow). Ensembles of chains with randomized orderings further improve robustness.

**Our approach.** We combine both paradigms: per-label XGBoost (binary relevance) serves as our strongest baseline, leveraging XGBoost's proven superiority on tabular hazard data, while classifier chains with logistic regression and random forest explicitly model the inter-label dependencies that define cascading disasters. This allows us to empirically test whether modeling cascade structure (chains) improves upon treating each secondary event independently (binary relevance).

---

*Document prepared for 10-718 Machine Learning in Practice, Spring 2026.*
