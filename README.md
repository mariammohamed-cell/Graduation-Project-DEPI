### BUSINESS OBJECTIVE

---

Reduce road accident fatalities and injuries by predicting accident severity and identifying the key risk factors related to location, time, and environmental conditions. The predicted severity levels can help authorities implement preventive measures and allocate resources efficiently.

### TARGETS

---

* **Accident Severity Prediction:** Predict accident severity (Slight, Serious, Fatal) and identify key contributing factors (speed limit, road type, weather, lighting, time).
* **Hotspot Prediction:** Detect high-risk vs. low-risk locations to support better resource allocation.
* **Time-based Risk:** Assess accident severity risk depending on the day of the week and time of day.
* **Environmental Impact:** Understand how weather, lighting, and road surface conditions contribute to accident severity.

### BUSINESS IMPACT / RECOMMENDATIONS

---

* Increase police/traffic presence in hotspots and peak hours.
* Enforce speed monitoring in rural and high-speed roads.
* Improve street lighting in areas with frequent night accidents.
* Implement preventive measures in adverse weather (e.g., electronic warnings, temporary speed limits).
* Add speed bumps or traffic signals at accident-prone junctions.
  *These recommendations are based on predictive insights and aim to reduce accident severity and frequency.*

### COLUMN DESCRIPTION

---

* **Accident\_Index (Text):** Unique identifier for each accident record.
* **Location\_Easting\_OSGR (Numeric):** Easting coordinate in the Ordnance Survey grid reference system.
* **Location\_Northing\_OSGR (Numeric):** Northing coordinate in the Ordnance Survey grid reference system.
* **Longitude (Float):** Geographic longitude of the accident location.
* **Latitude (Float):** Geographic latitude of the accident location.
* **Police\_Force (Categorical):** Code representing the police force that reported the accident.
* **Accident\_Severity (Categorical):** Level of accident severity (Slight, Serious, Fatal).
* **Number\_of\_Vehicles (Numeric):** Number of vehicles involved in the accident.
* **Number\_of\_Casualties (Numeric):** Number of casualties recorded in the accident.
* **Date (Date):** Date of accident occurrence.
* **Day\_of\_Week (Categorical):** Day of the week when the accident occurred.
* **Time (Time):** Time of accident occurrence.
* **Local\_Authority\_(District) (Categorical):** District-level local authority code for the accident location.
* **Local\_Authority\_(Highway) (Categorical):** Highway authority code for the accident location.
* **1st\_Road\_Class (Categorical):** Classification of the first road (e.g., Motorway, A, B, C, Unclassified).
* **1st\_Road\_Number (Numeric):** Road number of the first road where the accident occurred.
* **Road\_Type (Categorical):** Type of road (single carriageway, dual carriageway, roundabout, etc.).
* **Speed\_limit (Numeric):** Legal speed limit (in mph) at the accident location.
* **Junction\_Detail (Categorical):** Details of the junction (T-junction, crossroads, roundabout, etc.).
* **Junction\_Control (Categorical):** Type of junction control (e.g., traffic signal, authorized person, uncontrolled).
* **2nd\_Road\_Class (Categorical):** Classification of the second road at the junction (if applicable).
* **2nd\_Road\_Number (Numeric):** Road number of the second road at the junction (if applicable).
* **Pedestrian\_Crossing-Human\_Control (Categorical):** Whether a pedestrian crossing was controlled by a human.
* **Pedestrian\_Crossing-Physical\_Facilities (Categorical):** Type of physical pedestrian crossing (zebra, pelican, puffin, etc.).
* **Light\_Conditions (Categorical):** Lighting conditions at the time of the accident (daylight, darkness with/without lighting).
* **Weather\_Conditions (Categorical):** Weather conditions at the time of the accident (fine, rain, fog, snow, etc.).
* **Road\_Surface\_Conditions (Categorical):** Road surface condition (dry, wet, snow, frost/ice, etc.).
* **Special\_Conditions\_at\_Site (Categorical):** Any special conditions at the accident site (roadworks, diversion, etc.).
* **Carriageway\_Hazards (Categorical):** Hazards present on the carriageway (e.g., oil, mud, object on road).
* **Urban\_or\_Rural\_Area (Categorical):** Indicates whether the accident occurred in an urban or rural area.
* **Did\_Police\_Officer\_Attend\_Scene\_of\_Accident (Categorical):** Whether a police officer attended the accident scene.
* **LSOA\_of\_Accident\_Location (Categorical):** Lower Layer Super Output Area code (small geographic area identifier).
* **Year (Numeric):** Year in which the accident occurred.

### DATASET OVERVIEW

---

This dataset contains detailed records of road traffic accidents in the United Kingdom, collected by the police using the STATS19 accident reporting system. Each record represents a single accident event, capturing its location, timing, environmental conditions, road characteristics, vehicles involved, and casualties.

**Dataset Source & Structure**

---

* **Dataset Name:** UK Road Traffic Accidents
* **Source:** \[Kaggle]
* **Total Records:** \~1.6 million accident records
* **Files Included:**

  * 2005–2007: Accidents during this 3-year period
  * 2009–2011: Accidents during this 3-year period
  * 2012–2014: Accidents during this 3-year period

**Dataset Coverage**

---

* **Geographical Coverage:** Entire United Kingdom (urban and rural areas, all road types)
* **Temporal Coverage:** Multi-year, continuous annual records (2005–2014, with gap for 2008)
* **Granularity:** One row = one accident
